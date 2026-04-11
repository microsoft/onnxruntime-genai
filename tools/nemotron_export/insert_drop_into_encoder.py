#!/usr/bin/env python3
"""
ONNX graph surgery: Insert dynamic drop_extra_pre_encoded into Nemotron encoder.

Modifies the ONNX encoder graph to accept a `drop_count` input that controls
how many post-conv-subsampling frames to drop before the transformer layers.
This matches NeMo's runtime behavior:
  - First chunk:      drop_count = [0]  (no cache overlap)
  - Subsequent chunks: drop_count = [2]  (skip redundant cache-overlap frames)

Background:
  NeMo's conformer encoder has a `drop_extra_pre_encoded` step between the
  convolutional subsampling (pre_encode) and the transformer layers:

      mel → conv_subsampling → DROP → transformer → output

  For subsequent chunks, the pre-encode cache (9 mel frames) causes the conv
  to output 9 encoded frames instead of 7.  Two of those frames overlap with
  the previous chunk's output.  NeMo drops them so the transformer always sees
  7 new frames — matching the attention pattern it was trained with.

  The ONNX export sets drop=0 so the graph contains no Slice.  This script
  re-inserts a dynamic Slice controlled by a new input, so the C++ runtime
  can pass the correct drop value per chunk.

Graph changes:
  1. New input:  `drop_count` (int64, shape [1])
  2. Slice on pre_encode output: `linear_sliced = linear[:, drop:, :]`
  3. Adjust max_audio_length (Chain 1, shape-based): `add_197 -= drop`
  4. Adjust encoded_lengths (Chain 2, value-based): `encoded_lengths -= drop`
  5. All downstream ops (attention masks, positional encoding, caches) adapt
     automatically because both chains use the adjusted values.

Usage:
    python insert_drop_into_encoder.py \\
        --input  onnx_models_test_3_opt/encoder.onnx \\
        --output onnx_models_test_3_opt_drop/encoder.onnx
"""

import argparse
import os
import shutil
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def find_node_by_output(graph, output_name):
    """Find the graph node that produces the given output tensor name."""
    for i, node in enumerate(graph.node):
        if output_name in node.output:
            return i, node
    return None, None


def find_insertion_index(graph, after_output_name):
    """Return the index right after the node that produces after_output_name."""
    idx, _ = find_node_by_output(graph, after_output_name)
    if idx is None:
        raise ValueError(f"Node producing '{after_output_name}' not found in graph")
    return idx + 1


def rename_output(graph, old_name, new_name):
    """Rename a node output (producer side only, not consumers)."""
    for node in graph.node:
        for i, out in enumerate(node.output):
            if out == old_name:
                node.output[i] = new_name
                return True
    return False


def replace_input_in_node(node, old_name, new_name):
    """Replace a specific input name in a node."""
    for i, inp in enumerate(node.input):
        if inp == old_name:
            node.input[i] = new_name
            return True
    return False


def verify_graph_structure(graph):
    """Verify expected nodes/tensors exist before surgery."""
    required = ['linear', 'add_197', 'encoded_lengths', 'val_88', 'add_269',
                'val_279', 'floordiv_26']
    for name in required:
        idx, node = find_node_by_output(graph, name)
        if node is None:
            raise ValueError(
                f"Expected node producing '{name}' not found. "
                f"This script is designed for the Nemotron encoder exported with "
                f"drop_extra_pre_encoded=0 via export_nemotron_to_onnx_1.py."
            )

    # Count val_88 consumer types
    pre_drop_outputs = {'val_89', 'val_94', 'val_100', 'val_117'}
    n_pre = n_post = n_cache = 0
    for node in graph.node:
        if 'val_88' in node.input:
            out = node.output[0] if node.output else ''
            if out in pre_drop_outputs:
                n_pre += 1
            elif node.op_type == 'Slice':
                n_cache += 1
            else:
                n_post += 1
    print(f"  val_88 consumers: {n_pre} pre-drop, {n_post} post-drop reshape, {n_cache} cache slices")

    n_add269 = sum(1 for n in graph.node if 'add_269' in n.input)
    print(f"  add_269 consumers: {n_add269}")

    n_val279 = sum(1 for n in graph.node if 'val_279' in n.input)
    print(f"  val_279 consumers: {n_val279} (pos enc dim from floordiv_26)")


def insert_drop(model):
    """Insert dynamic drop_count logic into the encoder graph.

    Key insight: `add_197` (= conv subsampled length from audio_signal.shape)
    and `val_88` (= Reshape(add_197, [-1])) are shared between:

      PRE-DROP uses (conv masks in pre_encode — must keep original value):
        - val_89, val_94, val_100 (mask expand shapes)
        - val_117 (pre_encode linear reshape)

      POST-DROP uses (transformer — must use adjusted value):
        - val_299, val_280, val_263 (transformer reshape targets)
        - 24 cache Slice nodes (cache_keep_size)
        - add_269 (positional encoding + attention mask)
        - add_6620 (cache_last_channel_len update)

    We create parallel post-drop values (_pd_*) and selectively rewire only
    the transformer consumers.
    """
    graph = model.graph

    print("Verifying graph structure...")
    verify_graph_structure(graph)

    # ================================================================== #
    #  Step 1: Add the new input  `drop_count`  (int64, shape [1])
    # ================================================================== #
    drop_input = helper.make_tensor_value_info(
        'drop_count', TensorProto.INT64, [1]
    )
    graph.input.append(drop_input)
    print("Added input: drop_count (int64, [1])")

    # ================================================================== #
    #  Step 2: Add constant initializers
    # ================================================================== #
    consts = {
        '_drop_axes': np.array([1], dtype=np.int64),
        '_drop_end': np.array([np.iinfo(np.int64).max], dtype=np.int64),
        '_drop_squeeze_axes': np.array([0], dtype=np.int64),
        '_drop_zero': np.array(0, dtype=np.int64),
        '_drop_reshape_1d': np.array([-1], dtype=np.int64),
    }
    # val_3421 = 70 (last_channel_cache_size) already exists as initializer
    for name, arr in consts.items():
        graph.initializer.append(numpy_helper.from_array(arr, name=name))

    # ================================================================== #
    #  Step 3: Create post-drop value chain
    #    _pd_scalar     = Squeeze(drop_count)            scalar int64
    #    _pd_add_197    = Clip(add_197 - _pd_scalar, 0)  scalar int64
    #    _pd_val_88     = Reshape(_pd_add_197, [-1])      [1] int64
    #    _pd_add_269    = _pd_add_197 + 70                scalar int64
    # ================================================================== #
    new_nodes = [
        helper.make_node('Squeeze',
                         ['drop_count', '_drop_squeeze_axes'],
                         ['_pd_scalar'], name='_pd_squeeze'),
        helper.make_node('Sub',
                         ['add_197', '_pd_scalar'],
                         ['_pd_add_197_raw'], name='_pd_sub_197'),
        helper.make_node('Clip',
                         ['_pd_add_197_raw', '_drop_zero', ''],
                         ['_pd_add_197'], name='_pd_clip_197'),
        helper.make_node('Reshape',
                         ['_pd_add_197', '_drop_reshape_1d'],
                         ['_pd_val_88'], name='_pd_reshape_88'),
        helper.make_node('Add',
                         ['_pd_add_197', 'val_3421'],
                         ['_pd_add_269'], name='_pd_add_269'),
        # Post-drop positional encoding dimension:
        # Original: val_279 = 141 + 2*floordiv_26 = 2*add_269 - 1
        # floordiv_26 is a Chain 1 intermediate, so replacing add_269 consumers
        # doesn't fix val_279.  Compute _pd_val_279 = 2*_pd_add_269 - 1.
        helper.make_node('Mul',
                         ['val_2', '_pd_add_269'],
                         ['_pd_2x_add_269'], name='_pd_mul_2x'),
        helper.make_node('Sub',
                         ['_pd_2x_add_269', 'val_7'],
                         ['_pd_add_468'], name='_pd_sub_468'),
        helper.make_node('Reshape',
                         ['_pd_add_468', '_drop_reshape_1d'],
                         ['_pd_val_279'], name='_pd_reshape_279'),
    ]
    print("Created post-drop chain: _pd_add_197, _pd_val_88, _pd_add_269, _pd_val_279")

    # ================================================================== #
    #  Step 4: Slice the pre_encode output (the tensor)
    #    linear_pre_drop[:, drop_count:, :] → linear
    # ================================================================== #
    rename_output(graph, 'linear', 'linear_pre_drop')
    new_nodes.append(
        helper.make_node('Slice',
                         ['linear_pre_drop', 'drop_count', '_drop_end', '_drop_axes'],
                         ['linear'], name='_pd_slice_linear')
    )
    print("Created Slice on 'linear' (pre_encode output)")

    # ================================================================== #
    #  Step 5: Adjust encoded_lengths output (Chain 2, value-based)
    # ================================================================== #
    rename_output(graph, 'encoded_lengths', 'encoded_lengths_pre_drop')
    new_nodes.extend([
        helper.make_node('Sub',
                         ['encoded_lengths_pre_drop', 'drop_count'],
                         ['_pd_enc_len_raw'], name='_pd_sub_enc_len'),
        helper.make_node('Clip',
                         ['_pd_enc_len_raw', '_drop_zero', ''],
                         ['encoded_lengths'], name='_pd_clip_enc_len'),
    ])
    print("Created Sub+Clip on 'encoded_lengths'")

    # 5b: Fix conv mask — unsqueeze_13 feeds the pre_encode conv mask (lt_3)
    #     and must use the PRE-DROP length, not the post-drop encoded_lengths.
    #     After renaming, unsqueeze_13 now consumes the new (post-drop)
    #     'encoded_lengths'. We need to rewire it back to 'encoded_lengths_pre_drop'.
    fixed_conv_mask = 0
    for node in graph.node:
        if node.name == 'unsqueeze_13' or (
            node.op_type == 'Unsqueeze' and 'unsqueeze_13' in node.output
        ):
            if replace_input_in_node(node, 'encoded_lengths', 'encoded_lengths_pre_drop'):
                fixed_conv_mask += 1
    # Also search by output name pattern if not found by node name
    if fixed_conv_mask == 0:
        for node in graph.node:
            if (node.op_type == 'Unsqueeze'
                    and 'encoded_lengths' in node.input
                    and any('unsqueeze_13' in o or 'unsqueeze_13' == o for o in node.output)):
                if replace_input_in_node(node, 'encoded_lengths', 'encoded_lengths_pre_drop'):
                    fixed_conv_mask += 1
    # Fallback: find the Unsqueeze that feeds lt_3 (the Less node for conv mask)
    if fixed_conv_mask == 0:
        lt_inputs = set()
        for node in graph.node:
            if node.op_type == 'Less' and any('lt_3' == o or 'lt_' in o for o in node.output):
                lt_inputs.update(node.input)
        for node in graph.node:
            if (node.op_type == 'Unsqueeze'
                    and 'encoded_lengths' in node.input
                    and any(o in lt_inputs for o in node.output)):
                if replace_input_in_node(node, 'encoded_lengths', 'encoded_lengths_pre_drop'):
                    fixed_conv_mask += 1
    print(f"  Fixed conv mask (unsqueeze_13): {fixed_conv_mask} node(s) → use pre-drop length")

    # ================================================================== #
    #  Step 6: Rewire transformer consumers to use post-drop values
    # ================================================================== #

    # 6a: Replace val_88 → _pd_val_88 in POST-DROP consumers
    #     POST-DROP: val_299, val_280, val_263 (reshape Concats) + 24 cache Slices
    pre_drop_concat_outputs = {'val_89', 'val_94', 'val_100', 'val_117'}
    replaced_val88 = 0
    for node in graph.node:
        if 'val_88' not in node.input:
            continue
        out = node.output[0] if node.output else ''
        is_pre_drop = out in pre_drop_concat_outputs
        if not is_pre_drop:
            replace_input_in_node(node, 'val_88', '_pd_val_88')
            replaced_val88 += 1
    print(f"  Replaced val_88 → _pd_val_88 in {replaced_val88} nodes")

    # 6b: Replace add_269 → _pd_add_269 in ALL consumers
    #     (all add_269 consumers are transformer/attention related)
    replaced_add269 = 0
    for node in graph.node:
        if 'add_269' in node.input:
            replace_input_in_node(node, 'add_269', '_pd_add_269')
            replaced_add269 += 1
    print(f"  Replaced add_269 → _pd_add_269 in {replaced_add269} nodes")

    # 6c: Replace add_197 → _pd_add_197 in add_6620 (cache len update)
    #     add_6620 = cache_last_channel_len + add_197
    replaced_6620 = 0
    for node in graph.node:
        if 'add_6620' in node.output and 'add_197' in node.input:
            replace_input_in_node(node, 'add_197', '_pd_add_197')
            replaced_6620 += 1
    print(f"  Replaced add_197 → _pd_add_197 in add_6620: {replaced_6620} node(s)")

    # 6d: Replace val_279 → _pd_val_279 in ALL consumers
    #     val_279 = 2*add_269-1 (rel pos enc dim) computed via floordiv_26 (Chain 1 intermediate)
    #     Only consumer is val_280 Concat (attention head reshape)
    replaced_val279 = 0
    for node in graph.node:
        if 'val_279' in node.input:
            replace_input_in_node(node, 'val_279', '_pd_val_279')
            replaced_val279 += 1
    print(f"  Replaced val_279 → _pd_val_279 in {replaced_val279} nodes")

    # ================================================================== #
    #  Step 7: Insert new nodes at correct topological positions
    # ================================================================== #
    # All new nodes depend only on model inputs or pre-existing values
    # (add_197, val_3421, drop_count, linear_pre_drop, encoded_lengths_pre_drop).
    # Insert them right after 'linear_pre_drop' is produced, which is after
    # the entire pre_encode conv chain.
    idx = find_insertion_index(graph, 'linear_pre_drop')
    for i, node in enumerate(new_nodes):
        graph.node.insert(idx + i, node)

    print(f"Total nodes after surgery: {len(graph.node)}")
    return model


def validate_model(model):
    """Run ONNX checker on the modified model."""
    try:
        onnx.checker.check_model(model, full_check=False)
        print("ONNX checker passed (lightweight)")
    except Exception as e:
        print(f"WARNING: ONNX checker issue: {e}")


def run_inference_test(model_dir):
    """Quick sanity test: run encoder with drop_count=0 and drop_count=2."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available, skipping inference test")
        return

    encoder_path = os.path.join(model_dir, 'encoder.onnx')
    sess = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])

    input_names = [i.name for i in sess.get_inputs()]
    print(f"\nInference test — inputs: {input_names}")

    # Build dummy inputs matching the model's expected shapes/types
    feeds = {
        'audio_signal': np.random.randn(1, 65, 128).astype(np.float32),
        'length': np.array([65], dtype=np.int64),
        'cache_last_channel': np.zeros((1, 24, 70, 1024), dtype=np.float32),
        'cache_last_time': np.zeros((1, 24, 1024, 8), dtype=np.float32),
        'cache_last_channel_len': np.array([0], dtype=np.int64),
    }

    for drop_val in [0, 2]:
        feeds['drop_count'] = np.array([drop_val], dtype=np.int64)
        outputs = sess.run(None, feeds)
        enc_out = outputs[0]
        enc_len = outputs[1]
        print(f"  drop_count={drop_val}: "
              f"output shape={enc_out.shape}, "
              f"encoded_lengths={enc_len}")

    # Also test first-chunk size (49 mel frames)
    feeds_first = {
        'audio_signal': np.random.randn(1, 49, 128).astype(np.float32),
        'length': np.array([49], dtype=np.int64),
        'cache_last_channel': np.zeros((1, 24, 70, 1024), dtype=np.float32),
        'cache_last_time': np.zeros((1, 24, 1024, 8), dtype=np.float32),
        'cache_last_channel_len': np.array([0], dtype=np.int64),
        'drop_count': np.array([0], dtype=np.int64),
    }
    outputs = sess.run(None, feeds_first)
    print(f"  first chunk (49 mel, drop=0): "
          f"output shape={outputs[0].shape}, "
          f"encoded_lengths={outputs[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Insert dynamic drop_count into Nemotron ONNX encoder"
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to the input encoder.onnx (with external data alongside)'
    )
    parser.add_argument(
        '--output', required=True,
        help='Path to the output encoder.onnx'
    )
    parser.add_argument(
        '--skip-test', action='store_true',
        help='Skip the onnxruntime inference test'
    )
    args = parser.parse_args()

    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_dir = os.path.dirname(os.path.abspath(args.output))

    print(f"Loading model from: {args.input}")
    model = onnx.load(args.input)

    model = insert_drop(model)
    validate_model(model)

    # Create output directory and copy external data files
    os.makedirs(output_dir, exist_ok=True)
    if os.path.abspath(input_dir) != os.path.abspath(output_dir):
        skip = {'encoder.onnx', 'encoder.onnx.data'}
        for fname in os.listdir(input_dir):
            if fname in skip:
                continue
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"Copied {fname} to output directory")

    print(f"\nSaving modified model to: {args.output}")
    # Save with external data to handle >2GB models
    onnx.save(
        model,
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(args.output) + '.data',
    )
    print("Done!")

    if not args.skip_test:
        run_inference_test(output_dir)


if __name__ == '__main__':
    main()
