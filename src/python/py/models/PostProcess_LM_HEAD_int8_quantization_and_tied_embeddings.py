import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
import argparse
import tqdm
import os

def convert_gather_to_use_lm_head_weights(model, graph, quant_weight_name, scales_name, zero_points_name, use_zero_points, hidden_size):
    """
    Replace the embed_tokens/Gather with operations that reuse the quantized lm_head weights
    """
    # Find the Gather node for embeddings
    gather_node = None
    for node in graph.node:
        if node.name == "/model/embed_tokens/Gather":
            gather_node = node
            break

    if gather_node is None:
        print("Warning: /model/embed_tokens/Gather not found, skipping weight tying optimization")
        return

    # Save the original inputs and outputs of the Gather node
    embedding_weights_name = gather_node.input[0] 
    input_ids = gather_node.input[1]  # This is typically the input_ids tensor
    original_output = gather_node.output[0]
    
    # Create new nodes to replace the Gather operation
    
    # 1. Gather the quantized weights
    gathered_quant_weights = "gathered_quant_weights"
    gather_weights_node = helper.make_node(
        'Gather',
        inputs=[quant_weight_name, input_ids],
        outputs=[gathered_quant_weights],
        name='/model/embed_tokens/GatherQuantizedWeights',
        axis=0
    )
    
    # 2. Gather the scales
    gathered_scales_raw = "gathered_scales_raw"
    gather_scales_node = helper.make_node(
        'Gather',
        inputs=[scales_name, input_ids],
        outputs=[gathered_scales_raw],
        name='/model/embed_tokens/GatherScales',
        axis=0
    )

    # Reshape the scales to add an extra dimension for broadcasting
    unsqueeze_scales_node = helper.make_node(
        'Unsqueeze',
        inputs=[gathered_scales_raw, "scales_axes"],
        outputs=["gathered_scales"],
        name='/model/embed_tokens/UnsqueezeScales'
    )

    # Create axes tensor for unsqueeze operation (adding dimension at axis 2)
    scales_axes = np.array([3], dtype=np.int64)
    scales_axes_name = "scales_axes"
    scales_axes_initializer = numpy_helper.from_array(scales_axes, scales_axes_name)
    graph.initializer.extend([scales_axes_initializer])
    
    # Cast the quantized weights to floating point
    cast_weights_node = helper.make_node(
        'Cast',
        inputs=[gathered_quant_weights],
        outputs=["casted_quant_weights"],
        name='/model/embed_tokens/CastWeights',
        to=TensorProto.FLOAT16 if args.fp16 else TensorProto.FLOAT
    )
    
    # Create a constant for the zero point (128 for symmetric quantization)
    zero_point_const = np.array([128], dtype=np.uint8)
    zero_point_const_name = "zero_offset_const"
    zero_point_initializer = numpy_helper.from_array(zero_point_const, zero_point_const_name)
    graph.initializer.extend([zero_point_initializer])
    
    # Cast the zero point to the same type as weights
    cast_zp_node = helper.make_node(
        'Cast',
        inputs=[zero_point_const_name],
        outputs=["casted_zero_point"],
        name='/model/embed_tokens/CastZeroPoint',
        to=TensorProto.FLOAT16 if args.fp16 else TensorProto.FLOAT
    )
    
    # Subtract zero point from casted weights
    sub_node = helper.make_node(
        'Sub',
        inputs=["casted_quant_weights", "casted_zero_point"],
        outputs=["centered_weights"],
        name='/model/embed_tokens/SubtractZeroPoint'
    )
    
    # Multiply by scale
    dequantized_output = "dequantized_embeddings"
    mul_node = helper.make_node(
        'Mul',
        inputs=["centered_weights", "gathered_scales"],
        outputs=[dequantized_output],
        name='/model/embed_tokens/MultiplyByScale'
    )
    
    # 4. Reshape to the final embedding shape
    # Get token shape
    shape_node = helper.make_node(
        'Shape',
        inputs=[input_ids],
        outputs=["token_shape"],
        name='/model/embed_tokens/GetTokenShape'
    )
    
    # Add constant for hidden dimension
    const_hidden_size = np.array([hidden_size], dtype=np.int64)
    const_hidden_size_name = "const_hidden_size"
    hidden_size_initializer = numpy_helper.from_array(const_hidden_size, const_hidden_size_name)
    graph.initializer.extend([hidden_size_initializer])
    
    # Concat to get final shape
    concat_final_shape = helper.make_node(
        'Concat',
        inputs=["token_shape", const_hidden_size_name],
        outputs=["final_shape"],
        name='/model/embed_tokens/ConcatFinalShape',
        axis=0
    )
    
    # Final reshape to get the right output shape
    final_reshape_node = helper.make_node(
        'Reshape',
        inputs=[dequantized_output, "final_shape"],
        outputs=[original_output],
        name='/model/embed_tokens/FinalReshape'
    )
    
    # Find and remove the original Gather node
    for i, node in enumerate(graph.node):
        if node.name == gather_node.name:
            del graph.node[i]
            break
    
    # Remove the original embedding weights from initializers
    for i, initializer in enumerate(graph.initializer):
        if initializer.name == embedding_weights_name:
            print(f"Removing original embedding weights: {embedding_weights_name}")
            del graph.initializer[i]
            break
    
    # Add all new nodes to the graph
    new_nodes = [
        gather_weights_node,
        gather_scales_node,
        unsqueeze_scales_node,
        cast_weights_node,
        cast_zp_node,
        sub_node,
        mul_node,
        shape_node,
        concat_final_shape,
        final_reshape_node
    ]
    
    # Modify this part to handle asymmetric quantization if needed
    if use_zero_points:
        # Gather the zero points
        gathered_zero_points = "gathered_zero_points"
        gather_zero_points_node = helper.make_node(
            'Gather',
            inputs=[zero_points_name, input_ids],
            outputs=[gathered_zero_points],
            name='/model/embed_tokens/GatherZeroPoints',
            axis=0
        )
        
        # Unsqueeze zero points for broadcasting
        unsqueeze_zp_node = helper.make_node(
            'Unsqueeze',
            inputs=[gathered_zero_points, "scales_axes"],
            outputs=["unsqueezed_zero_points"],
            name='/model/embed_tokens/UnsqueezeZeroPoints'
        )
        
        # Cast zero points to float
        cast_gathered_zp_node = helper.make_node(
            'Cast',
            inputs=["unsqueezed_zero_points"],
            outputs=["casted_gathered_zero_point"],
            name='/model/embed_tokens/CastGatheredZeroPoint',
            to=TensorProto.FLOAT16 if args.fp16 else TensorProto.FLOAT
        )
        
        # Replace the standard zero_point subtraction with the gathered one
        sub_node.input[1] = "casted_gathered_zero_point"
        
        # Insert the new nodes
        new_nodes.insert(2, gather_zero_points_node)
        new_nodes.insert(3, unsqueeze_zp_node)
        new_nodes.insert(6, cast_gathered_zp_node)
    
    graph.node.extend(new_nodes)
    
    print("Successfully tied embedding weights to quantized LM head weights using Cast+Mul operations")

def convert_matmul_to_matmulnbits(model_path, output_path, block_size=128, use_zero_points=False, accuracy_level=None):
    # Load the ONNX model
    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph
    
    # Update the opset import to ensure MatMulNBits compatibility
    opset_version = 21
    for opset_import in model.opset_import:
        if opset_import.domain == "" or opset_import.domain == "ai.onnx":
            opset_import.version = opset_version
            print(f"Updated opset version to {opset_version}")
            break
    else:
        # If no default domain found, add a new opset import
        model.opset_import.extend([onnx.helper.make_opsetid("", opset_version)])
        print(f"Added opset version {opset_version}")

    # Find the MatMul node
    matmul_node = None
    for node in graph.node:
        if node.name == "/lm_head/MatMul" and node.op_type == "MatMul":
            matmul_node = node
            break

    if matmul_node is None:
        raise ValueError("/lm_head/MatMul node not found in the model")

    # Find the initializer corresponding to the MatMul node's weight (B)
    weight_initializer = None
    for initializer in graph.initializer:
        if initializer.name == matmul_node.input[1]:  # B is the second input
            weight_initializer = initializer
            break

    if weight_initializer is None:
        raise ValueError("Weight initializer for MatMul node not found")

    # Get the weight data and shape
    weight_data = numpy_helper.to_array(weight_initializer)
    original_shape = weight_data.shape
    print(f"Original weight shape: {original_shape}")
    
    if len(original_shape) != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {original_shape}")
    
    hidden_size = original_shape[0]  # Input dimension
    vocab_size = original_shape[1]   # Output dimension
    
    # For MatMulNBits, we need to transpose and reshape the weights
    # Original: [hidden_size, vocab_size]
    # Transposed: [vocab_size, hidden_size]
    # Blocked: [vocab_size, hidden_size/block_size, block_size]
    transposed_data = weight_data.transpose()
    
    if hidden_size % block_size != 0:
        raise ValueError(f"Hidden size {hidden_size} is not divisible by block size {block_size}")
    
    # Reshape to prepare for blocked format
    reshaped_data = transposed_data.reshape(vocab_size, hidden_size // block_size, block_size)
    
    # Prepare output arrays for quantization
    quantized_data = np.zeros(reshaped_data.shape, dtype=np.uint8)
    scales = np.zeros((vocab_size, hidden_size // block_size), dtype=np.float16 if args.fp16 else np.float32)
    zero_points = np.zeros((vocab_size, hidden_size // block_size), dtype=np.uint8)

    # Perform block-wise quantization with progress bar
    print("Performing block-wise quantization:")
    total_blocks = vocab_size * (hidden_size // block_size)
    progress_bar = tqdm.tqdm(total=total_blocks, unit="blocks")

    for i in range(vocab_size):
        for j in range(hidden_size // block_size):
            block = reshaped_data[i, j, :]
            block_min = np.min(block)
            block_max = np.max(block)

            # Compute scale and zero point
            if use_zero_points:
                # Asymmetric quantization
                scale = (block_max - block_min) / 255 if block_max > block_min else 1.0
                # Calculate zero point and ensure it's within uint8 range
                zero_point = np.round(128 - (block_max + block_min) / (2 * scale)) if scale > 0 else 128
                zero_point = np.clip(zero_point, 0, 255).astype(np.uint8)
            else:
                # Symmetric quantization
                abs_max = max(abs(block_min), abs(block_max))
                scale = abs_max / 128.0 if abs_max > 0 else 1.0
                zero_point = 128

            # Store scale and zero point
            scales[i, j] = scale
            zero_points[i, j] = zero_point

            # Quantize the block
            if scale > 0:
                quantized_block = np.clip(np.round(block / scale + zero_point), 0, 255).astype(np.uint8)
                quantized_data[i, j, :] = quantized_block

            progress_bar.update(1)

    progress_bar.close()

    # Create names for the new tensors
    quant_weight_name = weight_initializer.name + "_quantized"
    scales_name = weight_initializer.name + "_scales"
    zero_points_name = weight_initializer.name + "_zero_points"

    # Create new initializers
    quant_weight_initializer = numpy_helper.from_array(quantized_data, quant_weight_name)
    scales_initializer = numpy_helper.from_array(scales, scales_name)
    zero_points_initializer = numpy_helper.from_array(zero_points, zero_points_name)

    # Remove the original weight initializer
    for i, initializer in enumerate(graph.initializer):
        if initializer.name == weight_initializer.name:
            del graph.initializer[i]
            break

    # Add the new initializers
    graph.initializer.extend([quant_weight_initializer, scales_initializer])

    # Create the MatMulNBits node to replace the MatMul node
    matmulnbits_inputs = [
        matmul_node.input[0],  # A (input)
        quant_weight_name,     # B (quantized weights)
        scales_name            # scales
    ]
    
    # Add zero points if needed
    if use_zero_points:
        graph.initializer.extend([zero_points_initializer])
        matmulnbits_inputs.append(zero_points_name)
    
    matmulnbits_node = helper.make_node(
        "MatMulNBits",
        inputs=matmulnbits_inputs,
        outputs=matmul_node.output,
        name=matmul_node.name + "_Q8",
        domain="com.microsoft",
        bits=8,
        block_size=block_size,
        accuracy_level=accuracy_level,
        K = hidden_size,
        N = vocab_size,
    )
    
    # Find and remove the original MatMul node
    for i, node in enumerate(graph.node):
        if node.name == matmul_node.name:
            del graph.node[i]
            break

    # Add the new MatMulNBits node
    graph.node.append(matmulnbits_node)

    # If embedding weight tying is enabled, replace the embedding Gather
    if args.tie_embeddings:
        convert_gather_to_use_lm_head_weights(
            model, 
            graph, 
            quant_weight_name, 
            scales_name, 
            zero_points_name, 
            use_zero_points,
            hidden_size
        )

    # Save the modified model
    print(f"Saving model to {output_path}...")
    data_file = os.path.basename(output_path) + ".data"
    onnx.save(model, output_path, save_as_external_data=True, location=data_file)
    
    print(f"Model successfully converted with block size {block_size} and {'with' if use_zero_points else 'without'} zero points")
    if accuracy_level is not None:
        print(f"MatMulNBits accuracy_level set to {accuracy_level}")
    print(f"Saved to {output_path} with external data in {data_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MatMul to MatMulNBits with int8 quantization")
    parser.add_argument("model_path", type=str, help="Path to the input ONNX model")
    parser.add_argument("output_path", type=str, help="Path to save the modified ONNX model")
    parser.add_argument("--block_size", type=int, default=128,
                        choices=[16, 32, 64, 128, 256], help="Block size for quantization (default: 128)")
    parser.add_argument("--use_zero_points", action="store_true",
                        help="Use zero points in quantization (asymmetric quantization, default: False)")
    parser.add_argument("--accuracy_level", type=int, choices=[1, 2, 3, 4], default=0,
                        help="Accuracy level for the MatMulNBits operator (optional)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 for scales (default: False)")
    parser.add_argument("--tie_embeddings", action="store_true",
                    help="Tie embeddings to LM head weights (default: False)")
    args = parser.parse_args()

    convert_matmul_to_matmulnbits(
        args.model_path,
        args.output_path,
        args.block_size,
        args.use_zero_points,
        args.accuracy_level
    )