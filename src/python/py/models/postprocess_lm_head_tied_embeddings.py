# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import onnx
import numpy as np
from onnx import helper, numpy_helper, version_converter
from onnx.external_data_helper import load_external_data_for_model
import argparse
import os


def get_node_attribute(node: onnx.NodeProto, attribute_name: str):
    for attr in node.attribute:
        if attr.name == attribute_name:
            value = onnx.helper.get_attribute_value(attr)
            return value
    return None


def find_graph_input(graph, input_name):
    for input in graph.input:
        if input.name == input_name:
            return input
    return None


def find_graph_output(graph, output_name):
    for output in graph.output:
        if output.name == output_name:
            return output
    return None


def get_tensor_type_from_graph(graph, tensor_name: str):
    tensor_type_map = {obj.name: obj.type for obj in graph.value_info}

    if tensor_name in tensor_type_map:
        return tensor_type_map[tensor_name].tensor_type
    
    g_input = find_graph_input(graph, tensor_name)
    if g_input:
        return g_input.type.tensor_type

    g_output = find_graph_output(graph, tensor_name)
    if g_output:
        return g_output.type.tensor_type

    return None


def convert_gather_to_use_lm_head_weights_helper(graph):
    """
    Replace the embed_tokens/Gather with operations that reuse the quantized lm_head weights
    """
    matmul_node = None
    for node in graph.node:
        if node.name.startswith("/lm_head/MatMul"):
            if node.op_type == "MatMulNBits":
                matmul_node = node
                break
            else:
                print("/lm_head/MatMul node type is not MatMulNBits. Skipping weight tying optimization")
                return

    if matmul_node is None:
        print("/lm_head/MatMul node not found in the model. Skipping weight tying optimization")
        return

    # Inputs A and scale has the same type, but scale is in external data. So we can only get the type from A here.
    scale_value_type = get_tensor_type_from_graph(graph, matmul_node.input[0])
    if scale_value_type:
        scale_value_type = scale_value_type.elem_type
    else:
        raise ValueError("/lm_head/MatMul scale value type is None")

    hidden_size = get_node_attribute(matmul_node, "K")

    num_bits = get_node_attribute(matmul_node, "bits")
    if num_bits != 8:
        print("MatMulNBits node is not 8 bits. Skipping weight tying optimization")
        return

    use_zero_points = len(matmul_node.input) > 3
    quant_weight_name = matmul_node.input[1]  # B (quantized weights)
    scales_name = matmul_node.input[2]  # scales
    zero_points_name = matmul_node.input[3] if use_zero_points else None # zero_points

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
        to=scale_value_type
    )
    
    # Create a constant for the zero point (128 for symmetric quantization). We assume the /lm_head/MatMul node is 8 bits.
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
        to=scale_value_type
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
            to=scale_value_type
        )
        
        # Replace the standard zero_point subtraction with the gathered one
        sub_node.input[1] = "casted_gathered_zero_point"
        
        # Insert the new nodes
        new_nodes.insert(2, gather_zero_points_node)
        new_nodes.insert(3, unsqueeze_zp_node)
        new_nodes.insert(6, cast_gathered_zp_node)
    
    graph.node.extend(new_nodes)
    
    print("Successfully tied embedding weights to quantized LM head weights using Cast+Mul operations")


def convert_gather_to_use_lm_head_weights_helper_2(graph):
    """
    Replace the embed_tokens/Gather with operations that reuse the quantized lm_head weights
    """
    matmul_node = None
    for node in graph.node:
        if node.name.startswith("/lm_head/MatMul"):
            if node.op_type == "MatMulNBits":
                matmul_node = node
                break
            else:
                print("/lm_head/MatMul node type is not MatMulNBits. Skipping weight tying optimization")
                return

    if matmul_node is None:
        print("/lm_head/MatMul node not found in the model. Skipping weight tying optimization")
        return

    # Inputs A and scale has the same type, but scale is in external data. So we can only get the type from A here.
    scale_value_type = get_tensor_type_from_graph(graph, matmul_node.input[0])
    if scale_value_type:
        scale_value_type = scale_value_type.elem_type
    else:
        raise ValueError("/lm_head/MatMul scale value type is None")

    hidden_size = get_node_attribute(matmul_node, "K")
    block_size = get_node_attribute(matmul_node, "block_size")

    num_bits = get_node_attribute(matmul_node, "bits")
    if num_bits != 8:
        print("MatMulNBits node is not 8 bits. Skipping weight tying optimization")
        return

    use_zero_points = len(matmul_node.input) > 3
    quant_weight_name = matmul_node.input[1]  # B (quantized weights)
    scales_name = matmul_node.input[2]  # scales
    zero_points_name = matmul_node.input[3] if use_zero_points else None # zero_points

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
    gather_weights_shape_name = "gather_weights_shape"
    concat_final_shape = helper.make_node(
        'Concat',
        inputs=["token_shape", const_hidden_size_name],
        outputs=[gather_weights_shape_name],
        name='/model/embed_tokens/ConcatFinalShape',
        axis=0
    )
       
    # Final reshape to get the right output shape
    gather_weights_reshape_name = "gather_weights_reshape"
    gather_weights_reshape_node = helper.make_node(
        'Reshape',
        inputs=[gathered_quant_weights, gather_weights_shape_name],
        outputs=[gather_weights_reshape_name],
        name='/model/embed_tokens/GatherQuantizedWeightsReshape'
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

    # Create axes tensor for unsqueeze operation (adding dimension at axis 2)
    scales_axes = np.array([3], dtype=np.int64)
    scales_axes_name = "scales_axes"
    scales_axes_initializer = numpy_helper.from_array(scales_axes, scales_axes_name)
    graph.initializer.extend([scales_axes_initializer])
       
    # Create a constant for the zero point (128 for symmetric quantization). We assume the /lm_head/MatMul node is 8 bits.
    zero_point_const = np.array([128], dtype=np.uint8)
    zero_point_const_name = "zero_offset_const"
    zero_point_initializer = numpy_helper.from_array(zero_point_const, zero_point_const_name)
    graph.initializer.extend([zero_point_initializer])    
    
    # DequantizeLinear
    dequantized_node = helper.make_node(
        'DequantizeLinear',
        inputs=[gather_weights_reshape_name, gathered_scales_raw, zero_point_const_name],
        outputs=[original_output],
        name='/model/embed_tokens/DequantizeLinear',
        axis=-1,
        block_size=block_size
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
        gather_weights_reshape_node,
        shape_node,
        concat_final_shape,
        dequantized_node
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
               
        # Replace the standard zero_point subtraction with the gathered one
        dequantized_node.input[2] = gathered_zero_points
        
        # Insert the new nodes
        new_nodes.insert(2, gather_zero_points_node)
    
    graph.node.extend(new_nodes)
    
    print("Successfully tied embedding weights to quantized LM head weights using Cast+Mul operations")


def convert_gather_to_use_lm_head_weights(model_path, output_path):
    # Load the ONNX model
    print(f"Loading model from {model_path}...")
    model_name = "model.onnx"
    model = onnx.load(model_path + model_name, load_external_data=False)
    load_external_data_for_model(model, model_path)
     
    # If embedding weight tying is enabled, replace the embedding Gather
    convert_gather_to_use_lm_head_weights_helper(model.graph) 

    # Save the modified model
    print(f"Saving model to {output_path}...")
    data_file = os.path.basename(output_path) + model_name + ".data"
    onnx.save(model, output_path + model_name, save_as_external_data=True, location=data_file)

    print(f"Saved to {output_path} with external data in {data_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tie MatMulNBits with Gather for LM head weights")
    parser.add_argument("--input_path", type=str, help="Path to the input ONNX model")
    parser.add_argument("--output_path", type=str, help="Path to save the modified ONNX model")
    args = parser.parse_args()

    convert_gather_to_use_lm_head_weights(
        args.input_path,
        args.output_path
    )
