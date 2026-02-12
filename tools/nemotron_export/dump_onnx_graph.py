"""Dump ONNX model graphs to text files for diffing."""
import onnx
import os
from google.protobuf import text_format


def dump_graph(model_path, output_path):
    model = onnx.load(model_path)
    # Strip raw weight data to keep output readable
    for init in model.graph.initializer:
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.ClearField("int32_data")
        init.ClearField("int64_data")
        init.ClearField("double_data")
        del init.external_data[:]
    with open(output_path, "w") as f:
        f.write(text_format.MessageToString(model.graph))
    print(f"Written: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model1 = os.path.join(script_dir, "onnx_models", "encoder.onnx")
    model2 = os.path.join(script_dir, "onnx_models_st", "encoder.onnx")
    out1 = os.path.join(script_dir, "encoder_graph_1.txt")
    out2 = os.path.join(script_dir, "encoder_graph_2.txt")

    dump_graph(model1, out1)
    dump_graph(model2, out2)
    print(f"\nYou can now diff the outputs:\n  diff {out1} {out2}")
