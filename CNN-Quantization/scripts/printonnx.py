import onnx

def print_node_names(model_path):
    model = onnx.load(model_path)
    print("All ONNX Nodes:")
    for node in model.graph.node:
        print(node.name)

# 运行这个函数，检查 `ONNX` 里真实的 `node.name`
print_node_names("/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/quantized_model.onnx")
