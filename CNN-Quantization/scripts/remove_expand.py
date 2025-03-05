# import onnx
# from onnx import helper, TensorProto, numpy_helper
# from onnx import shape_inference
# import onnxslim

# def remove_expand_nodes(model_path, output_path):
#     # Load the model
#     model = onnx.load(model_path)
#     graph = model.graph

#     # Find all Expand nodes
#     nodes_to_remove = []
#     for node in graph.node:
#         if node.op_type == "Expand":
#             nodes_to_remove.append(node)

#     # Remove the Expand nodes and connect the previous and next nodes
#     for node in nodes_to_remove:
#         input_name = node.input[0]  # Input to the Expand node
#         output_name = node.output[0]  # Output of the Expand node

#         # Find the nodes that use the output of the Expand node
#         for next_node in graph.node:
#             for idx, input_name_next_node in enumerate(next_node.input):
#                 if input_name_next_node == output_name:
#                     # Replace the Expand node's output with its input in the next node
#                     next_node.input[idx] = input_name

#         # Remove the Expand node from the graph
#         graph.node.remove(node)

#     # Infer shapes (optional, but recommended after modifying the graph)
#     model = shape_inference.infer_shapes(model)

#     # Apply ONNX Slim optimization
#     model = onnxslim.slim(model)

#     # Save the modified model
#     onnx.save(model, output_path)
#     print(f"Successfully slimmed and saved the model to {output_path}")

# # Usage
# model_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/quantized_model.onnx"  # Input ONNX model file
# output_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/onnx/quantized_model1.onnx"  # Output ONNX model file

# remove_expand_nodes(model_path, output_path)
import onnx
from onnx import shape_inference
import onnxslim

def remove_expand_nodes(model_path, output_path):
    """
    移除 ONNX 模型中的 Expand 节点，并正确连接输入输出。
    """
    model = onnx.load(model_path)
    graph = model.graph

    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == "Expand":
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        input_name = node.input[0]  # Expand 节点的输入
        output_name = node.output[0]  # Expand 节点的输出

        # 遍历其他节点，将 Expand 输出替换成它的输入
        for next_node in graph.node:
            for idx, input_name_next in enumerate(next_node.input):
                if input_name_next == output_name:
                    next_node.input[idx] = input_name

        graph.node.remove(node)  # 删除 Expand 节点

    # 重新推理形状
    model = shape_inference.infer_shapes(model)

    # 保存移除 Expand 之后的模型
    onnx.save(model, output_path)
    print(f"Successfully removed Expand nodes and saved to {output_path}")

# def split_model_at_node(model, split_node_name):
#     """
#     在 `split_node_name` 位置拆分 ONNX 模型：
#     - `model_before`: `split_node_name` 之前的部分
#     - `model_after`: `split_node_name` 以及之后的部分
#     """
#     graph = model.graph
#     before_nodes = []
#     after_nodes = []
#     found_split = False
#     split_inputs = []
#     split_outputs = []

#     for node in graph.node:
#         if node.name == split_node_name:
#             found_split = True
#             split_inputs = node.input
#             split_outputs = node.output
#             after_nodes.append(node)
#             continue

#         if found_split:
#             after_nodes.append(node)
#         else:
#             before_nodes.append(node)

#     if not found_split:
#         raise ValueError(f"Node {split_node_name} not found in model.")

#     model_before = onnx.helper.make_model(
#         onnx.helper.make_graph(
#             before_nodes,
#             "model_before",
#             graph.input,
#             [onnx.helper.make_tensor_value_info(split_outputs[0], onnx.TensorProto.FLOAT, None)],
#             graph.initializer
#         )
#     )

#     model_after = onnx.helper.make_model(
#         onnx.helper.make_graph(
#             after_nodes,
#             "model_after",
#             [onnx.helper.make_tensor_value_info(split_inputs[0], onnx.TensorProto.FLOAT, None)],
#             graph.output,
#             graph.initializer
#         )
#     )

#     return model_before, model_after

# def merge_models(model_before, model_after):
#     """
#     重新合并 `model_before` 和 `model_after`，确保模型结构完整。
#     """
#     before_graph = model_before.graph
#     after_graph = model_after.graph

#     # 连接两个子模型
#     after_graph.input[0].name = before_graph.output[0].name

#     # **解决 TypeError：将 `RepeatedCompositeContainer` 转换为 `list`**
#     merged_graph = onnx.helper.make_graph(
#         list(before_graph.node) + list(after_graph.node),  # **转换为 list 再合并**
#         "merged_model",
#         before_graph.input,
#         after_graph.output,
#         list(before_graph.initializer) + list(after_graph.initializer)  # **转换为 list 再合并**
#     )

#     return onnx.helper.make_model(merged_graph)

# def split_model_at_node(model, split_node_name):
#     graph = model.graph
#     before_nodes = []
#     after_nodes = []
#     found_split = False
#     split_inputs = []
#     split_outputs = []

#     for node in graph.node:
#         if node.name == split_node_name:
#             found_split = True
#             split_inputs = node.input  # 获取 `Transpose` 的输入
#             split_outputs = node.output
#             after_nodes.append(node)
#             continue

#         if found_split:
#             after_nodes.append(node)
#         else:
#             before_nodes.append(node)

#     if not found_split:
#         raise ValueError(f"Node {split_node_name} not found in model.")

#     # **获取真正的 before_graph 输出（应为 /model.23/Concat_5_output_0）**
#     before_output_name = split_inputs[0]  # `/model.23/Transpose` 的输入

#     print(f"[DEBUG] before_graph outputs should be: {before_output_name}")

#     before_output = onnx.helper.make_tensor_value_info(
#         before_output_name, onnx.TensorProto.FLOAT, None
#     )

#     model_before = onnx.helper.make_model(
#         onnx.helper.make_graph(
#             before_nodes,
#             "model_before",
#             graph.input,
#             [before_output],  # 修正 before_graph 的输出
#             graph.initializer
#         )
#     )

#     model_after = onnx.helper.make_model(
#         onnx.helper.make_graph(
#             after_nodes,
#             "model_after",
#             [onnx.helper.make_tensor_value_info(split_outputs[0], onnx.TensorProto.FLOAT, None)],  # after_graph 仍然以 `Transpose` 的输出作为输入
#             graph.output,
#             graph.initializer
#         )
#     )

#     return model_before, model_after



# def merge_models(model_before, model_after):
#     before_graph = model_before.graph
#     after_graph = model_after.graph

#     if len(before_graph.output) == 0:
#         raise ValueError("Error: before_graph has no output!")
#     if len(after_graph.input) == 0:
#         raise ValueError("Error: after_graph has no input!")

#     # **调试信息**
#     print(f"[DEBUG] before_graph outputs: {[o.name for o in before_graph.output]}")
#     print(f"[DEBUG] after_graph inputs before fix: {[i.name for i in after_graph.input]}")

#     # **修正 after_graph 的输入**
#     connection_tensor_name = before_graph.output[0].name  # `/model.23/Concat_5_output_0`
#     after_graph.input[0].name = connection_tensor_name  # 让 `after_graph` 直接连接 `before_graph`

#     print(f"[DEBUG] after_graph inputs after fix: {[i.name for i in after_graph.input]}")
#     print(f"[DEBUG] Merging: {connection_tensor_name} -> {after_graph.input[0].name}")

#     # **合并模型**
#     merged_graph = onnx.helper.make_graph(
#         list(before_graph.node) + list(after_graph.node),
#         "merged_model",
#         before_graph.input,
#         after_graph.output,
#         list(before_graph.initializer) + list(after_graph.initializer)
#     )

#     return onnx.helper.make_model(merged_graph)

import onnx

def split_model_at_node(model, split_node_name, target_opset=13):
    graph = model.graph
    before_nodes = []
    after_nodes = []
    found_split = False
    split_inputs = []
    split_outputs = []

    for node in graph.node:
        if node.name == split_node_name:
            found_split = True
            split_inputs = node.input  # 获取 `Transpose` 的输入
            split_outputs = node.output
            after_nodes.append(node)
            continue

        if found_split:
            after_nodes.append(node)
        else:
            before_nodes.append(node)

    if not found_split:
        raise ValueError(f"Node {split_node_name} not found in model.")

    # **获取真正的 before_graph 输出**
    before_output_name = split_inputs[0]  # `/model.23/Transpose` 的输入

    print(f"[DEBUG] before_graph outputs should be: {before_output_name}")

    before_output = onnx.helper.make_tensor_value_info(
        before_output_name, onnx.TensorProto.FLOAT, None
    )

    # **创建 `model_before`，强制 opset=13**
    model_before = onnx.helper.make_model(
        onnx.helper.make_graph(
            before_nodes,
            "model_before",
            graph.input,
            [before_output],  # 修正 before_graph 的输出
            graph.initializer
        ),
        opset_imports=[onnx.helper.make_opsetid("", target_opset)]
    )

    # **创建 `model_after`，强制 opset=13**
    model_after = onnx.helper.make_model(
        onnx.helper.make_graph(
            after_nodes,
            "model_after",
            [onnx.helper.make_tensor_value_info(split_outputs[0], onnx.TensorProto.FLOAT, None)],  
            graph.output,
            graph.initializer
        ),
        opset_imports=[onnx.helper.make_opsetid("", target_opset)]
    )

    return model_before, model_after


def merge_models(model_before, model_after, target_opset=13):
    before_graph = model_before.graph
    after_graph = model_after.graph

    if len(before_graph.output) == 0:
        raise ValueError("Error: before_graph has no output!")
    if len(after_graph.input) == 0:
        raise ValueError("Error: after_graph has no input!")

    # **调试信息**
    print(f"[DEBUG] before_graph outputs: {[o.name for o in before_graph.output]}")
    print(f"[DEBUG] after_graph inputs before fix: {[i.name for i in after_graph.input]}")

    # **修正 after_graph 的输入**
    connection_tensor_name = before_graph.output[0].name  # `/model.23/Concat_5_output_0`
    after_graph.input[0].name = connection_tensor_name  # 让 `after_graph` 直接连接 `before_graph`

    print(f"[DEBUG] after_graph inputs after fix: {[i.name for i in after_graph.input]}")
    print(f"[DEBUG] Merging: {connection_tensor_name} -> {after_graph.input[0].name}")

    # **合并模型，确保 opset 版本为 target_opset**
    merged_graph = onnx.helper.make_graph(
        list(before_graph.node) + list(after_graph.node),
        "merged_model",
        before_graph.input,
        after_graph.output,
        list(before_graph.initializer) + list(after_graph.initializer)
    )

    merged_model = onnx.helper.make_model(merged_graph, opset_imports=[onnx.helper.make_opsetid("", target_opset)])

    return merged_model
    
from onnx import helper, numpy_helper

def modify_onnx_model(input_path, output_path):
    # 加载模型
    print(input_path)
    model = onnx.load(input_path)

    # 创建图的引用
    graph = model.graph

    # 1. 修改ReduceMax的输入为Sigmoid的输出，并更改axes
    sigmoid_output = "/model.23/Sigmoid_output_0"
    for node in graph.node:
        if node.op_type == "ReduceMax" and node.name == "/model.23/ReduceMax":
            node.input[0] = sigmoid_output
            for attr in node.attribute:
                if attr.name == "axes":
                    attr.ints[0] = 1

    # 2. 删除Concat_5, Transpose, Transpose_1和Split_2
    nodes_to_remove = ["/model.23/Concat_5", "/model.23/Transpose", "/model.23/Transpose_1", "/model.23/Split_2"]
    new_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
    graph.ClearField("node")  # 清空当前的节点
    graph.node.extend(new_nodes)  # 添加保留的节点

    # 3. 修改GatherElements_1的data输入为Sigmoid的输出
    for node in graph.node:
        if node.op_type == "GatherElements" and node.name == "/model.23/GatherElements_1":
            node.input[0] = sigmoid_output

    # 4. 修改GatherElements的data输入为Mul的输出
    for node in graph.node:
        if node.op_type == "GatherElements" and node.name == "/model.23/GatherElements":
            node.input[0] = "/model.23/Mul_output_0"

    # 5. 插入Transpose
    custom_transpose_1 = helper.make_node(
        'Transpose',
        inputs=["/model.23/Tile_output_0"],
        outputs=["custom_transpose_1"],
        perm=[0, 2, 1]
    )
    graph.node.append(custom_transpose_1)

    # 修改GatherElements的indices输入为Transpose的输出
    for node in graph.node:
        if node.op_type == "GatherElements" and node.name == "/model.23/GatherElements":
            node.input[1] = "custom_transpose_1"

    # 6. 插入Transpose到GatherElements的输出
    custom_transpose_2 = helper.make_node(
        'Transpose',
        inputs=["/model.23/GatherElements_output_0"],
        outputs=["custom_transpose_2"],
        perm=[0, 2, 1]
    )
    graph.node.append(custom_transpose_2)


    # 7. 修改GatherElements的axis为2
    for node in graph.node:
        if node.op_type == "GatherElements" and node.name == "/model.23/GatherElements":
            for attr in node.attribute:
                if attr.name == "axis":
                    attr.i = 2

    # 8. 修改GatherElements_2的data输入为Transpose的输出
    for node in graph.node:
        if node.op_type == "GatherElements" and node.name == "/model.23/GatherElements_2":
            node.input[0] = "custom_transpose_2"

    # 保存修改后的模型
    onnx.save(model, output_path)

def remove_unused_initializers(input_path, output_path):
    """
    移除 ONNX 模型中未被任何节点引用的 `initializer`
    """
    # 1. 加载 ONNX 模型
    model = onnx.load(input_path)
    graph = model.graph

    # 2. 获取所有使用中的 `initializer` 引用
    used_tensors = set()
    for node in graph.node:
        for input_name in node.input:
            used_tensors.add(input_name)

    # 3. 过滤未被引用的 `initializer`
    new_initializers = [init for init in graph.initializer if init.name in used_tensors]

    # 4. 更新 graph，清空原 `initializer` 并添加筛选后的 `initializer`
    graph.ClearField("initializer")
    graph.initializer.extend(new_initializers)

    # 5. 保存优化后的模型
    onnx.save(model, output_path)

    print(f"✅ 处理完成！未使用的 `initializer` 已移除。新模型保存在: {output_path}")



def process_model(model_path, remove_expand_nodes_path, merge_path, remove_unused_initializers_path, final_model_path, split_node_name):
    """
    1. 先移除 Expand 节点
    2. 拆分 `ONNX` 模型
    3. 对 `split_node_name` 之前的部分进行 `onnxslim` 优化
    4. 重新合并模型
    5. 优化transpose
    """
    # 1️ **移除 Expand 节点**
    remove_expand_nodes(model_path, remove_expand_nodes_path)

    # 2️ **加载去除 Expand 后的模型**
    model = onnx.load(remove_expand_nodes_path)

    # 3️ **拆分模型**
    model_before, model_after = split_model_at_node(model, split_node_name)

    # 4️ **优化 `/model.23/Transpose` 之前的部分**
    model_before = shape_inference.infer_shapes(model_before)
    model_before = onnxslim.slim(model_before)

    # 5️ **合并优化后的 `model_before` 和 `model_after`**
    merge_model = merge_models(model_before, model_after)

    # 6️ **保存模型**
    onnx.save(merge_model, merge_path)

    remove_unused_initializers_model = remove_unused_initializers(merge_path, remove_unused_initializers_path)

    modify_onnx_model(remove_unused_initializers_path, final_model_path)

    print(f"✅ 处理完成，最终模型保存在: {final_model_path}")

# 运行代码
model_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/quantized_model.onnx"
remove_expand_nodes_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/onnx/quantized_model1.onnx"
merge_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/onnx/quantized_model2.onnx"
remove_unused_initializers_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/onnx/quantized_model3.onnx"
final_model_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/onnx/final_optimized_model4.onnx"
split_node_name = "/model.23/Transpose"

process_model(model_path, remove_expand_nodes_path, merge_path, remove_unused_initializers_path, final_model_path, split_node_name)
