import os
import sys
import torch
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取当前目录的上一级目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# 将上一级目录添加到 sys.path 中
sys.path.append(parent_dir)
from model_v3 import mobilenet_v3_large, mobilenet_v3_small

def export_onnx_model(model_name, model_weight_path, output_onnx_path, input_size=(1, 3, 224, 224)):
    # 根据模型名称加载相应的模型
    if model_name == 'mobilenet_v3_small':
        model = mobilenet_v3_small(num_classes=2)
    elif model_name == 'mobilenet_v3_large':
        model = mobilenet_v3_large(num_classes=2)
    else:
        raise ValueError("Unsupported model name. Choose either 'mobilenet_v3_small' or 'mobilenet_v3_large'.")

    # 加载模型权重
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()

    # 创建一个假的输入张量，指定输入大小
    dummy_input = torch.randn(*input_size)

    # 导出模型为 ONNX 格式
    torch.onnx.export(
        model,                             # 模型实例
        dummy_input,                       # 示例输入张量
        output_onnx_path,                  # 导出的 ONNX 文件名
        input_names=['input'],             # 输入节点的名称（可选）
        output_names=['output'],           # 输出节点的名称（可选）
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态轴配置（可选）
    )
    print(f"模型已成功保存为 {output_onnx_path}")

# 示例调用函数
if __name__ == '__main__':
    # 示例调用参数
    model_name = 'mobilenet_v3_small'
    model_weight_path = './MobileNetV3_Small_5.5_4.4_distortion.pth'
    output_onnx_path = './MobileNetV3_Small_5.5_4.4_distortion.onnx'
    input_size = (1, 3, 224, 224)
    export_onnx_model(model_name, model_weight_path, output_onnx_path, input_size)
    # 示例调用参数
    model_weight_path = './MobileNetV3_Small_distortion.pth'
    output_onnx_path = './MobileNetV3_Small_distortion.onnx'
    export_onnx_model(model_name, model_weight_path, output_onnx_path, input_size)

