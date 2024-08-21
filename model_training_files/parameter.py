import os
import sys
import torch
import torch.nn as nn
# 获取 main.py 所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)  # 获取 project 目录的路径
# 将 project 目录添加到 sys.path 中
sys.path.append(main_dir)
from model_v3 import MobileNetV3,mobilenet_v3_large,mobilenet_v3_small
import matplotlib.pyplot as plt
import numpy as np
# model_path = "./MobileNetV3_Small_bird_real_non_freeze.pth"
# model_path = "./MobileNetV3_Small_bird_real_freeze.pth"
# model_path = "./MobileNetV3_Small_bird_real.pth"
# model_path ="./MobileNetV3_Small_bird_real_freeze.pth"

# model_path = "./MobileNetV3_Small_5.5_4.4.pth"
model_path = "./MobileNetV3_Small.pth"

# model_path = "./MobileNetV3_Small_bird_real_distortion.pth"


# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 实例化模型
model = mobilenet_v3_small(num_classes=2)

# 加载模型参数
model.load_state_dict(torch.load(model_path))

# 提取所有参数并保存到列表中
all_params = []
param_infos = []

for name, param in model.named_parameters():
    flattened_param = param.data.cpu().numpy().flatten()
    all_params.extend(flattened_param)
    param_infos.extend([(name, i) for i in range(len(flattened_param))])

# 将参数列表转换为 NumPy 数组
all_params = np.array(all_params)

# 找到前5个最大值和最小值的索引
max_indices = np.argpartition(-all_params, 5)[:5]
min_indices = np.argpartition(all_params, 5)[:5]

# 获取前5个最大值及其索引
max_values = all_params[max_indices]
min_values = all_params[min_indices]

# 打印最大值和最小值及其索引
for i in range(5):
    print(f'Max value {i+1}: {max_values[i]} at index {max_indices[i]}')
    print(f'Min value {i+1}: {min_values[i]} at index {min_indices[i]}')
    
# 找到参数的最大值和最小值
max_value = np.max(all_params)
min_value = np.min(all_params)
max_index = np.argmax(all_params)
min_index = np.argmin(all_params)    
    
# 绘制点图
plt.figure(figsize=(10, 6))
plt.scatter(range(len(all_params)), all_params, alpha=0.6, s=1)
plt.title('Distribution of Model Parameters')
plt.xlabel('Parameter Index')
plt.ylabel('Parameter Value')

# 标注最大值和最小值
plt.text(max_index, max_value+0.1, f'{max_value:.3f}', fontsize=10, color='red', ha='center', va='bottom')
plt.text(min_index, min_value-0.2, f'{min_value:.3f}', fontsize=10, color='blue', ha='center', va='top')


plt.show()










# 设置阈值 i
# up = 5.5
up = 5.5

# down = 4.8
down = 4.4

# 找到大于 i 和小于 -i 的参数的个数
num_greater_than_i = np.sum(all_params >= up)
num_less_than_minus_i = np.sum(all_params <= -down)

print(f'Number of parameters greater than {up}: {num_greater_than_i}')
print(f'Number of parameters less than -{down}: {num_less_than_minus_i}')

# 修改大于 i 和小于 -i 的参数值
all_params[all_params > up] = up
all_params[all_params < -down] = -down

# 将修改后的参数值重新赋值回模型
current_index = 0
for name, param in model.named_parameters():
    param_length = param.data.numel()
    param_data = all_params[current_index:current_index + param_length].reshape(param.data.shape)
    with torch.no_grad():
        param.data.copy_(torch.from_numpy(param_data))
    current_index += param_length

# 验证修改
modified_all_params = []
for name, param in model.named_parameters():
    modified_all_params.extend(param.data.cpu().numpy().flatten())
modified_all_params = np.array(modified_all_params)

# 将参数列表转换为 NumPy 数组
modified_all_params = np.array(modified_all_params)

# 找到前5个最大值和最小值的索引
max_indices = np.argpartition(-modified_all_params, 5)[:5]
min_indices = np.argpartition(modified_all_params, 5)[:5]

# 获取前5个最大值及其索引
max_values = modified_all_params[max_indices]
min_values = modified_all_params[min_indices]

# 打印最大值和最小值及其索引
for i in range(5):
    print(f'Max value {i+1}: {max_values[i]} at index {max_indices[i]}')
    print(f'Min value {i+1}: {min_values[i]} at index {min_indices[i]}')
    
# 找到参数的最大值和最小值
max_value = np.max(modified_all_params)
min_value = np.min(modified_all_params)
max_index = np.argmax(modified_all_params)
min_index = np.argmin(modified_all_params)  
# 绘制点图
plt.figure(figsize=(10, 6))
plt.scatter(range(len(modified_all_params)), modified_all_params, alpha=0.6, s=1)
plt.title(f'Distribution of Model Parameters modified{up}_{down}')
plt.xlabel('Parameter Index')
plt.ylabel('Parameter Value')
# 标注最大值和最小值
plt.text(max_index, max_value+0.1, f'{max_value:.3f}', fontsize=10, color='red', ha='center', va='bottom')
plt.text(min_index, min_value-0.2, f'{min_value:.3f}', fontsize=10, color='blue', ha='center', va='top')


plt.show()

# 保存修改后的模型
modified_model_path = f"./original_parameters/MobileNetV3_Small_{up}_{down}.pth"
torch.save(model.state_dict(), modified_model_path)
print(f'Model parameters saved to {modified_model_path}')