import os

def list_files(directory, output_file):
    try:
        with open(output_file, 'w') as f:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    f.write(file + '\n')
        print(f"File names have been written to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
directory = "ILSVRC2012_img_val_samples_my/"  # 将此处替换为你的目录路径
output_file = "./data_set20.txt"  # 输出文件名

list_files(directory, output_file)

