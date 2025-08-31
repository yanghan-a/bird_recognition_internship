import os
import cv2

def convert_jpg_to_yuv420p(source_folder, target_folder):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有JPG图像
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.jpg'):
            # 生成源文件路径
            src_path = os.path.join(source_folder, filename)
            
            # 读取图像
            img = cv2.imread(src_path)
            if img is None:
                print(f"Failed to load image {src_path}")
                continue

            # 获取图像的宽度和高度
            height, width = img.shape[:2]

            # 确保宽度和高度为偶数
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1

            img = cv2.resize(img, (width, height))

            # 将图像转换为YUV420p格式
            yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)

            # 生成目标文件路径
            base_filename = os.path.splitext(filename)[0]
            tgt_path = os.path.join(target_folder, base_filename + '.yuv')

            # 保存YUV420p格式的图像
            with open(tgt_path, 'wb') as f:
                f.write(yuv_img.tobytes())

            print(f"Converted {src_path} to {tgt_path}")


if __name__ == "__main__":
    source_folder = "../pictures/实习8.15/convert_resize/birdto4"  # 替换为目标文件夹路径
    target_folder = "../pictures/实习8.15/convert_resize/bird_yuv"  # 替换为目标文件夹路径

    convert_jpg_to_yuv420p(source_folder, target_folder)

    source_folder = "../pictures/实习8.15/convert_resize/otherto4"  # 替换为目标文件夹路径
    target_folder = "../pictures/实习8.15/convert_resize/other_yuv"  # 替换为目标文件夹路径

    convert_jpg_to_yuv420p(source_folder, target_folder)

    source_folder = "../pictures/实习8.15/convert/birdto4"  # 替换为目标文件夹路径
    target_folder = "../pictures/实习8.15/convert/bird_yuv"  # 替换为目标文件夹路径

    convert_jpg_to_yuv420p(source_folder, target_folder)

    source_folder = "../pictures/实习8.15/convert/otherto4"  # 替换为目标文件夹路径
    target_folder = "../pictures/实习8.15/convert/other_yuv"  # 替换为目标文件夹路径

    convert_jpg_to_yuv420p(source_folder, target_folder)