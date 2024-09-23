import os
import cv2

def resize_to_multiple_of_four(img):
    # 获取图像的宽度和高度
    height, width = img.shape[:2]

    # 计算最近的四的整数倍
    new_width = (width + 3) // 4 * 4
    new_height = (height + 3) // 4 * 4

    # 调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_img

def resize_images_in_folder(source_folder, target_folder):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有图像
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 生成源文件路径
            src_path = os.path.join(source_folder, filename)

            # 读取图像
            img = cv2.imread(src_path)
            if img is None:
                print(f"Failed to load image {src_path}")
                continue

            # 调整图像的宽度和高度为最近的四的整数倍
            resized_img = resize_to_multiple_of_four(img)

            # 生成目标文件路径
            tgt_path = os.path.join(target_folder, filename)

            # 保存调整大小后的图像
            cv2.imwrite(tgt_path, resized_img)

            print(f"Resized {src_path} and saved to {tgt_path}")

if __name__ == "__main__":
    source_folder = "../pictures/实习8.15/together_8_14/resize/bird_resize"  # 替换为源文件夹路径
    target_folder = "../pictures/实习8.15/convert_resize/birdto4"  # 替换为目标文件夹路径

    resize_images_in_folder(source_folder, target_folder)

    source_folder = "../pictures/实习8.15/together_8_14/resize/other_resize"  # 替换为源文件夹路径
    target_folder = "../pictures/实习8.15/convert_resize/otherto4"  # 替换为目标文件夹路径

    resize_images_in_folder(source_folder, target_folder)

    source_folder = "../pictures/实习8.15/together_8_14/no_resize/bird"  # 替换为源文件夹路径
    target_folder = "../pictures/实习8.15/convert/birdto4"  # 替换为目标文件夹路径

    resize_images_in_folder(source_folder, target_folder)

    source_folder = "../pictures/实习8.15/together_8_14/no_resize/other"  # 替换为源文件夹路径
    target_folder = "../pictures/实习8.15/convert/otherto4"  # 替换为目标文件夹路径

    resize_images_in_folder(source_folder, target_folder)
