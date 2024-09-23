import os
from PIL import Image

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

def get_next_index(folder, prefix):
    existing_indices = []
    for filename in os.listdir(folder):
        if filename.startswith(prefix):
            try:
                index = int(filename[len(prefix):-4])  # Assuming file extension is 3 characters long (e.g., .jpg, .png)
                existing_indices.append(index)
            except ValueError:
                continue
    return max(existing_indices, default=0) + 1

def rename_images(folder):
    i =0
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            old_path = os.path.join(folder, filename)
            width, height = get_image_size(old_path)
            prefix = f"{width}x{height}_"
            index = get_next_index(folder, prefix)
            new_filename = f"{prefix}{index}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
            i = i+1
    print(i)

if __name__ == "__main__":

    folder = "../pictures/实习8.15/convert_resize/birdto4"  # 替换为目标文件夹路径
    rename_images(folder)
    folder = "../pictures/实习8.15/convert_resize/otherto4"  # 替换为目标文件夹路径
    rename_images(folder)
    folder = "../pictures/实习8.15/convert/birdto4"  # 替换为目标文件夹路径
    rename_images(folder)
    folder = "../pictures/实习8.15/convert/otherto4"  # 替换为目标文件夹路径
    rename_images(folder)