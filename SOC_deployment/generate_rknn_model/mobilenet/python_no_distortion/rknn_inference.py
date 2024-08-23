import argparse
import os
import sys
import urllib
import urllib.request
import time
import traceback
import numpy as np
import cv2
from rknn.api import RKNN
from scipy.special import softmax

# DATASET_PATH = '../../../datasets/imagenet/ILSVRC2012_img_val_samples/dataset_20.txt'
DATASET_PATH = '../../datasets/ILSVRC2012_img_val_samples_my/dataset_20.txt'

MODEL_DIR = '../model/'

# mmse算法转换
OUT_RKNN_PATH = MODEL_DIR + './MobileNetV3_Small_5.5_4.4.rknn'
# OUT_RKNN_PATH = MODEL_DIR + './MobileNetV3_Small.rknn'


# normal算法转换
# OUT_RKNN_PATH = MODEL_DIR + './MobileNetV3_Small_5.5_4.4_normal.rknn'
# OUT_RKNN_PATH = MODEL_DIR + './MobileNetV3_Small_normal.rknn'


CLASS_LABEL_PATH =  'bird_other.txt'



RKNPU1_TARGET = ['rk1808', 'rv1109', 'rv1126']

def resize_and_crop(img, target_size=224):
    h, w, _ = img.shape

    # 计算缩放比例，较短边缩放到 target_size
    if h < w:
        scale = target_size / h
        new_h, new_w = target_size, int(w * scale)
    else:
        scale = target_size / w
        new_h, new_w = int(h * scale), target_size

    # 缩放图像
    resized_img = cv2.resize(img, (new_w, new_h))

    # 计算中心点并裁剪
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2
    cropped_img_middle = resized_img[start_y:start_y + target_size, start_x:start_x + target_size]

    start_x = 0
    start_y = 0
    cropped_img_left = resized_img[start_y:start_y + target_size, start_x:start_x + target_size]

    end_x = new_w
    end_y = new_h
    cropped_img_right = resized_img[end_y- target_size:end_y , end_x- target_size:end_x ]

    return cropped_img_middle,cropped_img_left,cropped_img_right

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MobileNet Python Demo', add_help=True)
    parser.add_argument('--target', type=str,
                        default='rv1103', help='RKNPU target platform')
    parser.add_argument('--output_path', type=str,
                        default=OUT_RKNN_PATH, help='output rknn model path')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    print('--> Loading model')
    ret = rknn.load_rknn(path=args.output_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target=args.target)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')




    # 设置输入文件夹和输出文件夹路径
    bird_input_folder = '../pictures/data_distortion/bird/'
    bird_output_folder = './bird_output_folder/'
    bird_total= 0 # 用来记录有鸟文件夹图像个数
    with_bird = 0 # 用来记录预测真实有鸟图像个数
    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(bird_input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            bird_total = bird_total+1
            if bird_total%20==0:
                print("当前个数：",bird_total)

            img_path = os.path.join(bird_input_folder, filename)
            # print(f"处理图片: {img_path}")
            # 读取图像
            img = cv2.imread(img_path)           
            # 调用函数处理图像
            cropped_img_middle,cropped_img_left,cropped_img_right = resize_and_crop(img)   


            # 保存处理后的图像
            # output_img_path = os.path.join(bird_output_folder, filename)
            # cv2.imwrite(output_img_path, cropped_img_middle)
            # print(f"保存处理后的图片: {output_img_path}")

            i = 0
            for img in [cropped_img_middle,cropped_img_left,cropped_img_right]:
                # 将图像扩展维度以适应模型输入
                img = np.expand_dims(img, 0)
                i=i+1
                # Inference
                # print('--> Running model')
                outputs = rknn.inference(inputs=[img])
                
                # 在此处处理推理输出，如果需要
                # Post Process
                # print('--> PostProcess')
                # with open(CLASS_LABEL_PATH, 'r') as f:
                #     labels = [l.rstrip() for l in f]

                scores = softmax(outputs[0])

                # print the top-i inferences class
                scores = np.squeeze(scores)
                a = np.argsort(scores)[::-1]
                # print("scores:",scores)
                if a[0] == 0:
                    # if i>0 :
                    #     if max(scores)>=0.75:
                    #         with_bird = with_bird +1
                    # else:
                    with_bird = with_bird +1
                break
                # for i in a[0:i]:
                #     print('[%d] score=%.2f class="%s"' % (i, scores[i], labels[i]))



    # 设置输入文件夹和输出文件夹路径
    other_input_folder = '../pictures/data_distortion/other/'
    other_output_folder = './other_output_folder/'
    other_total= 0 # 用来记录有鸟文件夹图像个数
    other_with_bird = 0 # 用来记录预测真实有鸟图像个数
    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(other_input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            other_total = other_total+1
            if other_total%20==0:
                print("当前个数：",other_total)
            img_path = os.path.join(other_input_folder, filename)
            # print(f"处理图片: {img_path}")
            # 读取图像
            img = cv2.imread(img_path)           
            # 调用函数处理图像
            cropped_img_middle,cropped_img_left,cropped_img_right = resize_and_crop(img)   

            # 保存处理后的图像
            # output_img_path = os.path.join(other_output_folder, filename)
            # cv2.imwrite(output_img_path, cropped_img_middle)
            # print(f"保存处理后的图片: {output_img_path}") 

            l=0
            for img in [cropped_img_middle,cropped_img_left,cropped_img_right]:
                # 将图像扩展维度以适应模型输入
                img = np.expand_dims(img, 0)
                l =l+1
                # Inference
                # print('--> Running model')
                outputs = rknn.inference(inputs=[img])
                
                # 在此处处理推理输出，如果需要
                # Post Process
                # print('--> PostProcess')
                # with open(CLASS_LABEL_PATH, 'r') as f:
                #     labels = [l.rstrip() for l in f]

                scores = softmax(outputs[0])

                # print the top-i inferences class
                scores = np.squeeze(scores)
                a = np.argsort(scores)[::-1]

                if a[0] == 0:
                    # if l>0 :
                    #     if max(scores)>=0.75:
                    #         other_with_bird = other_with_bird +1
                    # else:
                    other_with_bird = other_with_bird +1
                break
                # for i in a[0:i]:
                #     print('[%d] score=%.2f class="%s"' % (i, scores[i], labels[i]))

    print("有鸟正确率：",with_bird/bird_total)
    print("无鸟正确率：",(other_total-other_with_bird)/other_total)
    # Release
    rknn.release()
