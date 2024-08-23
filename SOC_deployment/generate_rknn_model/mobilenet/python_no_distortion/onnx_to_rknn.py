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

MODEL_PATH = MODEL_DIR + './MobileNetV3_Small_5.5_4.4.onnx'

# MODEL_PATH = MODEL_DIR + './MobileNetV3_Small.onnx'



OUT_RKNN_PATH = MODEL_DIR + './MobileNetV3_Small_5.5_4.4_normal.rknn'

# OUT_RKNN_PATH = MODEL_DIR + './MobileNetV3_Small_normal.rknn'


# CLASS_LABEL_PATH = MODEL_DIR + 'synset.txt'
# CLASS_LABEL_PATH = MODEL_DIR + 'class.txt'
CLASS_LABEL_PATH = MODEL_DIR + 'bird_other.txt'



RKNPU1_TARGET = ['rk1808', 'rv1109', 'rv1126']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MobileNet Python Demo', add_help=True)
    parser.add_argument('--target', type=str,
                        default='rv1103', help='RKNPU target platform')
    parser.add_argument('--npu_device_test', action='store_true',
                        default=False, help='Connected npu device run')
    parser.add_argument('--accuracy_analysis', action='store_true',
                        default=False, help='Accuracy analysis')
    parser.add_argument('--eval_perf', action='store_true',
                        default=False, help='Time consuming evaluation')
    parser.add_argument('--eval_memory', action='store_true',
                        default=False, help='Memory evaluation')
    parser.add_argument('--model', type=str,
                        default=MODEL_PATH, help='onnx model path')
    parser.add_argument('--output_path', type=str,
                        default=OUT_RKNN_PATH, help='output rknn model path')
    parser.add_argument('--dtype', type=str, default='i8',
                        help='dtype of model, i8/fp32 for RKNPU2, u8/fp32 for RKNPU1')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    # 使用mmse算法量化模型，效果更好
    rknn.config(mean_values=[[255*0.485, 255*0.456, 255*0.406]], std_values=[[
                255*0.229, 255*0.224, 255*0.225]], target_platform=args.target, quantized_algorithm="normal")
    # rknn.config(mean_values=[[255*0.485]], std_values=[[
    #         255*0.229]], target_platform=args.target)
    print('done')

    # Load model
    print('--> Loading model')
    if args.target in RKNPU1_TARGET:
        ret = rknn.load_onnx(model=args.model, inputs=['input'], input_size_list=[[3,224,224]])
    else:
        ret = rknn.load_onnx(model=args.model, inputs=['input'], input_size_list=[[1,3,224,224]])

    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    do_quant = True if (args.dtype == 'i8' or args.dtype == 'u8') else False
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)

    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
