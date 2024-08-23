这份代码中包括两种实现方式，一种是使用rga加速，另一种是使用GPU处理图像

需要指出，由于importbuffer_fd函数需要将外部的buffer导入到RGA驱动内，导致处理图像速度没有直接使用GPU执行的快

下面介绍使用方式：

build-linux_RV1106.sh和build-linux_RV1106_RGA.sh为两个编译源文件的执行脚本
在这里注意要添加交叉工具链的位置，在.bashrc文件下添加如下信息
export RK_RV1106_TOOLCHAIN=/to/your/path/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf

编译后请将install下的文件夹推送到板端，可以直接使用commands.sh和commands_rga.sh直接进行编译并推送

由于adb push不能推送到不存在的文件夹，因此需要保证板端具有相应的文件夹
mkdir rknn_mobilenet rknn_mobilenet_rga
cd rknn_mobilenet
mk lib model pictures
cd pictures
mkdir bird bird_yuv other other_yuv
此时可以直接运行commands.sh和commands_rga.sh

main函数可以调整选择的文件夹，目前是分别对含鸟和无鸟的两个文件夹进行推理，修改文件夹请查看main.cpp

目前提供了四个使用了不同优化方式的rknn模型进行推理，推荐使用MobileNetV3_Small_bird_real_non_freeze_modified_mmse.rknn
如需调整模型，请查看哪inference.cpp的宏定义，修改参数MODEL_PATH，文件IMAGENET_CLASSES_FILE定义了模型的类别

下面介绍整体pipeline：
未使用RGA：读取yuv图像-->yuv图像resize-->resize图像转为rgb-->rgb图像裁剪-->模型推理

使用RGA：读取yuv图像-->yuv图像resize并转为rgb-->rgb图像裁剪-->模型推理

考虑到真实场景的应用，测试的图片需要先转换为yuv420p格式，如果是其它格式对于rga模式只需修改rga_resize_yuv_to_rgb.cpp文件的参数即可，对于不使用RGA需实现特定格式转rgb的代码。

为了方便测验模型的准确率，这里使用的是读文件夹的方式，并对推理结果进行统计，统计信息请查看最终结果，可通过终端查看推理情况

注意：
由于图像处理过程需要使用图像的尺寸信息，而yuv原始数据不包含尺寸信息，因此需要对图像进行重命名，命名规则为：宽x高_索引值

对于一些图像预处理的操作均可使用python脚本实现，例如:rename.py, jpg2yuv420p

2024/8/8更新：
RGA加速的问题已经修改，但是rga在对yuv做resize时要求图像宽4对齐，但测试样例中存在非4对齐图像，造成准确率下降，真实情况下请不要使用非4对齐图像

修改了使用RGA时的pipeline：使用RGA：读取yuv图像-->yuv图像resize-->yuv转rgb-->rgb图像裁剪-->模型推理
