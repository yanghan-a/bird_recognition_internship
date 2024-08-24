# 项目内容介绍及使用方法

该项目主要包括两大部分代码，深度学习代码以及板端部署代码。先对代码内容进行简单介绍，为了精简代码，抛弃了版本迭代过程中的其它代码，仅保留实现最终结果的代码。项目链接为[yanghan-a/bird_recognition_internship (github.com)](https://github.com/yanghan-a/bird_recognition_internship)。

SOC_deployment文件夹中是板端模型部署相关的内容。

original_parameters为模型预训练参数。

model_training_files和model_training_files_distortion存放了训练模型的python代码

data_test、data_train、data_valid为不变形的原始图形，data_distortion中仅包含了变形的测试集图像，如需使用训练集和验证集图像可直接使用脚本对data_train、data_valid处理获取。

**model_modified**和**model_modified_distortion**存放了测试使用的网络模型。



model_other存放了100 epoch训练的全部模型，测试使用的是第23个

model_distortion_other存放了100 epoch训练的全部模型，测试使用的是第18个

## 深度学习代码

该部分代码分为两个文件夹，model_training_files和model_training_files_distortion，分别是使用正常图像训练和图像压缩变形后训练。内部文件基本相同，部分路径不同需要修改。

网络训练代码train.py，网络架构代码model_v3.py、model_v2.py，模型预训练参数文件model，模型文件，parameter.py调整模型参数outlier值，extract_visualize.py和visualize.py用来可视化模型在测试集上的结果。

## 板端部署代码

板端部署代码最初实现了热源触发，相机拍摄图片，使用获取的图片进行推理输出结果。

为了模拟真实部署需要让网络直接对视频流中的yuv格式数据进行推理，后续实现了基于内存中yuv格式图片的推理，这里分为两个版本CPU图像处理和RGA图像处理。

在实现rga图像处理功能上踩了很多坑，也对板端模型推理逻辑有了更清晰的理解。

### 遇到问题的答案

模型推理代码构建的基本逻辑是：构造上下文语义对象，加载模型，获取模型参数，根据参数分配模型输入输出地址。**这里尤其需要注意模型的输入输出是预先分配的，进行模型推理时需要将malloc加载的数据使用拷贝函数加载到指定位置。**

使用malloc函数分配的地址读取的数据不能直接被rga直接访问，要实现rga处理图像的功能需要dma_buf_alloc函数来分配内存，否则在构建图像处理句柄时的importbuffer_fd无法获取内存对应的fd值，也就无法使用importbuffer_fd函数。更重要的是RV1103/RV1106不支持importbuffer_virtualaddr函数。

