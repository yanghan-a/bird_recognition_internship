import os
import sys
import json
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
# 获取 main.py 所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)  # 获取 project 目录的路径
# 将 project 目录添加到 sys.path 中
sys.path.append(main_dir)
from model_v2 import MobileNetV2
from model_v3 import MobileNetV3, mobilenet_v3_large, mobilenet_v3_small
import torchvision.models.mobilenetv3

def plot_loss(train_loss, val_loss, epochs, save_path,val_accurate_values):
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    
    # 找到最小的训练和验证损失
    min_train_loss = min(train_loss)
    min_val_loss = min(val_loss)
    
    # 找到对应的epoch
    min_train_epoch = train_loss.index(min_train_loss) + 1
    min_val_epoch = val_loss.index(min_val_loss) + 1
    
    # 标注最小的训练损失
    plt.scatter(min_train_epoch, min_train_loss, color='red')
    plt.text(min_train_epoch, min_train_loss, f'Train Min: {min_train_loss:.4f}', 
             ha='right', color='red')
    
    # 标注最小的验证损失
    plt.scatter(min_val_epoch, min_val_loss, color='blue')
    plt.text(min_val_epoch, min_val_loss, f'Val Min: {min_val_loss:.4f}', 
             ha='right', color='blue')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_curve_new.png'))
    plt.show()
    plt.figure()
    
    plt.plot(range(1, epochs + 1), val_accurate_values, label='valid accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'accuracy_new.png'))
    plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 128
    epochs = 100
    learning_rate = 0.0001
    workers = 32

    # 定义数据转换
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.55,1.0),ratio=(1,1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224,scale=(0.55,1.0),ratio=(1,1)),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    image_path = os.path.join(data_root, "../data_set", "new_data")  # flower data set path
       
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'bird':0, 'other':1}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # create model
    net = mobilenet_v3_small(num_classes=2).to(device)

    # load pretrain weights
    # model_weight_path = "../original_parameters/mobilenet_v3_small.pth"
    model_weight_path = "../original_parameters/mobilenetv3_small_5.5_4.4.pth"
    
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = True

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate)

    best_acc = 0.0
    save_path = './model_new/MobileNetV3_Small_5.5_4.4.pth'

    train_steps = len(train_loader)

    scaler = torch.cuda.amp.GradScaler()  # for automatic mixed precision

    train_loss_values = []
    val_loss_values = []
    
    val_accurate_values = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # enable mixed precision
                logits = net(images)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_loss = running_loss / train_steps
        train_loss_values.append(train_loss)
        
        
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                with torch.cuda.amp.autocast():  # enable mixed precision
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels).sum().item()

                    loss = loss_function(outputs, val_labels.to(device))
                    val_loss += loss.item()
                
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        val_loss /= len(validate_loader)
        val_loss_values.append(val_loss)
        val_accurate_values.append(val_accurate)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        torch.save(net.state_dict(), save_path+str(epoch+1))
            
    # 绘制loss曲线
    plot_loss(train_loss_values, val_loss_values, epochs, current_dir,val_accurate_values)
    print('Finished Training')


if __name__ == '__main__':
    main()
