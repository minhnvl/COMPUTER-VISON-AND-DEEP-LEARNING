import os
import random
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 50
BATCH_SIZE = 32     
LR = 0.001 
Label_Cifar10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
TYPEOFTRAIN = "MNIST"
# Nếu là CIFAR10 thì Cifar10
# Nếu là MNIST thì MNIST
class LeNet(nn.Module):
    if TYPEOFTRAIN == "Cifar10":
        def __init__(self):
            super(LeNet, self).__init__()
            # 3 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

            # setup cost function and optimizer
            # self.setup_COST_Function_and_OPTimizer()
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    else:
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Sequential(     
                nn.Conv2d(1, 6, 5, 1, 2),
                nn.ReLU(),      
                nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),      #input_size=(16*10*10)
                nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU()
            )
            self.fc3 = nn.Linear(84, 10)

        # 定义前向传播过程，输入为x
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
            x = x.view(x.size()[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

def Pre_training(): 
    transform = transforms.ToTensor()
    if TYPEOFTRAIN == "Cifar10":
        trainset = tv.datasets.CIFAR10(
            root='./',
            train=True,
            download=True,
            transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            )
        testset = tv.datasets.CIFAR10(
            root='./',
            train=False,
            download=True,
            transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            )
    else:
        trainset = tv.datasets.MNIST(
            root='./',
            train=True,
            download=True,
            transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            )
        testset = tv.datasets.MNIST(
            root='./',
            train=False,
            download=True,
            transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            )
    return trainset, trainloader, testset, testloader

def Processing_Training():
    trainset, trainloader, testset, testloader = Pre_training()
    PATH_SAVE = './model_' + TYPEOFTRAIN + '/'
    if not os.path.exists(PATH_SAVE):
            os.makedirs(PATH_SAVE)
    net = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default= PATH_SAVE, help='folder to output images and model checkpoints') #模型保存路径
    parser.add_argument('--net', default= PATH_SAVE +'/net.pth', help="path to netG (to continue training)")  #模型加载路径
    opt = parser.parse_args()
    List_loss = []
    List_Percent_Training = []
    List_Percent_Testing = []

    # fig, axes = plt.subplots(1, 10, figsize=(12,5))
    # for i in range(10):
    #     index = random.randint(0,9999)
    #     image = trainset[index][0] 
    #     label = trainset[index][1] 
    #     print("The picture %s is showing: The %s" %(str(i+1),Label_Cifar10[int(label)]))
    #     # original_img = torch.convert_cifar10(image,pil=False)
    #     npimg = image.numpy()
    #     # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     axes[i].imshow(np.squeeze(npimg))
    #     axes[i].set_title(Label_Cifar10[int(label)])
    #     axes[i].set_xticks([])
    #     axes[i].set_yticks([])
    # plt.show()
    for epoch in range(EPOCH):
        sum_loss = 0.0
        correct_training = 0
        total_training = 0
        for i, data in enumerate(trainloader):
        
            images = data[0]
            labels = data[1]
            images, labels = images.to(device), labels.to(device)
            # a = np.expand_dims(images, axis=0)
            # npimg = images.numpy()
            # plt.imshow(np.squeeze(npimg))
            # plt.show()
            # input()
            # images = images.squeeze(0)
            # labels = torch.argmax(labels)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_training += labels.size(0)
            correct_training += (predicted == labels).sum()
            
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                        % (epoch + 1, i + 1, loss.item()))
                sum_loss = 0.0
        List_Percent_Training.append(100 * correct_training / total_training)
        List_loss.append(loss.item())
        print('Training Percent of %dth epoch is: %d%%' % (epoch + 1, (100 * correct_training / total_training)))
        with torch.no_grad():
            correct_testing = 0
            total_testing = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total_testing += labels.size(0)
                correct_testing += (predicted == labels).sum()
            List_Percent_Testing.append(100 * correct_testing / total_testing)
            print('Exactly Percent of %dth epoch is: %d%%' % (epoch + 1, (100 * correct_testing / total_testing)))
        
        torch.save(net.state_dict(), '%s/Lenet_%s_%03d.pth' % (opt.outf, TYPEOFTRAIN ,epoch + 1))
    return List_Percent_Training, List_Percent_Testing, List_loss
def Show_Grapth():
    List_Percent_Training, List_Percent_Testing, List_loss = Processing_Training()
    fig, axs = plt.subplots(2)
    fig.suptitle('Accuracy')
    axs[0].plot(List_Percent_Training,label = "Training")
    axs[0].plot(List_Percent_Testing,label = "Testing")
    axs[0].legend(loc='lower right')
    # axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('%')
    axs[0].set_ylim([0,100])
    axs[1].plot(List_loss)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    plt.show()

def Testting_Model_Training():
    net = LeNet()
    trainset, trainloader, testset, testloader = Pre_training()
    index = random.randint(0,9999)
    image1 = trainset[index][0]
    label = trainset[index][1] 

    PATH = "./model_%s/Lenet_%s_028.pth"%(TYPEOFTRAIN,TYPEOFTRAIN)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    # images, labels = image.to(device), label.to(device)
    image = image1.unsqueeze(0)
    npimg = image1.numpy()
    plt.imshow(np.squeeze(npimg))
    plt.show()
    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1)
    if TYPEOFTRAIN == "MNIST":
        print(predicted[0])
    else:
        print(Label_Cifar10[int(predicted[0])])
    # List_percent = [ ]
    # for x in outputs.data[0]:
    #     if (x < 0):
    #         x =0.01
    #     else:
    #         x = x/10
    #     List_percent.append(float(x))
    # print(List_percent)
    # fig, axs = plt.subplots(1,2,figsize=(14,6))
    # image111 = image1.numpy()
    # rects1 = axs[1].bar(Label, List_percent, color='b')
    # axs[1].set_ylim([0,0.8])
    # axs[0].imshow(np.transpose(image111, (1, 2, 0)))
    # axs[0].set_xlabel(Label[int(predicted[0])])
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # plt.show()

if __name__ == "__main__":
    # Xử lý không hiện graph()
    # Processing_Training()
    # Xử Lý có hiện Graph()
    Show_Grapth()
    # Testing model đã train
    # Testting_Model_Training()



# List_loss = [2.313, 2.295, 2.297, 2.297, 2.295, 2.293, 2.273, 2.276, 2.225, 2.254, 2.088, 2.055, 1.829, 1.98, 2.073, 1.912, 1.854, 1.888, 1.775, 1.692, 1.71, 1.737, 1.608, 1.554, 1.492, 1.56, 1.571, 1.572, 1.55, 1.536, 1.522, 1.292, 1.263, 1.236, 1.241, 1.252, 1.281, 1.271, 1.216, 1.208, 0.958, 0.927, 1.241, 1.041, 1.165, 1.112, 1.074, 1.293, 1.174, 0.979, 1.294, 1.181, 1.273, 1.25, 1.235, 1.102, 0.985, 1.277, 1.118, 1.211, 1.196, 0.936, 1.263, 1.419, 0.876, 1.158, 1.94, 1.265, 1.246, 1.213, 1.101, 1.327, 0.86, 1.151, 1.35, 0.792, 1.03, 0.937, 1.104, 1.223, 0.961, 1.297, 1.265, 1.074, 1.008, 1.29, 1.168, 0.99, 1.554, 1.399, 1.087, 1.047, 1.088, 1.43, 0.89, 1.133, 1.423, 1.118, 1.447, 1.153, 1.433, 1.192, 0.974, 1.245, 0.942, 1.153, 1.072, 0.674, 0.77, 1.433, 0.895, 0.996, 0.883, 0.954, 1.02, 1.075, 1.237, 1.22, 0.94, 1.229, 0.862, 0.983, 0.921, 0.847, 1.239, 0.886, 1.026, 0.925, 1.208, 1.02, 1.165, 1.028, 0.88, 0.996, 1.483, 0.727, 0.761, 1.429, 1.444, 1.061, 0.761, 1.08, 0.969, 0.977, 0.841, 0.887, 0.887, 0.928, 1.022, 0.783, 0.751, 0.906, 0.821, 0.921, 1.063, 1.152, 0.798, 1.32, 1.117, 1.034, 1.508, 0.813, 1.087, 0.761, 0.907, 0.789, 0.874, 0.721, 1.043, 0.944, 0.882, 1.047, 1.361, 0.918, 0.852, 0.985, 0.861, 1.076, 0.926, 0.828, 0.849, 0.74, 0.758, 0.793, 1.42, 1.011, 1.12, 1.236, 1.012, 0.763, 1.111, 
# 0.986, 1.105, 0.827, 0.683, 0.643, 1.005, 0.716, 0.888, 0.499, 0.885, 0.783, 0.945, 0.735, 0.541, 0.783, 0.878, 0.705, 0.621, 0.678, 0.693, 0.724, 1.054, 1.29, 0.871, 0.612, 0.649, 0.911, 0.936, 0.91, 1.049, 0.893, 0.876, 0.669, 1.083, 0.874, 0.569, 1.055, 0.712, 1.011, 0.727, 0.921, 1.012, 0.925, 1.085, 0.721, 0.676, 0.946, 0.757, 0.584, 0.546, 0.656, 0.879, 0.945, 1.065, 0.755, 0.84, 0.728, 0.804, 1.045, 0.761, 0.682, 0.592, 0.935, 
# 0.547, 0.756, 0.544, 0.696, 0.81, 0.743, 0.862, 0.734, 1.17, 0.866, 0.761, 0.51, 0.662, 1.036, 0.969, 0.821, 1.136, 0.994, 0.717, 0.712, 1.368, 0.826, 1.381, 0.854, 0.866, 0.873, 1.339, 0.858, 0.655, 1.067, 0.655, 0.814, 0.572, 0.598, 0.672, 1.026, 0.749, 0.617, 0.48, 1.032, 0.432, 0.552, 0.735, 0.847, 0.753, 0.659, 0.886, 0.973, 0.694, 0.972, 0.534, 0.479, 0.691, 0.565, 0.796, 0.579, 0.683, 0.683, 0.977, 0.689, 0.696, 1.087, 0.784, 0.481, 0.547, 0.752, 0.408, 0.795, 0.775, 0.847, 0.954, 0.73, 0.846, 0.775, 0.821, 0.587, 0.602, 0.786, 1.001, 0.862, 0.594, 0.67, 0.962, 0.865, 0.625, 0.685, 0.646, 0.605, 0.511, 0.643, 0.711, 0.782, 0.732, 0.779, 0.523, 0.596, 0.509, 0.715, 0.66, 0.635, 0.648, 0.741, 0.529, 0.674, 0.863, 0.828, 0.751, 0.392, 0.616, 0.843, 0.603, 0.687, 0.719, 0.803, 0.635, 0.496, 0.473, 0.478, 0.351, 0.605, 0.676, 0.797, 0.515, 0.854, 0.809, 0.725, 0.736, 0.625, 0.923, 0.675, 0.894, 0.468, 0.597, 0.411, 0.869, 0.747, 0.703, 0.606, 0.929, 0.634, 0.572, 0.63, 0.534, 0.426, 0.717, 0.369, 0.459, 0.604, 0.584, 0.596, 0.604, 0.842, 0.77, 0.633, 0.475, 0.788, 0.481, 0.239, 0.533, 0.621, 0.599, 0.416, 0.669, 0.909, 0.207, 0.495, 0.55, 0.608, 0.631, 0.378, 0.409, 0.713, 0.796, 0.577, 0.625, 0.499, 0.481, 0.319, 0.593, 0.514, 0.614, 0.721, 0.757, 0.67, 0.568, 0.494, 0.65, 0.732, 0.26, 0.407, 0.633, 0.453, 0.583, 0.559, 0.838, 0.532, 0.712, 0.839, 0.511, 0.413, 1.082, 0.599, 0.61, 0.194, 0.744, 0.451, 0.66, 0.428, 0.43, 0.544, 0.529, 0.532, 0.608, 0.584, 0.632, 0.905, 0.432, 0.315, 0.521, 0.482, 0.514, 0.501, 1.27, 0.741, 0.518, 0.617, 0.692, 0.949, 0.574, 0.357, 0.639, 0.575, 0.904, 0.331, 0.342, 0.303, 0.499, 0.412, 0.594, 0.631, 0.441, 0.607, 0.493, 0.364, 0.533, 0.582, 0.63, 0.642, 0.602, 0.91, 0.52, 0.532, 0.179, 0.781, 0.32, 0.44, 0.456, 0.528, 0.257, 0.386, 0.276, 0.324, 0.521, 0.363, 0.616, 0.43, 0.307, 0.379, 0.462, 0.5, 0.477, 0.719, 0.424, 0.679, 0.347, 0.739, 0.81, 0.71, 0.369, 0.73, 0.616, 0.174, 0.257, 0.276, 0.604, 0.461, 0.784, 0.499, 0.607, 0.467, 0.691, 0.253, 0.554, 0.547, 0.299, 0.415, 0.355, 0.634, 0.222, 0.719, 0.362, 0.524, 0.375, 0.161, 0.67, 0.434, 0.56, 0.47, 0.393, 0.842, 0.446, 0.164, 0.435, 0.182, 0.672, 0.37, 0.509, 0.427, 0.269, 0.274, 0.484, 0.31, 0.5, 0.391, 0.237, 0.197, 0.324, 0.157, 0.642, 0.161, 0.414, 0.5, 0.414, 0.429, 0.525, 0.407, 0.378, 0.387, 0.383, 0.436, 0.492, 0.392, 0.443, 0.292, 0.253, 0.512, 0.218, 0.373, 0.229, 0.584, 0.345, 0.438, 0.458, 0.41, 0.413, 0.286, 0.518, 0.301, 0.392, 0.258, 0.264, 0.296, 0.398, 0.432, 0.481, 0.331, 0.387, 0.24, 0.362, 0.235, 0.242, 0.501, 0.302, 0.375, 0.241, 0.431, 0.317, 0.402, 0.336, 0.215, 0.231, 
# 0.127, 0.414, 0.408, 0.248, 0.324, 0.144, 0.378, 0.597, 0.532, 0.374, 0.311, 0.317, 0.32, 0.347, 0.282]
# List_Percent = [17, 32, 39, 46, 47, 51, 52, 53, 55, 55, 58, 59, 60, 59, 59, 62, 61, 62, 62, 62, 63, 62, 62, 63, 61, 64, 64, 64, 63, 64, 63, 63, 64, 64, 64, 62, 63, 63, 62, 63, 63, 63, 62, 63, 61, 62, 62, 63, 61, 60]
# print(List_loss)
# print(List_Percent)