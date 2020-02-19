# *************** HW1 -  阮武黎明 ******************
# ************ CVDL2019 - OPEN CVDL2019 ***********


from Hw1_interface import Ui_Opencv_hw1
import sys
import os
import cv2 as cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt

# Training...... 
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random

FolderImage             = "./images/CameraCalibration/"
FolderTransImg          = "./images/"
Path_Img_Perspective    = FolderTransImg + 'OriginalPerspective.png'
Image_Perspective       = cv2.imread(Path_Img_Perspective)
ListImage = [f for f in os.listdir(FolderImage) if f.endswith(".bmp")]
ListTrain = ["Training Cifar-10", "Training MNIST"]
Label_Cifar10 = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Label_MNIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
if len(ListImage) != 0:
    ListImage = sorted(ListImage, key=lambda x: int(x[0:2].replace(".","")))
else:
    ListImage.append("No Image")
app = QtWidgets.QApplication(sys.argv)
Opencv_hw1 = QtWidgets.QWidget()
    
class TWSignals(QtCore.QObject):
	switch_window = QtCore.pyqtSignal(object)

class Process_Image(Ui_Opencv_hw1):
    signals = TWSignals()
      
    def __init__(self):
        super(Process_Image,self).__init__()
        self.setupUi(Opencv_hw1)
        self.btn_findcorners.clicked.connect(lambda check: self.Find_Corners())
        self.btn_intrinsic.clicked.connect(lambda check: self.Intrinsic())
        self.btn_distortion.clicked.connect(lambda check: self.Distortion())
        self.btn_extrinsic.clicked.connect(lambda check: self.Extrinsic())
        self.btn_augmented.clicked.connect(lambda check: self.Augmented_Reality())
        self.btn_roscaltran.clicked.connect(lambda check: self.Image_Transformation())
        self.btn_findcontour.clicked.connect(lambda check: self.Find_Contour())
        self.btn_prespective.clicked.connect(lambda check: self.Perspective_Transformation())
        self.btn_loadimg.clicked.connect(lambda check: self.Load_Image_File())
        self.btn_colorcov.clicked.connect(lambda check: self.Color_Conversion())
        self.btn_showimgtrain.clicked.connect(lambda check: self.Load_Dataset_Image())
        self.btn_showhyper.clicked.connect(lambda check: self.Show_ParameterTranining())
        self.btn_trainepoch.clicked.connect(lambda check: self.Show_Train1epoch())
        self.btn_showresulttrain.clicked.connect(lambda check: self.Show_ImageGraphTrain())
        self.btn_inferencetrain.clicked.connect(lambda check: self.Inference_ResultTraining())
        self.btn_imgflipping.clicked.connect(lambda check: self.Flip_Image())
        self.btn_blending.clicked.connect(lambda check: self.Blend_Image())
        self.btn_globthreshold.clicked.connect(lambda check: self.Global_Threshold())
        self.btn_localthreshold.clicked.connect(lambda check: self.Local_Threshold())
        self.btn_gaussian.clicked.connect(lambda check: self.ConvertImagetogray())
        self.btn_sobelX.clicked.connect(lambda check: self.Convert_Edge_To_Detect("Horizontal"))
        self.btn_sobelY.clicked.connect(lambda check: self.Convert_Edge_To_Detect("Vertical"))
        self.btn_magnitude.clicked.connect(lambda check: self.Convert_Edge_To_Detect("Magnitude"))
        self.cbb_selectimage.addItems(ListImage)
        self.cbb_selecttrain.addItems(ListTrain)

    def Processing_Detect(self):
        List_Objpoints = [] # 3d point in real world space
        List_Imgpoints = [] # 2d points in image plane.
        List_Processimage = []
        Chesscol = 11
        Chessrow = 8
      
        objp = np.zeros((Chessrow*Chesscol,3), np.float32)
        objp[:,:2] = np.mgrid[0:Chesscol,0:Chessrow].T.reshape(-1,2)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if len(ListImage) != 1:
            for filename in ListImage:
                Image_path = FolderImage + filename
                image = cv2.imread(Image_path)
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(image, (Chesscol,Chessrow), None)
                if ret == True:
                    
                    # corners2 = cv2.cornerSubPix(image,corners,(11,11),(-1,-1),criteria)
                    Image_detected = cv2.drawChessboardCorners(image, (Chesscol,Chessrow), corners, ret)
                    List_Objpoints.append(objp)
                    List_Imgpoints.append(corners)
                    List_Processimage.append(Image_detected)
                else:
                    print("No Image is detected")
            return List_Processimage, List_Objpoints, List_Imgpoints, image

    def Find_Corners(self):
        print("--------Find Corners--------")
        List_Processimage, List_Objpoints, List_Imgpoints, image = Process_Image.Processing_Detect(self)
        w = image.shape[:2][0]
        h = image.shape[:2][1]
        for indeximage in List_Processimage:
            resize_IMG = cv2.resize(indeximage, (int(w*0.5), int(h*0.5))) 
            cv2.imshow("Image", resize_IMG)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Intrinsic(self):
        print("--------Intrinsic--------")
        List_Processimage, List_Objpoints, List_Imgpoints, image = Process_Image.Processing_Detect(self)
        reprojection_error, camera_matrix, distortion_coefficient, rotation_v, translation_v = cv2.calibrateCamera(List_Objpoints, List_Imgpoints, image.shape[:2],None,None )
        print(camera_matrix)
        self.text_showvalue.setText(str(camera_matrix))  

    def Extrinsic(self):
        Nameimg = self.cbb_selectimage.currentText()
        ImagePath = FolderImage + Nameimg
        index = int(ListImage.index(Nameimg))
        print("--------Extrinsic of %s--------" %ListImage[index] )

        List_Processimage, List_Objpoints, List_Imgpoints, image = Process_Image.Processing_Detect(self)
        reprojection_error, camera_matrix, distortion_coefficient, rotation_v, translation_v = cv2.calibrateCamera(List_Objpoints, List_Imgpoints, image.shape[:2],None,None)
        
        Rotation_Mtx,_ = cv2.Rodrigues(rotation_v[index])
        Translation_Mtx = translation_v[index]
        Extrinsic_matrix = np.array([[Rotation_Mtx[0][0],Rotation_Mtx[0][1],Rotation_Mtx[0][2],Translation_Mtx[0][0]],
                                    [Rotation_Mtx[1][0],Rotation_Mtx[1][1],Rotation_Mtx[1][2],Translation_Mtx[1][0]],
                                    [Rotation_Mtx[2][0],Rotation_Mtx[2][1],Rotation_Mtx[2][2],Translation_Mtx[2][0]],
                                    ])

        self.text_showvalue.setText("Extrinsic of %s:\n%s" %(ListImage[index],str(Extrinsic_matrix))) 
        print(Extrinsic_matrix)

    def Distortion(self):
        print("--------Distortion--------")
        List_Processimage, List_Objpoints, List_Imgpoints, image = Process_Image.Processing_Detect(self)
        reprojection_error, camera_matrix, distortion_coefficient, rotation_v, translation_v = cv2.calibrateCamera(List_Objpoints, List_Imgpoints, image.shape[:2],None,None)
        print(distortion_coefficient)
        self.text_showvalue.setText(str(distortion_coefficient))  


    def Augmented_Reality(self):
        
        corners = np.float32([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0],[0, 0, -2]])
        List_Processimage, List_Objpoints, List_Imgpoints, image = Process_Image.Processing_Detect(self)
        reprojection_error, camera_matrix, distortion_coefficient, rotation_v, translation_v = cv2.calibrateCamera(List_Objpoints, List_Imgpoints, image.shape[:2],None,None)
        text = ""
        for index in range(5):
           # index = 0
            Rotation_Mtx= rotation_v[index]
            Translation_Mtx = translation_v[index]
            pyramid_corners_2d,_ = cv2.projectPoints(corners,Rotation_Mtx,Translation_Mtx,camera_matrix,distortion_coefficient)
            print("Processing %s........" %ListImage[index])
            text += "Processing %s........ \n" %ListImage[index]
            self.text_showvalue.setText(text)  
            ImagePath = FolderImage + ListImage[index]
            img=cv2.imread(ImagePath) #load the correct image
            line_width = 10
            red=(0,0,255) #red (in BGR)
            blue=(255,0,0) #blue (in BGR)
            cv2.line(img, tuple(pyramid_corners_2d[0][0]), tuple(pyramid_corners_2d[1][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[1][0]), tuple(pyramid_corners_2d[2][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[2][0]), tuple(pyramid_corners_2d[3][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[3][0]), tuple(pyramid_corners_2d[0][0]),red,line_width)

            cv2.line(img, tuple(pyramid_corners_2d[0][0]), tuple(pyramid_corners_2d[4][0]),blue,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[1][0]), tuple(pyramid_corners_2d[4][0]),blue,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[2][0]), tuple(pyramid_corners_2d[4][0]),blue,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[3][0]), tuple(pyramid_corners_2d[4][0]),blue,line_width)
            resize_IMG = cv2.resize(img, (int(image.shape[:2][0]*0.5), int(image.shape[:2][1]*0.5))) 
            cv2.imshow("Image", resize_IMG)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
        
    def Image_Transformation(self):
        Path_Img    = FolderTransImg + 'OriginalTransform.png'
        Image       = cv2.imread(Path_Img)
        height      = Image.shape[0]
        width       = Image.shape[1]
        try:
            ScaleImg    = float(self.lineEdit_scale.text())
            AngleImg    = int(self.lineEdit_angle.text())
            TxImg       = int(self.lineEdit_Tx.text())
            TyImg       = int(self .lineEdit_Ty.text())

            # Rotation Image
            RotationImg = cv2.getRotationMatrix2D((130,125),AngleImg,ScaleImg)
            DstImg1 = cv2.warpAffine(Image,RotationImg,(width,height))
            
            # Translation Image
            TranslationImg = np.float32([[1,0,TxImg],[0,1,TyImg]])
            DstImg2 = cv2.warpAffine(DstImg1,TranslationImg,(width,height))
            
            # Show Image
            # Orginal Image
            cv2.imshow('Image_process',Image)
            cv2.waitKey(1000)
            # Procced Image
            cv2.imshow('img',DstImg2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
            
    def Find_Contour(self):
        Path_Img    = FolderTransImg + 'Contour.png'
        Image       = cv2.imread(Path_Img)
        gray        = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        edged       = cv2.Canny(gray, 30, 200) 
        text = "Processing Image........ "
        self.text_showvalue.setText(text)  
        cv2.imshow('Image', Image) 
        cv2.waitKey(1000) 
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        cv2.drawContours(Image,contours, -1, (0, 0, 255), 3) 
        cv2.imshow('Contours', Image) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Perspective_Transformation(self):
        text = "Please click 4 points of corners. \nRemember: Start from top-left corner and then click clock-wise "
        self.text_showvalue.setText(text)  
        cv2.namedWindow('Perspective Image')
        cv2.setMouseCallback('Perspective Image',draw_circle)
        while(True):
            cv2.imshow('Perspective Image',Image_Perspective)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        if len(positions) > 0:
            src = np.float32([[positions[0]], [positions[1]], [positions[2]], [positions[3]]])
            dst = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
            matrix = cv2.getPerspectiveTransform(src,dst)
            result = cv2.warpPerspective(Image_Perspective,matrix, (450,450),flags=cv2.INTER_LINEAR)
            cv2.namedWindow('Perspective Result Image')
            cv2.imshow("Perspective Result Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ############################################## The End CVDL Homework ######################################
    ############################################## OPEN CV Homework #########################################

    
    def Load_Image_File(self):
        Path_Img    = FolderTransImg + 'dog.bmp'
        Image       = cv2.imread(Path_Img)
        Height      = Image.shape[0]
        Width       = Image.shape[1]
        text        = "Height = %s \nWidth = %s" %(Height,Width)
        self.text_showvalue.setText(text)  
        print(text)
        cv2.imshow("Image", Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def Color_Conversion(self):
        Path_Img    = FolderTransImg + 'color.png'
        Image       = cv2.imread(Path_Img)
        text = "Processing Image........ "
        self.text_showvalue.setText(text) 
        r_channel, b_channel, g_channel = cv2.split(Image) 
        img_RBG = cv2.merge((b_channel, g_channel, r_channel))
        cv2.imshow("Orginal Image", Image)
        cv2.waitKey(1500)
        cv2.imshow("Conversion Image", img_RBG)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Flip_Image(self):
        Path_Img         = FolderTransImg + 'dog.bmp'
        Image            = cv2.imread(Path_Img)
        flipHorizontal   = cv2.flip(Image, 1)
        cv2.imshow('Original image', Image)
        cv2.waitKey(1000)
        cv2.imshow('Flipped horizontal image', flipHorizontal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def nothing(self,x):
        pass

    # ________________________________

    def Blend_Image(self):
        Path_Img         = FolderTransImg + 'dog.bmp'
        Image            = cv2.imread(Path_Img)
        flipHorizontal   = cv2.flip(Image, 1)
        Image_pre             = cv2.addWeighted(Image, 0.5, flipHorizontal, 0.5, 0) 
        cv2.namedWindow("Blending Image")
        cv2.createTrackbar('Blending',"Blending Image",0,100,self.nothing)
        while(True):
            cv2.imshow('Blending Image', Image_pre)
            if cv2.waitKey(1) == 27:
                break

            alpha = cv2.getTrackbarPos('Blending',"Blending Image")/100
            beta = 1- alpha
            Image_pre = cv2.addWeighted(Image, alpha, flipHorizontal, beta, 0)

        cv2.destroyAllWindows()
    # _______________________

    def Global_Threshold(self):
        Path_Img         = FolderTransImg + 'QR.png'
        Image            = cv2.imread(Path_Img)
        grayscaled       = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
        ret,thresImage   = cv2.threshold(grayscaled,80,255,cv2.THRESH_BINARY)
        cv2.imshow('Original image', Image)
        cv2.waitKey(1000)
        cv2.imshow('Global Threshold Image', thresImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # _________________________________

    def Local_Threshold(set):
        Path_Img         = FolderTransImg + 'QR.png'
        Image            = cv2.imread(Path_Img)
        grayscaled       = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        threshImage      = cv2.adaptiveThreshold(grayscaled,255,-cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19,-1)
        cv2.imshow('Original image', Image)
        cv2.waitKey(1000)
        cv2.imshow('Local Threshold image', threshImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def ConvertImagetogray(self):
        print("Processing Image. Please, Waiting......")
        text = "Processing Image. Please, Waiting......"
        self.text_showvalue.setText(text) 
        Path_Img    = FolderTransImg + 'School.jpg'
        Image       = cv2.imread(Path_Img)
        grayimg     = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        h, w        = grayimg.shape[:2]
        smoothimage  = grayimg

        for y in range(1, w-1):
            for x in range(1, h-1):     
                px1 = smoothimage[x-1][y-1] #0/0
                px2 = smoothimage[x-1][y] #0/1
                px3 = smoothimage[x-1][y+1] #0/2
                px4 = smoothimage[x][y-1] #1/0
                px5 = smoothimage[x][y] #1/1
                px6 = smoothimage[x][y+1] #1/2
                px7 = smoothimage[x+1][y-1] #2/0
                px8 = smoothimage[x+1][y] #2/1
                px9 = smoothimage[x+1][y+1] #2/2

                average = px1/9. + px2/9. + px3/9. + px4/9. + px5/9. + px6/9. + px7/9. + px8/9. + px9/9.

                smoothimage[x][y] = average   #1/1
        cv2.imshow('Ogrinal image', Image)
        cv2.waitKey(500)
        cv2.imshow('Smooth Gray Image', smoothimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Convert_Edge_To_Detect(self, CheckButton):
        print("Processing Image. Please, Waiting......")
        text = "Processing Image. Please, Waiting......"
        self.text_showvalue.setText(text) 
        Path_Img          = FolderTransImg + 'School.jpg'
        Image             = cv2.imread(Path_Img)
        horizontal        = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # s2
        vertical          = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # s1
        h, w              = Image.shape[:2]
        img               = Image
        DetectEdge_Image  = Image
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                verticalGrad    = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                                    (horizontal[0, 1] * img[i - 1, j]) + \
                                    (horizontal[0, 2] * img[i - 1, j + 1]) + \
                                    (horizontal[1, 0] * img[i, j - 1]) + \
                                    (horizontal[1, 1] * img[i, j]) + \
                                    (horizontal[1, 2] * img[i, j + 1]) + \
                                    (horizontal[2, 0] * img[i + 1, j - 1]) + \
                                    (horizontal[2, 1] * img[i + 1, j]) + \
                                    (horizontal[2, 2] * img[i + 1, j + 1])

                horizontalGrad  = (vertical[0, 0] * img[i - 1, j - 1]) + \
                                    (vertical[0, 1] * img[i - 1, j]) + \
                                    (vertical[0, 2] * img[i - 1, j + 1]) + \
                                    (vertical[1, 0] * img[i, j - 1]) + \
                                    (vertical[1, 1] * img[i, j]) + \
                                    (vertical[1, 2] * img[i, j + 1]) + \
                                    (vertical[2, 0] * img[i + 1, j - 1]) + \
                                    (vertical[2, 1] * img[i + 1, j]) + \
                                    (vertical[2, 2] * img[i + 1, j + 1])

                # Edge Detection
                if CheckButton == "Horizontal":
                    Horizontal  =  np.sqrt(pow(horizontalGrad, 2.0))
                    DetectEdge_Image[i - 3, j - 3 ] = Horizontal
                elif CheckButton == "Vertical":
                    Vertical    =  np.sqrt(pow(verticalGrad, 2.0))
                    DetectEdge_Image[i - 3,  j - 3 ]   = Vertical
                else:
                    Horizontal  =  np.sqrt(pow(horizontalGrad, 2.0))
                    Vertical    =  np.sqrt(pow(verticalGrad, 2.0))
                    Magnitude   =  Vertical + Horizontal
                    DetectEdge_Image[i - 3,  j - 3 ]   = Magnitude

        DetectEdge_Image = cv2.cvtColor(DetectEdge_Image, cv2.COLOR_BGR2GRAY)
        if CheckButton == "Horizontal":
            cv2.imshow("Hozonrital_Edge_Detect", DetectEdge_Image)

        elif CheckButton == "Vertical":
            cv2.imshow("Vertical_Edge_Detect", DetectEdge_Image)
        else:
            cv2.imshow("Magnitude_Edge_Detect", DetectEdge_Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    ############################################## Training Cifar10 - Training MNIST ######################################
    ############################################## Training Cifar10 - Training MNIST ######################################
    def Pre_training(self):
        transform = transforms.ToTensor()
        EPOCH , BATCH_SIZE, LR = self.Parameter_Training()
        if self.cbb_selecttrain.currentText() == "Training Cifar-10" :
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
    def Parameter_Training(self):
        EPOCH = 1
        BATCH_SIZE = 32     
        LR = 0.001  
        
        return EPOCH , BATCH_SIZE, LR
    
    
    def Load_Dataset_Image(self):
        trainset, trainloader, testset, testloader = self.Pre_training()
        EPOCH , BATCH_SIZE, LR = self.Parameter_Training()
        fig, axes = plt.subplots(1, 10, figsize=(12,5))
        text = ''
        for i in range(10):
            index = random.randint(0,9999)
            image = trainset[index][0] 
            label = trainset[index][1] 
            if self.cbb_selecttrain.currentText() == "Training Cifar-10":
                text += "The picture %s is showing: The %s\n" %(str(i+1),Label_Cifar10[int(label)])
                Showimage = image.numpy()
                Showimage = np.transpose(Showimage, (1, 2, 0))
                Showlabel = Label_Cifar10[int(label)]
            else:
                text += "The picture %s is showing: Number %s\n" %(str(i+1),Label_MNIST[int(label)])
                Showimage = image.numpy()
                Showimage = np.squeeze(Showimage)
                Showlabel = Label_MNIST[int(label)]
            
            axes[i].imshow(Showimage)
            axes[i].set_xlabel(Showlabel)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        print(text)
        self.text_showvalue.setText(text)  
        plt.show()
    
    def Show_ParameterTranining(self):
        EPOCH , BATCH_SIZE, LR = self.Parameter_Training()
        text = """Hyperparameter\nBATCH_SIZE = %s\nLearning rate = %s\nOptimizer = SGD """ %(BATCH_SIZE, LR)
        self.text_showvalue.setText(text)  
        print(text)

    def Processing_Training(self):
        trainset, trainloader, testset, testloader = self.Pre_training()
        EPOCH , BATCH_SIZE, LR = self.Parameter_Training()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.cbb_selecttrain.currentText() =="Training Cifar-10":
            net = LeNet_Cifar10().to(device)
        else:
            net = LeNet_MNIST().to(device)
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        # print(trainloader)
        List_loss = []
     
        for epoch in range(EPOCH):
            sum_loss = 0.0
            for i, data in enumerate(trainloader):
            
                images = data[0]
                labels = data[1]
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                List_loss.append(loss.item())
                if i % 100 == 99:
                    text = '[Epoch = %d, Batch = %d] loss: %.03f' % (epoch + 1, i + 1, loss.item())
                    print(text)
                    sum_loss = 0.0

            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                text = 'Exactly Percent of %dth epoch is: %d%%' % (epoch + 1, (100 * correct / total))
                print(text)
        return List_loss
    def Show_Train1epoch(self):
        text = 'Done train !'
        self.text_showvalue.setText(text) 
        List_loss = self.Processing_Training()
        plt.plot(List_loss)
        plt.title("Loss per Iteration for 1 Epoch")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        if self.cbb_selecttrain.currentText() == "Training Cifar-10":
            plt.ylim([2.27,2.33])
            plt.xlim([0, 1200])
        else:
            plt.ylim([2.15,2.35])
            plt.xlim([0, 1000])
        plt.show()
    
    def Show_ImageGraphTrain(self):
        text = 'Graph is showing...'
        self.text_showvalue.setText(text) 
        if self.cbb_selecttrain.currentText() == "Training Cifar-10":
            Image = FolderTransImg + "TrainingModelCifar10.JPG"
        else:
            Image = FolderTransImg + "TrainingModelMNIST.JPG"
        
        Image = cv2.imread(Image)
        cv2.imshow("Accuracy Train", Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def Inference_ResultTraining(self):
        try: 
            index    = int(self.lineEdit_testimg.text())
        except:
            print("Please, Enter index image")
            self.text_showvalue.setText("Please, Enter index image") 
            return
        trainset, trainloader, testset, testloader = self.Pre_training()
        EPOCH , BATCH_SIZE, LR = self.Parameter_Training()
        if self.cbb_selecttrain.currentText() == "Training Cifar-10":
            PATH_Model = "./model_Cifar10/LenetCifar10_028.pth"
            net = LeNet_Cifar10()
        else:
            PATH_Model = "./model_MNIST/Lenet_MNIST_028.pth"
            net = LeNet_MNIST()
        net.load_state_dict(torch.load(PATH_Model))
        net.eval()
        imagetest = trainset[index][0]
        labeltest = trainset[index][1] 

        Image_to_test = imagetest.unsqueeze(0)
        outputs = net(Image_to_test)
        _, predicted = torch.max(outputs.data, 1)
       
        List_x = []
        List_percent = []
        sumx = 0
        for x in outputs.data[0]:
            if (x < 0):
                x =0.0
            sumx += x
            List_x.append(float(x))
        List_percent = [y/sumx for y in List_x]
        fig, axs = plt.subplots(1,2,figsize=(14,6))
        if self.cbb_selecttrain.currentText() == "Training Cifar-10":
            Result_predict = Label_Cifar10[int(predicted[0])]
            ImageShow= imagetest.numpy()
            ImageShow= np.transpose(ImageShow, (1, 2, 0))
            rects1 = axs[1].bar(Label_Cifar10, List_percent, color='b')
            text = 'The result is the %s' %Result_predict
        else:
            ImageShow = imagetest.numpy()
            ImageShow = np.squeeze(ImageShow)
            Result_predict = int(predicted[0])
            rects1 = axs[1].bar(Label_MNIST, List_percent, color='b')
            text = 'The result is number %s' %Result_predict
        
        self.text_showvalue.setText(text) 
        print(text)
        axs[1].set_ylim([0,0.8])
        axs[0].imshow(ImageShow)
        axs[0].set_xlabel(Result_predict)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        plt.show()

    ############################################## CLASS TRAINING LENET FOR CIFAR10 AND MNIST ########################################################
class LeNet_Cifar10(nn.Module):
    def __init__(self):
        super(LeNet_Cifar10, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet_MNIST(nn.Module):
    def __init__(self):
        super(LeNet_MNIST, self).__init__()
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
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

############################################################################################

positions = []
def draw_circle(event,x,y,flags,param):
    
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(Image_Perspective,(x,y),7,(255,0,0),-1)
        if(len(positions) < 4):
            positions.append([x,y])

def mainprocessing():
	ui = Process_Image()
	Opencv_hw1.show()
	sys.exit(app.exec_())

if __name__ == "__main__":         
    mainprocessing()
