import cv2
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import os
import sys
import cv2
from skimage import measure, morphology, segmentation
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QIcon,QPixmap,QImage
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from Interface_Detect_Cancer_v1 import Ui_Form
from os.path import expanduser
import sys
sys.path.append("../DSB2017")
from layers import nms,iou  

# from ..DSB2017.layers import nms,iou  
# # nms,iou



app = QtWidgets.QApplication(sys.argv)
Project = QtWidgets.QWidget()


class TWSignals(QtCore.QObject):
	switch_window = QtCore.pyqtSignal(object)

class Process_Image(Ui_Form):
    signals = TWSignals()
    def __init__(self):
        super(Process_Image,self).__init__()
        self.setupUi(Project)
        self.btnprocessing.clicked.connect(lambda check: self.Showimg_preprocessing())
        self.btnResult.clicked.connect(lambda check: self.Detection_Training())
        self.btnBrowser.clicked.connect(lambda check: self.Browser_directory())
        self.comboX.currentIndexChanged.connect(self.onCurrentIndexChanged)
        # self.Slider_original.valueChanged.connect(lambda check: self.Choose_Slice_Original())
        self.Slider_process.valueChanged.connect(lambda check: self.choose_slice()) #valueChanged


    def Browser_directory(self):
        print("------Open Browser-------")
        dialog_style = QFileDialog.DontUseNativeDialog
        dialog_style |= QFileDialog.DontUseCustomDirectoryIcons
        self.input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:',  "../Image",)
        self.link_Browser.setText(self.input_dir) 
        self.Path_Image = ("/").join(self.input_dir.split("/")[0:-1])
        self.List_Path_Image = [f for f in os.listdir(self.Path_Image)]
        
        self.List_Image = [ f for f in os.listdir(self.input_dir) if f.endswith('.npy')]
        self.List_Image.sort(reverse=True)
        self.comboX.addItems(self.List_Image)
        self.Path_Original = self.input_dir + "/" + str(self.comboX.currentText())
        self.Path_Preprocessing = self.Path_Image + "/" + self.List_Path_Image[1]
        self.Path_Detection = self.Path_Image + "/" + self.List_Path_Image[0]
        #print(self.List_Path_Image)
    def onCurrentIndexChanged(self, indeximg):
        self.Path_img = self.input_dir + "/" + str(self.comboX.currentText())
        
        self.Load_numpy_img = np.load(self.Path_img, allow_pickle=True)
    
        cv2.imwrite(self.Path_img.replace(".npy",".jpg"),self.Load_numpy_img[0,0])
        img = QPixmap(self.Path_img.replace(".npy",".jpg") )
        self.Slider_process.setValue(0)
        self.label_Input.setPixmap(img)
        self.label_Input.show()


##############################################################################
    def Show_Preprocessing(self):
        """"""
        self.Path_preprocessing = self.input_dir.split("/")[0:-1]
        self.Path_preprocessing = ("/").join(self.Path_preprocessing) + "2_Preprocessing_img"
        #print(self.Path_preprocessing)
        """"""

        Path_Preprocessing_img = self.Path_Preprocessing + "/" + str(self.comboX.currentText())
        #print(self.Path_Preprocessing)
        self.Load_numpy_img = np.load(Path_Preprocessing_img)
        
        cv2.imwrite(Path_Preprocessing_img.replace(".npy",".jpg"),self.Load_numpy_img[0,0])
        img = QPixmap(Path_Preprocessing_img.replace(".npy",".jpg") )

        self.label_Pre_Processing.setPixmap(img)
        # self.label_Pre_Processing.setGeometry(540, 120, 500, 400)
        self.label_Pre_Processing.show()

    def Detection_Training(self):
        from shutil import copyfile
        self.Path_preprocessing1 = self.input_dir.split("/")[0:-1]
        self.Path_preprocessing_img = ("/").join(self.Path_preprocessing1) + "/2Preprocessing_img/"
        self.Path_preprocessing_label = ("/").join(self.Path_preprocessing1) + "/2Preprocessing_label/"
        self.Path_preprocessing_Detect = ("/").join(self.Path_preprocessing1) + "/3Detection_img/"
        self.Path_preprocessing_image = ("/").join(self.Path_preprocessing1) + "/preprocess_result/"
        self.Path_pp = self.Path_img.replace('original','preprocess_result').replace('_origin','pp_clean')
        # Lisst_del = [f for in os.listdir(self.Path_Preprocessing)]
        for i in os.listdir(self.Path_preprocessing_img):
            os.remove(self.Path_preprocessing_img + i)
        for i in os.listdir(self.Path_preprocessing_label):
            os.remove(self.Path_preprocessing_label + i)
        for i in os.listdir(self.Path_preprocessing_Detect):
            os.remove(self.Path_preprocessing_Detect + i)
        #print(self.Path_pp)
        copyfile(self.Path_pp, self.Path_preprocessing_img + os.path.basename(self.Path_pp).replace("pp_clean","_clean"))
        copyfile(self.Path_pp.replace("pp_clean","_label"), self.Path_preprocessing_img.replace('_img','_label') + os.path.basename(self.Path_pp.replace("pp_clean","_label")))
       
        
        sys.path.append("../DSB2017")
        from main import main_init
        main_init()
        Process_Image.Detection_Image(self)


    def Detection_Image(self):
        Path_Detection_img = self.Path_preprocessing_img.replace("2Preprocessing_img","3Detection_img") + "/" + str(self.comboX.currentText()).replace("_origin","_pbb")
        Path_Preprocessing_img = self.Path_preprocessing_img + "/" + str(self.comboX.currentText()).replace("_origin","_clean")
        [a,img] = np.load(Path_Preprocessing_img, allow_pickle=True)

        pbb = np.load(Path_Detection_img)

        pbb = pbb[pbb[:,0]>-1]
        pbb = nms(pbb,0.05)
        box = pbb[0].astype('int')[1:]
        print(box)
        text = "Slice: %s  Height: %s  Width: %s  Chanels: %s" %(box[0],box[1],box[2],box[3])
        self.result_Browser.setText(text)
        ax = plt.subplot(1,1,1)
        ax.axis('off')
        plt.imshow(img[0,box[0]],'gray')
        plt.axis('off')
        rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
        ax.add_patch(rect)
        # plt.show()
        ax.figure.savefig(Path_Detection_img.replace(".npy",".jpg"))
        plt.cla() 
        img = QPixmap(Path_Detection_img.replace(".npy",".jpg"))
        self.label_Output.setPixmap(img)
        # self.label_Output.setGeometry(1060, 120, 500, 400)
        self.label_Output.show()
        
    def choose_slice(self):
        
               
        self.Slider_process.valueChanged.connect(self.changeValue) # [Phu] valueChanged.connect(self.changeValue)
        self.Path_img = self.input_dir + "/" + str(self.comboX.currentText())
        self.Path_pp = self.Path_img.replace('original','preprocess_result').replace('_origin','pp_clean')
        [self.spacing_pre, self.img_pre] = np.load(self.Path_pp , allow_pickle=True)
        self.Slider_process.setMinimum(0)
        self.Slider_process.setMaximum(self.img_pre.shape[1] - 1 )
        
        #self.input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:',  "../Image",)
        
        self.Load_numpy_img = np.load(self.Path_img)
        
        # self.setWindowTitle("SpinBox demo")

    def changeValue(self):
        
        pre_size = self.Slider_process.value()
        self.result_slide_process.setText(str(pre_size))
        #print(size)
        #new_size = img.shape[1] spacing[0] 
        img_size = int(pre_size / self.spacing_pre[0])
   
        cv2.imwrite(self.Path_pp.replace('npy','png'),self.img_pre[0,pre_size])
        self.im = QPixmap(self.Path_pp.replace('npy','png')) 
        self.label_Pre_Processing.setPixmap(self.im)
        self.label_Pre_Processing.show()        
        
        print(pre_size, self.spacing_pre, img_size)

        """Show Original Image from Pre Number"""

        print(self.Load_numpy_img.shape)
        self.Load_numpy_img_new = np.expand_dims(self.Load_numpy_img, axis=0)
        # cv2.imwrite(self.Path_img.replace(".npy",".jpg"),self.Load_numpy_img_new[0,img_size])

        # plt.imshow(self.Load_numpy_img_new[0,img_size],'gray')
        plt.imsave(self.Path_img.replace(".npy",".jpg"),self.Load_numpy_img_new[0,img_size], cmap='gray')
        img = QPixmap(self.Path_img.replace(".npy",".jpg") )


        self.label_Input.setPixmap(img)
        self.label_Input.show()


    #""""""
#################################### processing ###############################        
        # self.im1 = QPixmap(joinfolder + "/Image_Pr" +  "/" + self.Path_Original ) 
        
        # # self.Input_label = QLabel()
        """
        img = np.load('./Image/Image_Pr/001_clean.npy')
        
        import matplotlib.pyplot as plt
        cv2.imwrite("./Image/Image_Pr/001_clean.jpg",img[0,25])
        # cv2.imshow("aa",img[0,25])

        # cv2.waitKey(0)
        # print(img[0,25,12,23])
        # img = Image.fromarray(img[0,25,5], 'RGB')
        # # print(img.shape)
        self.im = QPixmap('./Image/Image_Pr/001_clean.jpg' ) 
        self.label_Pre_Processing.setPixmap(self.im)
        self.label_Pre_Processing.setGeometry(450, 110, 411, 331) 
        self.label_Pre_Processing.show()
        # pbb = np.load('./bbox_result/000_pbb.npy')

        # pbb = pbb[pbb[:,0]>-1]
        # pbb = nms(pbb,0.05)
        # box = pbb[0].astype('int')[1:]
        # print(box)

        
        # ax = plt.subplot(1,1,1)
        # plt.imshow(img[0,box[0]],'gray')
        # plt.axis('off')


        # ListImageOut = [h for h in os.listdir(joinfolder) if h.endswith(".jpg")]
        # self.im2 = QPixmap(joinfolder + "/Image_Out" + "/" + self.Path_Original ) 
        # # self.Input_label = QLabel()
        # self.label_Output.setPixmap(self.im2) 
        # self.label_Output.setGeometry(880, 110, 411, 331) 
        # self.label_Output.show()
        """

    # def ImageOut(self):

    #     FolderImageOut = "./Image_Out/"
    #     ListImageOut = [h for h in os.listdir(FolderImageOut) if h.endswith(".jpg")]

    # def ShowImage(self):
    #     lb = QtGui.QLabel(self)




def Main_Function():
	ui = Process_Image()
	Project.show()
	sys.exit(app.exec_())
    # label.show()

if __name__ == "__main__":         
    Main_Function()
