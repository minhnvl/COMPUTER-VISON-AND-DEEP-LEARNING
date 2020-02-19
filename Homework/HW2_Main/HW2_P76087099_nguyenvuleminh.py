import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from HW2_interface import Ui_Widget_HW2
import sys

# Define path dir
FolderImage = "./Image/"
ListImage = [f for f in os.listdir(FolderImage) if f.endswith(".bmp")]
# Define Interface
app = QtWidgets.QApplication(sys.argv)
Opencv_hw2 = QtWidgets.QWidget()
    
class TWSignals(QtCore.QObject):
	switch_window = QtCore.pyqtSignal(object)

class Process_Image(Ui_Widget_HW2):
    signals = TWSignals()
    def __init__(self):
        super(Process_Image,self).__init__()
        self.setupUi(Opencv_hw2)
        self.btn_disparity.clicked.connect(lambda check: self.Disparity_CVDL())
        self.btn_ncc.clicked.connect(lambda check: self.NCC_CVDL())
        self.btn_keypoints.clicked.connect(lambda check: self.Show_Keypoint())
        self.btn_MK.clicked.connect(lambda check: self.MatchKP_SIFT())
        self.btn_disparity2.clicked.connect(lambda check: self.Disparity_OpenCV())
        self.btn_BS.clicked.connect(lambda check: self.Background_Subtraction())
        self.btn_preprocessing.clicked.connect(lambda check: self.Pre_video_Tracking())
        self.btn_videotracking.clicked.connect(lambda check: self.videotracking())
        self.btn_AR.clicked.connect(lambda check: self.Augmented_Reality())
        
    ####### Q1: Stereo Disparity Map #########
    def Disparity_CVDL(self):
        text =  "Processing Disparity...."
        print(text)
        self.textEdit_show.setText(text) 

        imgL = cv2.imread(FolderImage + 'imL1.png',0)
        imgR = cv2.imread(FolderImage + 'imR2.png',0)

        StereoBM_create = cv2.StereoBM_create(numDisparities=64, blockSize=9)
        Disparity = StereoBM_create.compute(imgL,imgR)

        plt.imshow(Disparity,'gray')
        plt.show()

    ####### Q2: Normalized Cross Correlation #########
    def NCC_CVDL(self):
        text =  "Processing Normalized Cross Correlation...."
        print(text)
        self.textEdit_show.setText(text) 

        img_NCC = cv2.imread(FolderImage + 'ncc_img.jpg')
        img_Template = cv2.imread(FolderImage + 'ncc_template.jpg')
        img_NCC_Gray = cv2.cvtColor(img_NCC, cv2.COLOR_BGR2GRAY)
        img_Template_Gray = cv2.cvtColor(img_Template, cv2.COLOR_BGR2GRAY)

        Height = img_Template_Gray.shape[0]
        Width = img_Template_Gray.shape[1]

        res = cv2.matchTemplate(img_NCC_Gray,img_Template_Gray,cv2.TM_CCOEFF_NORMED)
        cv2.imshow('Matching_Feature',res)

        for i in range(5):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + Width, top_left[1] + Height)
            cv2.rectangle(img_NCC,top_left, bottom_right, 0, 2)
            cv2.rectangle(res,top_left, bottom_right, 0, -1)
                
        cv2.imshow('Detected_Template',img_NCC)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ####### Q3: SIFT #########
    # Edit Parameter
    def Para_KeyPoint_SIFT(self):
        img1 = cv2.imread(FolderImage + 'Aerial1.jpg',0)          # queryImage
        img2 = cv2.imread(FolderImage + 'Aerial2.jpg',0) 
        # Parameter SIFT
        sift = cv2.xfeatures2d.SIFT_create(15,3,0.001,20,40)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=2)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        list_check=[]

        # ratio test as per Lowe's paper
        check = kp2[matches[1][0].queryIdx].pt
        for i,(m,n) in enumerate(matches):
            if  m.distance < 0.48*n.distance:
                matchesMask[i]=[1,0]
                if i == 7:
                    kp2[n.queryIdx].pt = check
                # print(i)
            else:
                if i!=1 and i != 8:
                    kp1[m.queryIdx].pt = (0,0)
                if i != 10 and i != 1:
                    kp2[n.queryIdx].pt = check
        #Drawkeypoint
        keypoint1_img1 = cv2.drawKeypoints(img1,kp1,img1,(255,0,0),flags=cv2.DrawMatchesFlags_DEFAULT)
        keypoint2_img2 = cv2.drawKeypoints(img2,kp2,img2,(255,0,0),flags=cv2.DrawMatchesFlags_DEFAULT)
        return keypoint1_img1,keypoint2_img2,img1,img2,matches,matchesMask,kp1,kp2

    # Q3_1: Show Keypoint
    def Show_Keypoint(self):
        text =  "Showing 6 feature points..."
        print(text)
        self.textEdit_show.setText(text) 

        keypoint1_img1,keypoint2_img2,img1,img2,matches,matchesMask,kp1,kp2 = Process_Image.Para_KeyPoint_SIFT(self)

        cv2.imwrite( FolderImage + "SIFT_Show_Keypoints.jpg",np.hstack([keypoint1_img1, keypoint2_img2]))
        cv2.imshow("Key_Points", np.hstack([keypoint1_img1, keypoint2_img2]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Q3_2: Matches Keypoint
    def MatchKP_SIFT(self):
        text =  "Showing the matched feature points between two images..."
        print(text)
        self.textEdit_show.setText(text) 

        keypoint1_img1,keypoint2_img2,img1,img2,matches,matchesMask,kp1,kp2 = Process_Image.Para_KeyPoint_SIFT(self)
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = 0)

        Img_Matches = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        
        # plt.imshow(img3,),plt.show()
        cv2.imshow("MatchKP_SIFT", Img_Matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

############################# OPEN CV #####################################

    ####### Q1: Stereo Disparity Map #########
    def Disparity_OpenCV(self):
        text =  "Processing Disparity...."
        print(text)
        self.textEdit_show.setText(text) 

        imgL = cv2.imread( FolderImage + 'imL.png',0)
        imgR = cv2.imread( FolderImage + 'imR.png',0)
        left_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=64,            
                blockSize=9,
                P1=130,    
                P2=1900,
                disp12MaxDiff=1,
                uniquenessRatio=5,
                speckleWindowSize=200,
                speckleRange=3,
                # preFilterCap=30,
                mode=cv2.StereoSGBM_MODE_SGBM
            )
        disparity = left_matcher.compute(imgL,imgR)
        plt.imshow(disparity,'gray')
        plt.show()
    
    ####### Q2: Background Subtraction   #########
    def Background_Subtraction(self):
        text =  "Processing Background Subtraction...."
        print(text)
        self.textEdit_show.setText(text) 

        cap = cv2.VideoCapture(FolderImage +  "bgSub.mp4")
        backSub = cv2.createBackgroundSubtractorMOG2(history = 50,	varThreshold = 150,detectShadows = True)
        kernel=np.ones((5,5),np.uint8)
        while True:
            ret, frame = cap.read()
            if ret:
                if frame is None:
                    break
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
                
                fgMask = backSub.apply(gray_frame)
                
                cv2.imshow('Frame', frame)
                cv2.imshow('FG Mask', fgMask)
                
                keyboard = cv2.waitKey(10)
                if keyboard == 'q' or keyboard == 27:
                    break
            else:
                break
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        
    ####### Q3: Video Tracking   #########

    # Parameter for video
    def Parameter_video_tracking():
        cap = cv2.VideoCapture(FolderImage +  "featureTracking.mp4")
        _, first_frame = cap.read()
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_frame_gray = cv2.GaussianBlur(first_frame_gray, (5, 5), 0)

        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 80
        params.maxThreshold = 300
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 55
        # Filter by Circularity
        params.filterByCircularity = True
        params.maxCircularity = 1
        params.minCircularity = 0.8
        # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.9
        # params.maxConvexity = 1
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        detector = cv2.SimpleBlobDetector()

        is_v2 = cv2.__version__.startswith("2.")
        if is_v2:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)
    
        keypoints = detector.detect(first_frame)
        List_point_all = []
        
        for i in range(len(keypoints)):
            x,y = np.int(keypoints[i].pt[0]),np.int(keypoints[i].pt[1])
            sz = np.int(keypoints[i].size)
            list_point = [[x,y]]
            List_point_all.append(list_point)
            if sz > 1:
                sz = np.int(sz/2)
            im_with_keypoints = cv2.rectangle(first_frame, (x-2-sz,y-2-sz), (x+10-sz,y+10-sz), color=(0,0,255), thickness=2)
        # cv2.imshow("First Frame", im_with_keypoints)
        # cv2.waitKey(0)
        return im_with_keypoints,List_point_all
        # cap.release()
        # cv2.destroyAllWindows()

    # Q3_1: Preprocessing 1st Frame
    def Pre_video_Tracking(self):
        text =  "Processing 7 blue circles of the 1st frame..."
        print(text)
        self.textEdit_show.setText(text) 

        im_with_keypoints,List_point_all = Process_Image.Parameter_video_tracking()
        cv2.imshow("Pre_tracking", im_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Q3_2: Showing Video Tracking
    def videotracking(self):
        text =  "Processing video tracking..."
        print(text)
        self.textEdit_show.setText(text) 

        cap = cv2.VideoCapture(FolderImage +  "featureTracking.mp4")

        lk_params = dict( winSize  = (21,21),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors

        ret, old_frame = cap.read()
        old_frame,List_point_all = Process_Image.Parameter_video_tracking()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = np.array(List_point_all,np.float32)
    
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while(1):
            ret,frame = cap.read()
            
            if (ret):
                # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                first_frame_blue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                blueLower = (25,50,50)  #130,150,80
                blueUpper = (160,250,250) #250,250,120
                mask1 = cv2.inRange(first_frame_blue, blueLower, blueUpper)
                try:
                    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                except:
                    _,contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                frame_coppy = frame.copy()
                cv2.drawContours(frame_coppy, contours, -1, (255,0,0), 8)
                # cv2.imshow("sds",frame_coppy)
                frame_gray = cv2.cvtColor(frame_coppy, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("89898",old_gray)
                # # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
                # # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]
                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    # print(a,b,c,d)
                    mask = cv2.line(mask, (a,b),(c,d), (0,0,255), 2)
                    frame = cv2.rectangle(frame, (int(a)-6,int(b)-6), (int(a)+7,int(b)+7), color=(0,0,255), thickness=-1)
                
                img = cv2.add(frame,mask)
                
                cv2.imshow('frame',img)
                k = cv2.waitKey(20) & 0xff

                if k == 27:
                    break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            else:
                break

        cv2.destroyAllWindows()
        cap.release()


    # def Processing_Detect():
    #     List_Objpoints = [] # 3d point in real world space
    #     List_Imgpoints = [] # 2d points in image plane.
    #     List_Processimage = []
    #     Chesscol = 11
    #     Chessrow = 8
        
    #     objp = np.zeros((Chessrow*Chesscol,3), np.float32)
    #     objp[:,:2] = np.mgrid[0:Chesscol,0:Chessrow].T.reshape(-1,2)
    #     # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #     if len(ListImage) != 1:
    #         for filename in ListImage:
    #             Image_path = FolderImage + filename
    #             image = cv2.imread(Image_path)
    #             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #             ret, corners = cv2.findChessboardCorners(gray, (Chesscol,Chessrow), None)
    #             if ret == True:
                    
    #                 # corners2 = cv2.cornerSubPix(image,corners,(11,11),(-1,-1),criteria)
    #                 Image_detected = cv2.drawChessboardCorners(gray, (Chesscol,Chessrow), corners, ret)
    #                 List_Objpoints.append(objp)
    #                 List_Imgpoints.append(corners)
    #                 List_Processimage.append(Image_detected)
    #             else:
    #                 print("No Image is detected")
    #         return List_Processimage, List_Objpoints, List_Imgpoints, gray

    def Augmented_Reality(self):
        text =  "Processing Augmented Reality ...\n"
        print(text)
        corners = np.float32([[1, 1, 0], [1, 5, 0], [5, 5, 0], [5, 1, 0],[3, 3, -4]])
        camera_matrix = np.array([[2225.49585482, 0., 1025.5459589 ], [ 0., 2225.18414074, 1038.58518846], [ 0., 0., 1. ]])
        distortion_coefficient = np.array([[-0.12874225, 0.09057782, -0.00099125, 0.00000278, 0.0022925]])
        extrinsix = np.array([[[-0.97157425, -0.01827487, 0.23602862, 6.81253889], [ 0.07148055, -0.97312723, 0.2188925, 3.37330384], [ 0.22568565, 0.22954177, 0.94677165, 16.71572319]], 
                    [[-0.8884799, -0.14530922, -0.435303, 3.3925504 ], [ 0.07148066, -0.98078915, 0.18150248, 4.36149229], [-0.45331444, 0.13014556, 0.88179825, 22.15957429]], 
                    [[-0.52390938, 0.22312793, 0.82202974, 2.68774801], [ 0.00530458, -0.96420621, 0.26510046, 4.70990021], [ 0.85175749, 0.14324914, 0.50397308, 12.98147662]],
                    [[-0.63108673, 0.53013053, 0.566296, 1.22781875], [ 0.13263301, -0.64553994, 0.75212145, 3.48023006], [ 0.76428923, 0.54976341, 0.33707888, 10.9840538 ]], 
                    [[-0.87676843, -0.23020567, 0.42223508, 4.43641198], [ 0.19708207, -0.97286949, -0.12117596, 0.67177428], [ 0.43867502, -0.02302829, 0.89835067, 16.24069227]]])
        
        # List_Processimage, List_Objpoints, List_Imgpoints, image = Processing_Detect()
        # reprojection_error, camera_matrix, distortion_coefficient, rotation_v, translation_v = cv2.calibrateCamera(List_Objpoints, List_Imgpoints, image.shape[:2],None,None)
        text += "-------Intrinsic:------- \n%s\n" %camera_matrix
        text += "-------Distortion:------- \n%s\n" %distortion_coefficient

        print("-------Intrinsic:------- \n%s" %camera_matrix)
        print("-------Distortion:------- \n%s" %distortion_coefficient)
        List_image = []
        # out = cv2.VideoWriter('Augmented_Reality.avi',cv2.VideoWriter_fourcc(*'mp4v'), 1000,(int(image.shape[:2][0]*0.5), int(image.shape[:2][1]*0.5)))
        for index in range(5):
            # index = 0
            Rotation_Mtx= extrinsix[index][0:3,0:3]
            Translation_Mtx = extrinsix[index][0:3,3:]
            # t_Rotation_Mtx,_ = cv2.Rodrigues(rotation_v[index])
            # Translation_Mtx = translation_v[index]
            # Extrinsic_matrix = np.array([[t_Rotation_Mtx[0][0],t_Rotation_Mtx[0][1],t_Rotation_Mtx[0][2],Translation_Mtx[0][0]],
            #                 [t_Rotation_Mtx[1][0],t_Rotation_Mtx[1][1],t_Rotation_Mtx[1][2],Translation_Mtx[1][0]],
            #                 [t_Rotation_Mtx[2][0],t_Rotation_Mtx[2][1],t_Rotation_Mtx[2][2],Translation_Mtx[2][0]],
            #                 ])
            # Image_path = FolderImage + filename
            # image = cv2.imread(Image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            Extrinsic_matrix = extrinsix[index]
            pyramid_corners_2d,_ = cv2.projectPoints(corners,Rotation_Mtx,Translation_Mtx,camera_matrix,distortion_coefficient)
            
            print("-------Extrinsic of %s:------- \n%s" %(ListImage[index],Extrinsic_matrix))
            text += "Extrinsic of %s........ \n%s\n" %(ListImage[index],Extrinsic_matrix)
            # self.text_showvalue.setText(text)  
            ImagePath = FolderImage + ListImage[index]
            img=cv2.imread(ImagePath) #load the correct image
            line_width = 10
            red=(0,0,255) #red (in BGR)
            blue=(255,0,0) #blue (in BGR)
            cv2.line(img, tuple(pyramid_corners_2d[0][0]), tuple(pyramid_corners_2d[1][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[1][0]), tuple(pyramid_corners_2d[2][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[2][0]), tuple(pyramid_corners_2d[3][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[3][0]), tuple(pyramid_corners_2d[0][0]),red,line_width)

            cv2.line(img, tuple(pyramid_corners_2d[0][0]), tuple(pyramid_corners_2d[4][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[1][0]), tuple(pyramid_corners_2d[4][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[2][0]), tuple(pyramid_corners_2d[4][0]),red,line_width)
            cv2.line(img, tuple(pyramid_corners_2d[3][0]), tuple(pyramid_corners_2d[4][0]),red,line_width)
            resize_IMG = cv2.resize(img, (int(img.shape[:2][0]*0.4), int(img.shape[:2][1]*0.4))) 
            List_image.append(resize_IMG)
        self.textEdit_show.setText(text) 
        count = 0
        while(1):
            for i in range(len(List_image)):
                cv2.imshow('Augmented_Reality',List_image[i])
                cv2.waitKey(700)
            if count == 2:
                break
            count += 1
        cv2.waitKey(600)
        cv2.destroyAllWindows()
# Interface
def mainprocessing():
	ui = Process_Image()
	Opencv_hw2.show()
	sys.exit(app.exec_())

# Main
if __name__ == "__main__":         
    mainprocessing()

