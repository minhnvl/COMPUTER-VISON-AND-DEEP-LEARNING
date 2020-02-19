import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import os
FolderImage = "./Image/"
ListImage = [f for f in os.listdir(FolderImage) if f.endswith(".bmp")]

 
################################### OPEN CVDL ##############################################
def Disparity_OpenCV():
    imgL = cv2.imread( FolderImage + 'imL.png',0)
    imgR = cv2.imread( FolderImage + 'imR.png',0)

    window_size = 9
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
    
def Background_subtraction():
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

            # fgMask=cv2.erode(fgMask,kernel,iterations=1)  
            # counter=np.sum(fgmask==255)  
                    
            
            # cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            # cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            
            
            cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask', fgMask)
            
            keyboard = cv2.waitKey(10)
            if keyboard == 'q' or keyboard == 27:
                break
        else:
            input()

def Pre_video_tracking():
    cap = cv2.VideoCapture(FolderImage +  "featureTracking.mp4")
    _, first_frame = cap.read()
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_frame_gray = cv2.GaussianBlur(first_frame_gray, (5, 5), 0)
    # first_frame_blue = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    # blueLower = (60,0,0)  #130,150,80
    # blueUpper = (150,255,255) #250,250,120
    # mask = cv2.inRange(first_frame_blue, blueLower, blueUpper)
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # first_frame_coppy = first_frame.copy()

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
    cv2.imshow("First Frame", im_with_keypoints)
    cv2.waitKey(1000)
    return im_with_keypoints,List_point_all
    # cap.release()
    # cv2.destroyAllWindows()



def videotracking():
    cap = cv2.VideoCapture(FolderImage +  "featureTracking.mp4")

    lk_params = dict( winSize  = (21,21),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors

    ret, old_frame = cap.read()
    old_frame,List_point_all = Pre_video_tracking()
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
            input()

    cv2.destroyAllWindows()
    cap.release()


def Processing_Detect():
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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (Chesscol,Chessrow), None)
            if ret == True:
                
                # corners2 = cv2.cornerSubPix(image,corners,(11,11),(-1,-1),criteria)
                Image_detected = cv2.drawChessboardCorners(gray, (Chesscol,Chessrow), corners, ret)
                List_Objpoints.append(objp)
                List_Imgpoints.append(corners)
                List_Processimage.append(Image_detected)
            else:
                print("No Image is detected")
        return List_Processimage, List_Objpoints, List_Imgpoints, gray

def Augmented_Reality():
    corners = np.float32([[1, 1, 0], [1, 5, 0], [5, 5, 0], [5, 1, 0],[3, 3, -4]])
    List_Processimage, List_Objpoints, List_Imgpoints, image = Processing_Detect()
    reprojection_error, camera_matrix, distortion_coefficient, rotation_v, translation_v = cv2.calibrateCamera(List_Objpoints, List_Imgpoints, image.shape[:2],None,None)
    text = ""
    print("-------Intrinsic:------- \n%s" %camera_matrix)
    print("-------Distortion:------- \n%s" %distortion_coefficient)
    List_image = []
    out = cv2.VideoWriter('Augmented_Reality.avi',cv2.VideoWriter_fourcc(*'mp4v'), 1000,(int(image.shape[:2][0]*0.5), int(image.shape[:2][1]*0.5)))
    for index in range(5):
        # index = 0
        Rotation_Mtx= rotation_v[index]
        t_Rotation_Mtx,_ = cv2.Rodrigues(rotation_v[index])
        Translation_Mtx = translation_v[index]
        Extrinsic_matrix = np.array([[t_Rotation_Mtx[0][0],t_Rotation_Mtx[0][1],t_Rotation_Mtx[0][2],Translation_Mtx[0][0]],
                        [t_Rotation_Mtx[1][0],t_Rotation_Mtx[1][1],t_Rotation_Mtx[1][2],Translation_Mtx[1][0]],
                        [t_Rotation_Mtx[2][0],t_Rotation_Mtx[2][1],t_Rotation_Mtx[2][2],Translation_Mtx[2][0]],
                        ])

        pyramid_corners_2d,_ = cv2.projectPoints(corners,Rotation_Mtx,Translation_Mtx,camera_matrix,distortion_coefficient)
        
        print("-------Extrinsic of %s:------- \n%s" %(ListImage[index],Extrinsic_matrix))
        text += "Extrinsic of %s........ \n" %ListImage[index]
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
        resize_IMG = cv2.resize(img, (int(image.shape[:2][0]*0.4), int(image.shape[:2][1]*0.4))) 
        List_image.append(resize_IMG)
        cv2.imshow('Augmented_Reality',resize_IMG)
        cv2.waitKey(700)



def Random():
    import numpy as np
    a = "kaggleluna_full.npy"
    idcs = np.load(a)
    print(idcs)

if __name__ == "__main__":
    #################### OPEN CVDL #####################
    # Disparity_OpenCV()
    # Background_subtraction()
    # Pre_video_tracking() #Remenber open imshow()
    # videotracking()
    # Augmented_Reality()
    Random()


