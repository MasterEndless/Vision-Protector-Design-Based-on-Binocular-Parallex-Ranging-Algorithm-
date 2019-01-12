# encoding=UTF8
# import the necessary packages
import cv2
import numpy as np
import dlib
cv2.namedWindow("4")
cv2.createTrackbar("num", "4", 0, 10, lambda x: None)
cv2.createTrackbar("blockSize", "4", 5, 255, lambda x: None)
size = (360, 480) # 图像尺寸*
detector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
times = 2

left_camera_matrix = np.array([[643.15011, 0., 245.27399],
                               [0., 640.66412, 305.51468],
                               [0., 0., 1.]])
left_distortion = np.array([[0.30889,-1.29271,-0.00429,-0.00080,0.00000]])

right_camera_matrix = np.array([[642.66378, 0., 240.25534],
                                [0., 640.34419, 328.10160],
                                [0., 0., 1.]])
right_distortion = np.array([[0.33489,-1.44933,-0.00177,-0.00171,0.00000]])

om = np.array([0.00846,-0.01536,0.00121]) # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([-41.31244,0.25378,0.72602]) # 平移关系向量


# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
  right_camera_matrix, right_distortion, size, R,
  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)


camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(0)


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


while True:
    ret1, imageL = camera1.read()
    ret2, imageR = camera2.read()
    if not ret1 or not ret2:
        print 'camera error'
        exit()
    frameL = rotate_bound(imageL, 90)
    frameR = rotate_bound(imageR, 90)
    frameL = cv2.resize(frameL, size, interpolation=cv2.INTER_AREA)  # resize resolution
    frameR = cv2.resize(frameR, size, interpolation=cv2.INTER_AREA)
    img1_rectified = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)
    # 将图片置为灰度图，为StereoSGBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    imgL = cv2.equalizeHist(imgL)
    imgR = cv2.equalizeHist(imgR)
    sss1 = np.zeros([size[1], size[0]], dtype=np.uint8)
    sss2 = np.zeros([size[1], size[0]], dtype=np.uint8)
    imgL_=cv2.resize(imgL, (0,0),fx=0.5,fy=0.5)
    imgR_=cv2.resize(imgR, (0,0),fx=0.5,fy=0.5)
    faces1 = detector(imgL_, 0)
    faces2 = detector(imgR_, 0)
    # print faces1
    if len(faces1) and len(faces2):
        for i, d in enumerate(faces1):
            x = d.left() * times
            y = d.top() * times
            w = (d.right() - d.left()) * times
            h = (d.bottom() - d.top()) * times
            # cv2.rectangle(imgL,(x,y),(x+w,y+w),(0,255,0),2)
            sss1[y+int(0.25*w):y+int(0.75*w), x+int(0.25*w):x+int(0.75*w)] = 255
        for i, d in enumerate(faces2):
            x = d.left() * times
            y = d.top() * times
            w = (d.right() - d.left()) * times
            h = (d.bottom() - d.top()) * times
            # cv2.rectangle(imgR,(x,y),(x+w,y+w),(0,255,0),2)
            sss2[y+int(0.25*w):y+int(0.75*w), x+int(0.25*w):x+int(0.75*w)] = 255
        imgL1 = cv2.add(imgL, np.zeros(np.shape(imgL), dtype=np.uint8), mask=sss1)
        imgR1 = cv2.add(imgR, np.zeros(np.shape(imgR), dtype=np.uint8), mask=sss2)
        num = cv2.getTrackbarPos("num", "4")
        blockSize = cv2.getTrackbarPos("blockSize", "4")
        # blockSize = 55
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 5:
            blockSize = 5
        stereo = cv2.StereoBM_create(numDisparities=16 * 3, blockSize=41 )
        disparity2 = stereo.compute(imgL1, imgR1)
        disp2 = cv2.normalize(disparity2, disparity2, alpha=1, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("disparity", disp2)
        cv2.imshow("original", imgL)
        cv2.waitKey(30)
        threeD2 = cv2.reprojectImageTo3D(disparity2.astype(np.float32) / 16., Q)
        #print "stop cal"
        cont = 0
        sum = 0
        # GPIO.output(5,1)
        for i, d in enumerate(faces1):
            x = d.left() * times
            y = d.top() * times
            w = (d.right() - d.left()) * times
            h = (d.bottom() - d.top()) * times
            for x1 in range(x + int(0.25 * w), x + int(0.75 * w)):
                for y1 in range(y + int(0.25 * h), y + int(0.75 * h)):
                    if threeD2[y1][x1][2] > 0 and threeD2[y1][x1][2] != float("inf"):
                        sum += threeD2[y1][x1][2]
                        cont += 1
        if cont != 0:
            ave = (sum / cont) -200
            print ave
        else:
            print 'cont = 0'

    else:
        print "no face!!!!!!!"


