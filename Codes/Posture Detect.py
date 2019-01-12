# encoding=UTF8
import numpy as np
import cv2
import threading
# from multiprocessing import Process, Pipe
# import RPi.GPIO as GPIO
import os,sys
import time
import dlib
import math

'''cv2.namedWindow("L1")
cv2.namedWindow("R1")
cv2.namedWindow("L")
cv2.namedWindow("R")
cv2.namedWindow("4")'''


##classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
lock = threading.RLock()
'''GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(13,GPIO.IN,pull_up_down=GPIO.PUD_DOWN) # 关机
GPIO.setup(15,GPIO.IN,pull_up_down=GPIO.PUD_UP) # 标准位置设定
GPIO.setup(3,GPIO.OUT) # 程序工作状态
GPIO.setup(5,GPIO.OUT) # 成功识别到人脸
GPIO.setup(7,GPIO.OUT) # 是否在规定区域内
GPIO.setup(11,GPIO.OUT) # 标准位置确认'''
camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(0)
size = (360, 480) # 图像尺寸*
'''camera1.set(3, size[0])
camera1.set(4, size[1])
camera2.set(3, size[0])
camera2.set(4, size[1])'''
for  i in range(40):  # 消除摄像头刚打开时的过曝
    camera1.read()
    camera2.read()
detector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


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


def anglecal(rects, img, camera_matrix, dist_coeffs):
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
    try:
        image_points = np.array([
            (landmarks[30, 0], landmarks[30, 1]),  # Nose tip
            (landmarks[8, 0], landmarks[8, 1]),  # Chin
            (landmarks[36, 0], landmarks[36, 1]),  # Left eye left corner
            (landmarks[45, 0], landmarks[45, 1]),  # Right eye right corne
            (landmarks[48, 0], landmarks[48, 1]),  # Left Mouth corner
            (landmarks[54, 0], landmarks[54, 1])  # Right mouth corner
        ], dtype="double")
    except:
        angle = float("inf")
        return angle
    # else:
    # print('Detect Face successfully!')
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    # Camera internals
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    for p in image_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (255,255,255), -1)

    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
    x = (x * 360) / (2 * 3.14) -180
    if x<= -180:
        x = x + 360
    y = (y * 360) / (2 * 3.14)
    z = (z * 360) / (2 * 3.14)
    angle = (x,y,z)
    print(angle)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return angle, p1, p2


while True:
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()
    if not ret1 or not ret2:
        print('camera error')
        exit()
    imageL = rotate_bound(frame1, 90)
    imageR = rotate_bound(frame2, 90)
    frameL = cv2.resize(imageL, size, interpolation=cv2.INTER_AREA)  # resize resolution
    frameR = cv2.resize(imageR, size, interpolation=cv2.INTER_AREA)  # resize resolution
    frameL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
    frameR = cv2.remap(frameR, left_map1, left_map2, cv2.INTER_LINEAR)
    '''clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frameL = clahe.apply(frameL)
    frameR = clahe.apply(frameR)'''
    imgL_ = cv2.resize(frameL, (0, 0), fx=0.5, fy=0.5)
    size1 = imgL_.shape
    imgR_ = cv2.resize(frameR, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("left", imgL_)
    cv2.imshow("right", imgL_)
    cv2.waitKey(100)
    rectsL = detector(imgL_, 0)
    rectsR = detector(imgR_, 0)
    focal_length = size1[1]
    center = (size1[1]/2, size1[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    
    # print("Camera Matrix :\n {0}".format(camera_matrix))
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    if len(rectsL) and len(rectsR):
        rotation_L, p1, p2 = anglecal(rectsL, imgL_, camera_matrix, dist_coeffs)
        rotation_R, _, _ = anglecal(rectsR, imgR_, camera_matrix, dist_coeffs)
        if rotation_R != float("inf") and rotation_L != float("inf"):
            # angle = rotation_L
            angle = ((rotation_L[0] + rotation_R[0]) / 2, (rotation_L[1] + rotation_R[1]) / 2, (rotation_L[2] + rotation_R[2]) / 2)
            x = angle[0]
            y = angle[1]
            z = angle[2]
            '''font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(imgL_, str(x), (10, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(imgL_, str(y), (10, 75), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(imgL_, str(z), (10, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(im, str(yaw), y, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(im, str(roll), z, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)'''
            cv2.line(imgL_, p1, p2, (255, 0, 0), 2)
            cv2.imshow("angle", imgL_)
            cv2.waitKey(1)
