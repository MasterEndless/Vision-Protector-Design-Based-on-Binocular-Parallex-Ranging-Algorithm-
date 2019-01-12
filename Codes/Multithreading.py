# encoding=UTF8
# import the necessary packages
import cv2
import numpy as np
import dlib
import math
import threading
from time import clock
import time
cv2.namedWindow("4")
size = (360, 480) # 图像尺寸*
times = 2
detector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
flag1 = 1
flag2 = 1
start = 0
stop = 0
en_detect_out = 0
pos=float("inf")
angle=float("inf")
lock = threading.RLock()


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

'''class WebcamVideoStream:
    def __init__(self):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream1 = cv2.VideoCapture(1)
        self.stream2 = cv2.VideoCapture(0)
        (self.grabbed1, self.frame1) = self.stream1.read()
	(self.grabbed2, self.frame2) = self.stream2.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed1, self.frame1) = self.stream1.read()
            (self.grabbed2, self.frame2) = self.stream2.read()
    def read(self):
        # return the frame most recently read
        return self.frame1, self.frame2

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True'''


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


def thread_detector():
    global flag1, flag2, en_detect_out, imgL_, frameL, frameR, rectsL, rectsR, frame1, frame2
    while True:
        en_detect_out = 0
        # grab the dimensions of the image and then determine the
        # center
	lock.acquire()
        imageL = frame1
	imageR = frame2
	lock.release()
        imageL = rotate_bound(imageL, 90)
	imageR = rotate_bound(imageR, 90)
	frameL = cv2.resize(imageL, size, interpolation=cv2.INTER_AREA)  # resize resolution
        frameR = cv2.resize(imageR, size, interpolation=cv2.INTER_AREA)  # resize resolution
        frameL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
        frameR = cv2.remap(frameR, left_map1, left_map2, cv2.INTER_LINEAR)
        frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        '''clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frameL = clahe.apply(frameL)
        frameR = clahe.apply(frameR)'''
        frameL = cv2.equalizeHist(frameL)
	frameR = cv2.equalizeHist(frameR)
        imgL_ = cv2.resize(frameL, (0, 0), fx=0.5, fy=0.5)
        imgR_ = cv2.resize(frameR, (0, 0), fx=0.5, fy=0.5)
	'''cv2.imshow("oout", imgL_)
	cv2.waitKey(1)'''
        rectsL = detector(imgL_, 0)
        rectsR = detector(imgR_, 0)
        en_detect_out = 1
        while flag2 or flag1:
            None


def landmark_angle():
    global rectsL,frameL,flag1, en_detect_out, angle
    while True:
        flag1=1
        while True:
            if en_detect_out==1:
                rects = rectsL
                im = frameL
                break
        flag1=0
        # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        count = 0
	landmarks = None        
	for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
        try:
            image_points = np.array([
                (times*landmarks[30, 0], times*landmarks[30, 1]),  # Nose tip
                (times*landmarks[8, 0], times*landmarks[8, 1]),  # Chin
                (times*landmarks[36, 0], times*landmarks[36, 1]),  # Left eye left corner
                (times*landmarks[45, 0], times*landmarks[45, 1]),  # Right eye right corne
                (times*landmarks[48, 0], times*landmarks[48, 1]),  # Left Mouth corner
                (times*landmarks[54, 0], times*landmarks[54, 1])  # Right mouth corner
            ], dtype="double")
        except:
            print('angle error')
            angle = float("inf")
	    continue
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

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)

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
        x = (x * 360) / (2 * 3.14) - 180
        if x <= -180:
            x = x + 360
        y = (y * 360) / (2 * 3.14)
        z = (z * 360) / (2 * 3.14)
        # print(x, y, z)
	angle =(x,y,z)

        '''font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, str(x), (10, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(im, str(y), (10, 75), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(im, str(z), (10, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(im, str(yaw), y, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(im, str(roll), z, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)'''
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(im, p1, p2, (255, 0, 0), 2)

        # Display image
        cv2.imshow("Output", im)
        cv2.waitKey(15)


def BM():
    global flag2, frameL, frameR, rectsL, rectsR, pos, en_detect_out
    while True:
        flag2=1
        while True:
            if en_detect_out==1:
                imgL = frameL
                imgR = frameR
                faces1 = rectsL
                faces2 = rectsR
                break
        flag2=0
        if len(faces1) and len(faces2):
            sss1 = np.zeros([size[1], size[0]], dtype=np.uint8)
            sss2 = np.zeros([size[1], size[0]], dtype=np.uint8)
            for i, d in enumerate(faces1):               
		x = d.left()*times
                y = d.top()*times
                w = (d.right() - d.left())*times
                h = (d.bottom() - d.top())*times
                # cv2.rectangle(imgL,(x,y),(x+w,y+w),(0,255,0),2)
                sss1[y+int(0.25*w):y+int(0.75*w), x+int(0.25*w):x+int(0.75*w)] = 255
		# sss1[y:y+h, x:x+w] = 255
	    for i, d in enumerate(faces2):               
		x = d.left()*times
                y = d.top()*times
                w = (d.right() - d.left())*times
                h = (d.bottom() - d.top())*times
                # cv2.rectangle(imgR,(x,y),(x+w,y+w),(0,255,0),2)
                sss2[y+int(0.25*w):y+int(0.75*w), x+int(0.25*w):x+int(0.75*w)] = 255
		# sss2[y:y+h, x:x+w] = 255
            imgL = cv2.add(imgL, np.zeros(np.shape(imgL), dtype=np.uint8), mask=sss1)
            imgR = cv2.add(imgR, np.zeros(np.shape(imgR), dtype=np.uint8), mask=sss2)
            # cv2.imshow("L", imgL)
            # cv2.imshow("R", imgR)
            # cv2.waitKey(30)
            blockSize = 42
            # 根据Block Matching方法生成差异图
            stereo = cv2.StereoBM_create(numDisparities=16 * 3, blockSize=blockSize )
            disparity2 = stereo.compute(imgL, imgR)
            disp2 = cv2.normalize(disparity2, disparity2, alpha=1, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow("4", disp2)
            cv2.waitKey(15)
            threeD2 = cv2.reprojectImageTo3D(disparity2.astype(np.float32) / 16., Q)
            # print "stop cal"
            cont = 0
            sum = 0
	    for i, d in enumerate(faces1):               
		x = d.left()*times
                y = d.top()*times
                w = (d.right() - d.left())*times
                h = (d.bottom() - d.top())*times
                cv2.rectangle(imgL,(x,y),(x+w,y+w),(0,255,0),2)
                for x1 in range(x + int(0.25 * w), x + int(0.75 * w)):
                    for y1 in range(y + int(0.25 * h), y + int(0.75 * h)):
                        if threeD2[y1][x1][2] > 0 and threeD2[y1][x1][2] != float("inf"):
                            sum += threeD2[y1][x1][2]
                            cont += 1
                if cont != 0:
                    ave = (sum / cont) - 100
                    print(ave)
                    pos = [(x + x + w) / 2, (y + y + h) / 2, ave]
                else:
                    pos = float("inf")
		    print("depth error")


def LorR():
    global camera1, camera2
    process = 1
    while True:
        ret1, img1 = camera1.read()
        ret2, img2 = camera2.read()
        if not ret1 or not ret2:
            print 'camera error'
            exit()
        frame1 = rotate_bound(img1, 90)
        frame2 = rotate_bound(img2, 90)
        if process:
            frame1 = cv2.resize(frame1, size, interpolation=cv2.INTER_AREA)  # resize resolution
            frame2 = cv2.resize(frame2, size, interpolation=cv2.INTER_AREA)
            frame1= cv2.resize(frame1, (0,0),fx=0.5,fy=0.5)
            frame2= cv2.resize(frame2, (0,0),fx=0.5,fy=0.5)
            faces1 = detector(frame1, 0)
            faces2 = detector(frame2, 0)
	    print(faces1)
	    print(faces2)
	    if len(faces1) and len(faces2):
		for i, d in enumerate(faces1):               
		    x1 = d.left()
	        for i, d in enumerate(faces2):
	    	    x2 = d.left()
	        if x1 < x2:
                    camera1.release()
                    camera2.release()
                    camera1 = cv2.VideoCapture(0)
                    camera2 = cv2.VideoCapture(1)
                    time.sleep(3)
                    for  i in range(30):  # 消除摄像头刚打开时的过曝
                        camera1.read()
                        camera2.read()
                    break
                else:
                    break
            else:
                continue
        process = not process




def initial_position():
    global ini_angle, pos_ini, pos, angle, ini, com
    if flag2 and flag1 and pos != float("inf") and angle != float("inf"):
        pos_ini = pos
        ini_angle = angle
	print("success")
	print(ini_angle)
	ini = 0
	com = 1
    elif pos != float("inf") or angle != float("inf"):
        print("no face!")
    else:
        print("depth error")


def compare():
    global ini_angle, pos_ini, pos, angle, start, stop
    if flag2 and flag1 and pos != float("inf") and angle != float("inf"):
        dif_ang0 = abs(ini_angle[0] - angle[0])
        dif_ang1 = abs(ini_angle[1] - angle[1])
        dif_ang2 = abs(ini_angle[2] - angle[2])
        sqr = np.square(pos_ini[0] - pos[0]) + np.square(pos_ini[1] - pos[1]) + np.square(
        pos_ini[2] - pos[2])
        distance = np.sqrt(sqr)
        if dif_ang0 > 6 or dif_ang1 > 25 or dif_ang2 > 20 or distance > 50:
            start = clock()
        elif dif_ang0 < 10 and dif_ang1 < 35 and dif_ang2 < 20 and distance < 50:
            stop = clock()
        if (start-stop) > 5:
            print("move too far away!")
        else:
            print("good position")



flag = 0
LR = 1
ini = 0
com = 0
p2 = threading.Thread(target=thread_detector)
p3 = threading.Thread(target=landmark_angle)
p4 = threading.Thread(target=BM)
p2.setDaemon(True)
p3.setDaemon(True)
p4.setDaemon(True)
while True:
    global frame1, frame2
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()
    if not ret1 or not ret2:
        print 'camera error'
        exit()
    if LR:
	LorR()
	LR = 0
	flag = 1 
    if flag:
	p2.start()
	p3.start()
	p4.start()
	flag = 0
	ini = 1
    if ini:
	initial_position()
    if com:
	compare()
    '''if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()'''




# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

