# Gaze_lib.py
# this script contains a couple helper functions that is needed for gaze estimation, pose estimation, and other related tasks.

import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from models.XGazeModel import gaze_network
import matplotlib.pyplot as plt
import math
from skspatial.objects import Sphere, Line, Cylinder
import scipy as sp

# for gaze estimation
trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
    return rvec, tvec

def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix
    R_inv = np.linalg.inv(R)

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped, R_inv, face_center


def denormalize_predicted_gaze(gaze_yaw_pitch, R_inv):
    pred_gaze_cancel_nor = pitchyaw_to_vector(gaze_yaw_pitch.reshape(1,2)).reshape(3,1) # get 3d gaze direction as a vector
    pred_gaze_cancel_nor = np.matmul(R_inv, pred_gaze_cancel_nor.reshape(3,1)) # apply inverse transformation to convert it back to camera coord system
    pred_gaze_cancel_nor = pred_gaze_cancel_nor / np.linalg.norm(pred_gaze_cancel_nor) # vector normalization
    pred_yaw_pitch_cancel_nor = vector_to_pitchyaw(pred_gaze_cancel_nor.reshape(1,3)) # convert to yaw and pitch
    return pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor

def map_to_camera_plane(gaze3d, face_center):
    n = (-face_center[2]) / gaze3d[2]
    x2d = face_center[0] + n * gaze3d[0]
    y2d = face_center[1] + n * gaze3d[1]
    return x2d, y2d

def virtual_intersection(gaze3d, face_center, radius=1000, cylinderLength=100000, method="cylinder"):
    if method == "cylinder":
        cylinder = Cylinder(point=[0, -cylinderLength, 0], vector=[0, cylinderLength, 0], radius=radius)
        line = Line(point=[face_center[0][0], face_center[1][0], face_center[2][0]], direction=[gaze3d[0][0], gaze3d[1][0], gaze3d[2][0]])
        try:
            point_a, point_b = cylinder.intersect_line(line)
            result = [point_a, point_b]
            result = sorted(result, key=lambda x: x[2])
            if gaze3d[2][0] < 0:
                return result[0]
            else:
                return result[1]
        except:
            return [0,0,0] # this happens when the estimated gaze center is outside the cylinder. In such cases the head pose estimation might fail.

def load_camera_model(video_height, video_width):
    # adjust video height to allow for other minor video resolutions 
    if abs(video_height - 480) < 20: video_height = 480
    if abs(video_height - 720) < 20: video_height = 720
    if abs(video_height - 1080) < 20: video_height = 1080
    if video_height not in [480, 720, 1080]:
        print("Warning: This video has an unexpected resolution. Height =", video_height, "Width =", video_width, "Using forced guess")
        cam_file_name = 'camera/cam720.xml'
        fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
        camera_distortion = fs.getNode('Distortion_Coefficients').mat()
        camera_matrix[0][0] = video_width // 2
        camera_matrix[1][1] = video_width // 2
        camera_matrix[0][2] = video_width // 2
        camera_matrix[1][2] = video_height // 2
        return camera_matrix, camera_distortion
    # load camera information
    if video_height == 1080:
        print("Video is 1080p")
        cam_file_name = 'camera/cam1080.xml'

    if video_height == 720:
        print("Video is 720p")
        cam_file_name = 'camera/cam720.xml'

    if video_height == 480:
        print("Video is 480p")
        cam_file_name = 'camera/cam480.xml'
    
    if not os.path.isfile(cam_file_name):
        print('no camera calibration file is found.')
        exit(0)
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('Camera_Matrix').mat() 
    camera_distortion = fs.getNode('Distortion_Coefficients').mat()
    return camera_matrix, camera_distortion

def load_face_models():
    # load face model
    face_model = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model[landmark_use, :]
    facePts = face_model.reshape(6, 1, 3)
    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks_GTX.dat') # this version works better with cnn detector
    face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')

    print('load gaze estimator')
    model = gaze_network()
    model.cuda() 
    pre_trained_model_path = './modules/epoch_24_ckpt.pth.tar'
    if not os.path.isfile(pre_trained_model_path):
        print('the pre-trained gaze estimation model does not exist.')
        exit(0)
    else:
        print('load the pre-trained model: ', pre_trained_model_path)
    ckpt = torch.load(pre_trained_model_path)
    model.load_state_dict(ckpt['model_state'], strict=True)  
    model.eval()  

    pose3d = np.loadtxt('./modules/body_model.txt').reshape(6,1,3)

    return model, face_detector, predictor, face_model, facePts, pose3d

def computeInverseTransformation(hr, ht):
    hR = cv2.Rodrigues(hr)[0]
    hR = np.concatenate((hR, ht), axis=1)
    hR = np.concatenate((hR, np.array([[0,0,0,1]])), axis=0)
    return np.linalg.inv(hR) # returns 4x4 transformation matrix

# input format is [left, top, right, bottom]
def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# returns a list of IOU scores.
def compare_bbs(known_bbs, bb_to_check):
    res = []
    for t in known_bbs:
        res.append(IoU(t, bb_to_check))
    return res 

def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

