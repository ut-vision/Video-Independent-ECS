##########################################################################
# Inference Demo for Eye Contact Segmentation with Simple Face Tracking  #
##########################################################################
import torch
import dlib, cv2, os
import numpy as np
from models.MSTCN.model import MS_TCN2
import gaze_utils
from imutils import face_utils
import argparse

class TrackletManager():
    def __init__(self, IoUThreshod):
        self.container = [] # records a list of tracklets
        self.living_index = [] # records the index of tracklets still alive in the new frame 
        self.IoUThreshod = IoUThreshod
    def matchTracklets(self, frame, bbx, feat):
        # finds the best matching tracklet and register the current bounding box
        # otherwise initialize a new tracklet
        index = -1
        max_score = self.IoUThreshod # This is the minimum tracking threshold
        for i in range(len(self.container)):
            if not self.container[i].active: continue
            current_score = self.container[i].computeIoU(bbx)
            if current_score > max_score:
                max_score = current_score
                index = i
        
        if index == -1: # new tracklet
            self.container.append(Tracklet(frame, bbx, feat))
            self.living_index.append(len(self.container) - 1)
        else: # update the existing tracklet
            self.container[index].registerNewFrame(bbx, feat)
            self.living_index.append(index)
    
    def OnEnteringNextFrame(self):
        # This function has to be called at the end of each frame
        for i in range(len(self.container)):
            if i not in self.living_index:
                self.container[i].active = False
        self.living_index = []

class Tracklet():
    def __init__(self, start_frame, bbx, feat):
        self.active = True # should be turn to false when the tracklet breaks
        self.start_frame = start_frame
        self.bounding_boxes = []
        self.bounding_boxes.append(bbx)
        self.gaze_features = []
        self.gaze_features.append(feat)

    def computeIoU(self, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        boxA = self.bounding_boxes[-1]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def registerNewFrame(self, bbx, feat):
        self.bounding_boxes.append(bbx)
        self.gaze_features.append(feat)

    def inference(self, model):
        x = np.stack(self.gaze_features, axis=0).T
        x = torch.from_numpy(x).to(torch.float32).cuda().view(1, 2048, -1)
        with torch.no_grad():
            output = model(x)
        output = output[-1] 
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu().data.numpy()
        return predicted.reshape(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECS Inference Demo')
    parser.add_argument('--input', type=str, default='',
                            help='input video path')
    parser.add_argument('--output', type=str, default='',
                            help='output video path')
    parser.add_argument('--iou', type=float, default=0.4,
                        help='threshold for face tracking. Default: 0.4')
    args = parser.parse_args()

    print("Loading resources...")
    # Load the pretrained segmentaion model
    model = MS_TCN2(11, 10, 4, 64, 2048, 2)
    model.load_state_dict(torch.load('modules/weights.pth'))
    model.cuda() # comment this line out if you are not using CUDA
    
    # load dlib's face and facial landmark detector, gaze estimator, and face models
    gaze_model, face_detector, lmk_predictor, face_model, facePts, _ = gaze_utils.load_face_models()

    # load the example video
    cap = cv2.VideoCapture(args.input)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    # load dummy camera calibration data
    camera_matrix, camera_distortion = gaze_utils.load_camera_model(video_height, video_width)

    # input to the segmentation model are tracklets of framewsie gaze features extracted from ETH-XGaze
    manager = TrackletManager(args.iou)
    frames = []
    frame_number = 0
    # Since this video only contain only one person, no tracking is needed.
    # Otherwise we need to track human faces.
    # We perform gaze estimation on the input frame
    print("Extracting face tracklets with gaze features...")
    while True:
        ret, image = cap.read() 
        if not ret: break
        frames.append(image)
        # face detection
        detected_faces = face_detector(image, 0) # we assume there is only one person in the image

        for detected_face in detected_faces:
            detected_face = detected_face.rect
            # compute landmarks
            shape = lmk_predictor(image, detected_face) 
            shape = face_utils.shape_to_np(shape)
            landmarks = []
            for (x, y) in shape:
                landmarks.append((x, y))
            
            # Head Pose Estimation for gaze estimation
            landmarks = np.asarray(landmarks)
            landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
            landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
            landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
            hr, ht = gaze_utils.estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
            # gaze face normalization 
            img_normalized, landmarks_normalized, R_inv, face_center_camera_cord = gaze_utils.normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)
            # gaze estimation
            input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
            input_var = gaze_utils.trans(input_var)
            input_var = torch.autograd.Variable(input_var.float().cuda())
            input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
            pred_gaze, feature = gaze_model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
            pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
            pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
            feature = feature[0].cpu().data.numpy()
            
            # now we get a face with its gaze feature, we need to register it
            manager.matchTracklets(frame_number, [detected_face.left(), detected_face.top(), detected_face.right(), detected_face.bottom()], feature)

        manager.OnEnteringNextFrame()
        frame_number += 1


    print("Detected", len(manager.container), "tracklets")
    print("Performing ECS for each tracklet")
    # For each tracklet with minimum length l00, perform ECS and draw bounding boxes colored by predictions
    #for tracklet in manager.container:
    for idx in range(len(manager.container)):
        tracklet = manager.container[idx]
        start_frame = tracklet.start_frame
        bounding_boxes = tracklet.bounding_boxes
        if len(bounding_boxes) < 100: continue
        prediction = tracklet.inference(model)

        for i in range(len(bounding_boxes)):
            image = frames[start_frame + i]
            face_box = bounding_boxes[i]
            if prediction[i] == 1:
                color = (0, 255, 0) # Green
            else:
                color = (0, 0, 255) # Red
            cv2.rectangle(image, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, 5)
            cv2.putText(image, "No."+str(int(idx)), (face_box[0], face_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3, cv2.LINE_AA)



    print("Writing as a video")
    # create output video
    outPath = args.output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outPath, fourcc, 25.0, (int(video_width),int(video_height)))
    for image in frames:
        out.write(image)

    print("Done")

