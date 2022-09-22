##################################################
# Demo for Eye Contact Segmentation Inference   ##
##################################################
import torch
import dlib, cv2, os
import numpy as np
from models.MSTCN.model import MS_TCN2
import gaze_utils
from imutils import face_utils

if __name__ == "__main__":

    print("Loading resources...")
    # Load the pretrained segmentaion model
    model = MS_TCN2(11, 10, 4, 64, 2048, 2)
    model.load_state_dict(torch.load('modules/weights.pth'))
    model.cuda() # comment this line out if you are not using CUDA
    
    # load dlib's face and facial landmark detector, gaze estimator, and face models
    gaze_model, face_detector, lmk_predictor, face_model, facePts, _ = gaze_utils.load_face_models()

    # load the example video
    cap = cv2.VideoCapture('example/example.avi')
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    # load dummy camera calibration data
    camera_matrix, camera_distortion = gaze_utils.load_camera_model(video_height, video_width)

    # input to the segmentation model are framewsie gaze features extracted from ETH-XGaze
    gaze_features = []
    frames = []
    face_boxes = []

    # Since this video only contain only one person, no tracking is needed.
    # Otherwise we need to track human faces.
    # We perform gaze estimation on the input frame
    print("Extracting Gaze Features...")
    while True:
        ret, image = cap.read() 
        if not ret: break
        frames.append(image)
        # face detection
        detected_face = face_detector(image, 0)[0].rect # we assume there is only one person in the image
        face_boxes.append([int(detected_face.left()), int(detected_face.top()), int(detected_face.right()), int(detected_face.bottom())])
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
        gaze_features.append(feature)
    
    # Eye Contact Segmentation
    x = np.stack(gaze_features, axis=0).T
    x = torch.from_numpy(x).to(torch.float32).cuda().view(1, 2048, -1)
    with torch.no_grad():
        output = model(x)
    output = output[-1] 
    _, predicted = torch.max(output.data, 1)
    predicted = predicted.cpu().data.numpy()
    predicted = predicted.reshape(-1)
    print("Predicted:")
    print(predicted)

    print("writing as a video")
    # create output video
    outPath = os.path.join("example", "result.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outPath, fourcc, 25.0, (int(video_width),int(video_height)))
    for i in range(len(frames)):
        image = frames[i]
        face_box = face_boxes[i]
        if predicted[i] == 1:
            color = (0, 255, 0) # Green
        else:
            color = (0, 0, 255) # Red
        cv2.rectangle(image, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, 5)
        out.write(image)
    print("Done")
