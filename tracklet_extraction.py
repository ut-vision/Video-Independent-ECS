# This script aims at extracting training/test samples from voxceleb2 dataset using an automatic pipeline.
import glob
import os, random, sys
import cv2, dlib, torch
import pandas as pd
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path
from pathlib import PurePath
import gaze_utils
import tools

def run(args):
    # This class represents a tracklet of the faces. 
    # This is used to keep record of the targeted celebrity
    class tracklet():
        def __init__(self, start, tracklet_num=0):
            self.tracklet_num = tracklet_num
            self.start_frame = start
            self.bounding_box = [] 
            self.gaze_feature = []
            self.landmark = []

        def update(self, bbx, landmark, gaze_feature):
            self.bounding_box.append(bbx)
            self.gaze_feature.append(gaze_feature)
            self.landmark.append(landmark)

        def canForm(self):
            if len(self.bounding_box) > args.segment_length: return True
            else: return False

        def getTrackletLength(self):
            return len(self.bounding_box)
        
        def canTrack(self, bbx):
            if len(self.bounding_box) == 0: return True
            else:
                iouScore = gaze_utils.IoU(bbx, self.bounding_box[-1])
                return iouScore > args.thres_tracking

        def saveTracklet(self, path, video_width, video_height):
            file_name = str(self.tracklet_num)
            while len(file_name) < 3:
                file_name = "0" + file_name
            file_name = file_name + ".npz"
            file_name = os.path.join(path, file_name)
            print("Saving tracklet. Frame:", self.start_frame, "Length:", self.getTrackletLength(), "to", file_name)
            bbx = self.bounding_box
            for i in range(len(bbx)):
                bbx[i][0] /= video_width
                bbx[i][1] /= video_height
                bbx[i][2] /= video_width
                bbx[i][3] /= video_height
            bbx = np.array(bbx)
            np.savez_compressed(file_name, start_frame=self.start_frame, bbx=bbx, landmark=np.stack(self.landmark, axis=0), gaze_feature=np.stack(self.gaze_feature, axis=0))
            self.tracklet_num += 1

        def getNextTrackletNum(self):
            return self.tracklet_num

        def saveAndKill(self, savePath, msg, video_width, video_height):
            """
            Prints the msg(str), then checks if the current tracklet is long enough to be saved. To be called when this tracklet is broken
            """
            print(msg)
            # No face detected at this stage, we should save the current tracklet, and make the next tracklet
            if self.canForm():
                self.saveTracklet(savePath, video_width, video_height)
                print("registered frame", self.start_frame, "-", self.start_frame + self.getTrackletLength())
            else: 
                print("frame", self.start_frame, "-", self.start_frame + self.getTrackletLength(), "too short. Discard")

    # Load face and gaze models
    gaze_model, face_detector, lmk_predictor, face_model, facePts, pose3d = gaze_utils.load_face_models()
    # getting glob information
    print("getting the list of celebrity ids...")
    directories = glob.glob(os.path.join(args.path, "*/"))
    print("...Done!")

    for id_dir in directories:
        id_name = Path(id_dir).parts[-1]
        print("processing celebrity id", id_name) 

        mkv_list = glob.glob(os.path.join(id_dir, '*_fps25.mkv'))
        if len(mkv_list) == 0: continue
        
        # loading the precomputed embeddings
        if not os.path.isfile(os.path.join(id_dir, id_name + "_face_embedding.npz")): continue
        celebEmbeddings = np.load(os.path.join(id_dir, id_name + "_face_embedding.npz"))['arr_0']

        for mkv in mkv_list:
            mkvName = os.path.split(mkv)[1].split(".")[0]
            print("---Processing video", mkv)
            savePath = list(PurePath(id_dir).parts)
            savePath = os.path.join(*savePath)
            savePath = os.path.join(savePath, "tracklets", mkvName)
            print("Tracklets will be saving to", savePath)
            Path(savePath).mkdir(parents=True, exist_ok=True)

            if os.path.isfile(os.path.join(savePath, "fixations.csv")):
                print("This video has been processed. Next.")
                continue

            # now load the video
            cap = cv2.VideoCapture(mkv)
            video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
            video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("This video has length", video_length)
            if (video_length // 25) < 120: 
                print("This video is too short. Next")
                continue
            if (video_length // 25) > 720: 
                print("This video is too long. Next")
                continue
            camera_matrix, camera_distortion = gaze_utils.load_camera_model(video_height, video_width)

            face_tracklet = tracklet(0) # initialize the first tracklet
            gaze_fixation_list = [] # for the final framewise csv recording frame number, gaze point locations.
            # for each frame check if a tracklet can be continued
            for frame_num in range(video_length):
                ret, image = cap.read()
                if (ret == False) : 
                    print("Warning: Cannot read ", frame_num)
                    cap.release()
                    break

                detected_faces = face_detector(image, 0)

                face_record = [] # [cos_similarity, rect obj, 6 lmks]

                for current_face in detected_faces:
                    current_face = current_face.rect
                    # compute landmarks
                    shape = lmk_predictor(image, current_face) 
                    shape = face_utils.shape_to_np(shape)
                    landmarks = []
                    for (x, y) in shape:
                        landmarks.append((x, y))

                    #############################################
                    # modify here for face recognition
                    face_embedding = tools.encode_face()
                    #############################################
                    cos_sim = max(tools.face_similarity(celebEmbeddings, face_embedding)) # get the highest within these 30 faces

                    # Head Pose Estimation
                    landmarks = np.asarray(landmarks)
                    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
                    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
                    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
                    face_record.append((cos_sim, current_face, landmarks_sub))

                if len(face_record) == 0:
                    # No face detected at this stage, we should save the current tracklet, and make the next tracklet
                    face_tracklet.saveAndKill(savePath, "no face detected. Tracklet broken", video_width, video_height)
                    face_tracklet = tracklet(frame_num + 1, tracklet_num=face_tracklet.getNextTrackletNum())
                else:
                    face_record = sorted(face_record, key=lambda x: x[0], reverse=True)[0]
                    cos_sim = face_record[0]
                    current_face = face_record[1]
                    landmarks_sub = face_record[2]
                    landmarks_to_save = landmarks_sub.reshape(6,2)
                    landmarks_to_save[:, 0] /= video_width
                    landmarks_to_save[:, 1] /= video_height
                    landmarks_to_save = landmarks_to_save.reshape(6,1,2)
                    face_bbx = [current_face.left(), current_face.top(), current_face.right(), current_face.bottom()]

                    if cos_sim < args.thres_face:
                        face_tracklet.saveAndKill(savePath, "face tracking failed due to face recognition failure. Tracklet broken", video_width, video_height)
                        face_tracklet = tracklet(frame_num + 1, tracklet_num=face_tracklet.getNextTrackletNum())
                    else:
                        # head pose estimation for gaze estimation
                        hr, ht = gaze_utils.estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
                        # gaze face normalization 
                        img_normalized, landmarks_normalized, R_inv, face_center_camera_cord = gaze_utils.normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)
                        input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
                        input_var = gaze_utils.trans(input_var)
                        input_var = torch.autograd.Variable(input_var.float().cuda())
                        input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
                        pred_gaze, feature = gaze_model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
                        pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
                        pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
                        feature = feature[0].cpu().data.numpy()
                        pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor = gaze_utils.denormalize_predicted_gaze(pred_gaze_np, R_inv)
                        pred_yaw_pitch_cancel_nor = pred_yaw_pitch_cancel_nor.reshape(2)
                        # compute intersection with camera plane 
                        x2d, y2d = gaze_utils.map_to_camera_plane(pred_gaze_cancel_nor, face_center_camera_cord)

                        # 2D pose estimation and association with face bounding boxes.
                        # Please modify below according to your pose estimator.
                        # Here you need to associate the face bounding box with the estimated pose.
                        ###############################################################################
                        humans = tools.compute_pose()
                        
                        pose_list = []
                        for human in humans:    
                            # 0: nose 1: neck 2: left shoulder 5: right shoulder 
                            # but you need to change this part if your pose estimator does not work this way
                            if 0 in human.body_parts and 1 in human.body_parts and 2 in human.body_parts and 5 in human.body_parts:
                                nose = human.body_parts[0]
                                center_nose = np.array([int(nose.x * video_width + 0.5), int(nose.y * video_height + 0.5)])
                                if current_face.left() <= center_nose[0] <= current_face.right() and current_face.top() <= center_nose[1] <= current_face.bottom():
                                    center_bbx = np.array([current_face.right() - current_face.left(), current_face.bottom() - current_face.top()])
                                    distance_to_bbx_center = np.linalg.norm(center_bbx - center_nose)
                                    pose_list.append((distance_to_bbx_center, human))
                        ###############################################################################

                        if len(pose_list) == 0:
                            # No associated pose detected at this stage, we should save the current tracklet, and make the next tracklet
                            face_tracklet.saveAndKill(savePath, "No pose matched. Tracklet broken", video_width, video_height)
                            face_tracklet = tracklet(frame_num + 1, tracklet_num=face_tracklet.getNextTrackletNum())
                            # NB: in this case since pose and face does not match, we cannot perform pose coordinate system estimation, and thus this frame will be skipped
                            # Otherwise, we run pose estimation and the result will be recorded whatever the tracking goes
                        else:
                            human = sorted(pose_list, key=lambda x:x[0])[0][1]
                            # retrieve and estimate keypoints
                            nose = human.body_parts[0]
                            neck = human.body_parts[1]
                            l_shoulder = human.body_parts[2]
                            r_shoulder = human.body_parts[5]
                            center_left = (int(l_shoulder.x * video_width + 0.5), int(l_shoulder.y * video_height + 0.5))
                            center_right = (int(r_shoulder.x * video_width + 0.5), int(r_shoulder.y * video_height + 0.5))
                            center_neck = (int(neck.x * video_width + 0.5), int(neck.y * video_height + 0.5))
                            shoulder_length = np.linalg.norm([center_right[0] - center_left[0], center_right[1] - center_left[1]])
                            center_left_torso = (center_left[0], int(center_left[1] + shoulder_length))
                            center_right_torso = (center_right[0], int(center_right[1] + shoulder_length))
                            center_nose = (center_neck[0], int(center_neck[1] - (shoulder_length / 2 * 0.816)))
                            
                            # estimate body pose coordinate system 
                            pose2d = [list(center_nose), list(center_neck), list(center_left), list(center_right), list(center_left_torso), list(center_right_torso)]
                            pose2d = np.asarray(pose2d).reshape(6,1,2).astype(float)
                            hr, ht = gaze_utils.estimateHeadPose(pose2d, pose3d, camera_matrix, camera_distortion)
                            
                            # The previous hr ht brings points from model to camera. To get gaze direction from camera to model, we compute the inverse transformation
                            hR_inv = gaze_utils.computeInverseTransformation(hr, ht)
                            
                            # Compute the gaze ray (fc_model -> fg_model) in the model coordinate system. Note that we do this in homogenous coordinate system
                            fc_model = np.concatenate((face_center_camera_cord, [[1]]), axis = 0) # homogenous system
                            fc_model = hR_inv @ fc_model

                            fg_model = face_center_camera_cord + pred_gaze_cancel_nor * -112
                            fg_model = np.concatenate((fg_model, [[1]]), axis = 0)
                            fg_model = hR_inv @ fg_model
                            
                            # this is now the gaze vector in the body coordinate system
                            fc_model = fc_model[:-1, :] 
                            fg_model = fg_model[:-1, :]
                            gaze = fg_model - fc_model
                            gaze = gaze / np.linalg.norm(gaze)

                            # compute the gaze point on cylinder 
                            poi = gaze_utils.virtual_intersection(gaze, fc_model, method='cylinder')
                            gaze_fixation_list.append([frame_num, poi[0], poi[1], poi[2], x2d[0], y2d[0]])
                            
                            # check if the tracklet continues
                            if face_tracklet.canTrack(face_bbx):
                                face_tracklet.update(face_bbx, landmarks_to_save, feature)
                            else:
                                face_tracklet.saveAndKill(savePath, "IoU failed. Tracklet broken", video_width, video_height)
                                face_tracklet = tracklet(frame_num + 1, tracklet_num=face_tracklet.getNextTrackletNum())

            # End for frame_num in range(video_length)
            gaze_fixation_list = pd.DataFrame(gaze_fixation_list, columns=['frame', 'x', 'y', 'z', "camera_x", "camera_y"])
            gaze_fixation_list.to_csv(os.path.join(savePath, "fixations.csv"), index=False)
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VoxCeleb2 extracklet extraction')

    parser.add_argument('--segment_length', type=int, default=100,
                        help='the lower bound number of consecutive frames that allows a tracklet to form. Default:100')
    parser.add_argument('--thres_face', type=float, default=0.4,
                        help='threshold for face recognition. Default: 0.4')
    parser.add_argument('--thres_tracking', type=float, default=0.4,
                        help='threshold for face IoU tracking. Default: 0.4')               
    parser.add_argument('--path', type=str, default="randomVox", metavar='P',
                        help='path to the downloaded randomly sampled raw VoxCeleb2 dataset')                    
    args = parser.parse_args()
    run(args)
