import glob, os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# This script generates voxdataset_train.csv and voxdataset_test.csv file under pwd
# voxdataset_*.csv is used for torch dataloaders loading all training and testing data

def make_training(args):
    metacsv = []
    nolabel = []
    total_frames = 0
    negative = 0
    total = 0
    # getting glob information
    print("getting the list of celebrity ids...")
    directories = glob.glob(os.path.join(args.trainpath, "*/"))
    print("...Done!")

    for id_dir in directories:
        id_name = Path(id_dir).parts[-1]
        print("processing celebrity id", id_name) 

        tracklets = os.path.join(id_dir, "tracklets")
        if not os.path.isdir(tracklets):
            print("This id has not been preprocessed. [Tracklet folder not found]")
            continue

        processedVideos = glob.glob(os.path.join(tracklets, "*/"))
        for processedVideo in processedVideos:
            video_name = Path(processedVideo).parts[-1]
            detected_tracklets = glob.glob(os.path.join(processedVideo, "*.npz"))
            

            for detected_tracklet in detected_tracklets:
                if Path(detected_tracklet).parts[-1][0] == "y" and Path(detected_tracklet).parts[-1][1] == "p": continue
                if Path(detected_tracklet).parts[-1][0] == "f" and Path(detected_tracklet).parts[-1][1] == "i": continue

                print("processing", detected_tracklet)
                npz_name = Path(detected_tracklet).parts[-1]
                tracklet = np.load(detected_tracklet)
                start_frame = tracklet['start_frame']

                gaze_feature = tracklet['gaze_feature']
                if args.minimum != -1 and args.minimum > gaze_feature.shape[0]:
                    print(gaze_feature.shape[0], "<", args.minimum, "Next")
                    continue
                
                pl_npz = os.path.join(processedVideo, 'processed', 'pl_'+npz_name)
                if not os.path.isfile(pl_npz):
                    print("labels not found. Next")
                    nolabel.append(detected_tracklet)
                    continue
                metacsv.append([id_name, video_name, npz_name, start_frame, start_frame + gaze_feature.shape[0]])
                total_frames += gaze_feature.shape[0]

                labels = np.load(pl_npz, allow_pickle = True)['label']
                total += labels.shape[0]
                for i in labels:
                    if i == 0: negative += 1

    df = pd.DataFrame(metacsv, columns = ["id", "video", "tracklet", "start", "end"])
    df.to_csv("voxdataset_train.csv", index=False)
    print("negative:", negative, "total_frames", total_frames)
    print("Trackelets without labels:")
    for l in nolabel:
        print(l)


def make_testing(args):

    metacsv = []
    nolabel = []
    total_frames = 0
    negative = 0
    total = 0
    videoList = pd.read_csv(os.path.join(args.annotation, 'meta_list.csv'))

    for idx, row in videoList.iterrows():
        print("processing", row['celebrity_id'], row['video_name'])
        detected_tracklets = glob.glob(os.path.join(args.testpath, row['celebrity_id'], 'tracklets', row['video_name']+"_fps25", "*.npz"))
        for detected_tracklet in detected_tracklets:
            if Path(detected_tracklet).parts[-1][0] == "y" and Path(detected_tracklet).parts[-1][1] == "p": continue
            if Path(detected_tracklet).parts[-1][0] == "f" and Path(detected_tracklet).parts[-1][1] == "i": continue

            print("processing", detected_tracklet)
            npz_name = Path(detected_tracklet).parts[-1]
            tracklet = np.load(detected_tracklet)
            start_frame = tracklet['start_frame']

            gaze_feature = tracklet['gaze_feature']
            if args.minimum != -1 and args.minimum > gaze_feature.shape[0]:
                print(gaze_feature.shape[0], "<", args.minimum, "Next")
                continue
            
            pl_npz = os.path.join(args.testpath, row['celebrity_id'], 'tracklets', row['video_name']+"_fps25", 'processed', 'pl_'+npz_name)
            if not os.path.isfile(pl_npz):
                print("labels not found. Next")
                nolabel.append(detected_tracklet)
                continue
            metacsv.append([row['celebrity_id'], row['video_name']+"_fps25", npz_name, start_frame, start_frame + gaze_feature.shape[0]])

            total_frames += gaze_feature.shape[0]

            labels = np.load(pl_npz)['label']
            total += labels.shape[0]
            for i in labels:
                if i == 0: negative += 1

    df = pd.DataFrame(metacsv, columns = ["id", "video", "tracklet", "start", "end"])
    df.to_csv("voxdataset_test.csv", index=False)
    print("negative:", negative, "total_frames", total_frames)
    print("Trackelets without labels:")
    for l in nolabel:
        print(l)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VoxCeleb2 create Dataset csv file')
    parser.add_argument('--trainpath', type=str, default="randomVox", metavar='P',
                        help='path to the downloaded raw VoxCeleb2 dataset')
    parser.add_argument('--testpath', type=str, default="<add path here>", metavar='P',
                        help='path to the downloaded raw VoxCeleb2 dataset') 
    parser.add_argument('--annotation', type=str, default="<add path here>", metavar='A',
                        help='path to the annotation folder')   
    parser.add_argument('--make_training', action='store_true')
    parser.add_argument('--make_testing', action='store_true')
    parser.add_argument('--minimum', type=int, default=-1)
                    
    args = parser.parse_args()

    if args.make_training:
        make_training(args)
    if args.make_testing:
        make_testing(args)
