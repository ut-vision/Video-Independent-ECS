import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import os, math, glob, cv2, sys, argparse
import argparse
from pathlib import Path
sys.path.append('./')
from training.cylinder_clustering import *
import pathlib

def run_train(args):
    # convert each video's gaze point onto cylinder plane, apply optics and assign framewise pseudolabels. 

    # getting glob information
    print("getting the list of celebrity ids...")
    directories = glob.glob(os.path.join(args.tpath, "*/"))
    print("...Done!")
    corrupted = 0
    for id_dir in directories:
        id_name = Path(id_dir).parts[-1]
        print("processing celebrity id", id_name) 

        tracklets = os.path.join(id_dir, "tracklets")

        if not os.path.isdir(tracklets):
            print("This id has not been preprocessed. [Tracklet folder not found]")
            continue

        processedVideos = glob.glob(os.path.join(tracklets, "*/"))
        for processedVideo in processedVideos:
            videoname = Path(processedVideo).parts[-1] + ".mkv"
            videopath = os.path.join(args.path, id_name, videoname)
            cap = cv2.VideoCapture(videopath)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 25
            print("video frame length:", video_length)
            if video_length < 120:
                print("This video is too short", video_length)
                continue
            if video_length > 720:
                print("This video is too long", video_length)
                continue

            final_csv_path = os.path.join(processedVideo, "training.csv")

            if args.skip:
                if os.path.isfile(final_csv_path):
                    print(final_csv_path, " found! Next")
                    continue

            detected_tracklets = glob.glob(os.path.join(processedVideo, "*.npz"))
            if len(detected_tracklets) <= 1:
                print("No tracklets found in this folder. Next")
                continue
            # Unrolling the cylinder
            fixation_csv_path = os.path.join(processedVideo, "fixations.csv")
            if not os.path.isfile(fixation_csv_path):
                print(fixation_csv_path, "does not exist. Next")
                continue
            
            fixation_csv = pd.read_csv(fixation_csv_path)
            if fixation_csv.shape[0] < 1500 or fixation_csv.shape[0] > 30000:
                print("video fixation points requirement not satisfied. Next")
                continue

            fixation_out_path = os.path.join(processedVideo, "fixations_unrolled.csv")
            unrollingCylinder(fixation_csv_path, fixation_out_path)
            # OPTICS clustering
            clusterLocation(fixation_out_path, final_csv_path, m_eps = args.m_eps, cluster_size=args.cluster_size, plot=args.plot, pltname = id_name + "_" + Path(processedVideo).parts[-1] +".png")
            os.remove(fixation_out_path)

def label_association_train(args):     
    # associate pseudolabels with the tracklets extracted
    # getting glob information
    print("getting the list of celebrity ids...")
    directories = glob.glob(os.path.join(args.tpath, "*/"))
    print("...Done!")
    corrupted = 0
    for id_dir in directories:
        id_name = Path(id_dir).parts[-1]
        print("processing celebrity id", id_name) 

        tracklets = os.path.join(id_dir, "tracklets")

        if not os.path.isdir(tracklets):
            print("This id has not been preprocessed by tracklet_extraction.py. [Tracklet folder not found]. Next")
            continue

        processedVideos = glob.glob(os.path.join(tracklets, "*/"))
        for processedVideo in processedVideos:
            ####################################################
            videoname = Path(processedVideo).parts[-1] + ".mkv"
            videopath = os.path.join(args.path, id_name, videoname)
            cap = cv2.VideoCapture(videopath)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 25
            print("video frame length:", video_length)
            if video_length < 120:
                print("This video is too short", video_length)
                continue
            if video_length > 720:
                print("This video is too long", video_length)
                continue
            #####################################################

            final_csv_path = os.path.join(processedVideo, "training.csv")
            print(final_csv_path)
            if not os.path.isfile(final_csv_path):
                print(final_csv_path, " not found! Next")
                continue

            final_csv = pd.read_csv(final_csv_path)

            if final_csv.shape[0] < 1500 or final_csv.shape[0] > 30000:
                print("video length requirement not satisfied. Next")
                continue

            detected_tracklets = glob.glob(os.path.join(processedVideo, "*.npz"))
            if len(detected_tracklets) <= 1:
                print("No tracklets found in this folder. Next")
                continue
            
            ### any video pl adjustment / post processing should be here

            for detected_tracklet in detected_tracklets:
                if Path(detected_tracklet).parts[-1][0] == "f" or Path(detected_tracklet).parts[-1][0] == "p": continue 
                print("processing", detected_tracklet)
                npz_name = "pl_" + Path(detected_tracklet).parts[-1]

                savePath = os.path.join(processedVideo, "processed", npz_name)
                p = pathlib.Path(os.path.join(processedVideo, "processed"))
                p.mkdir(parents=True, exist_ok=True)

                tracklet = np.load(detected_tracklet)
                base_index = tracklet['start_frame']
                pseduo_label = []
                trackletLength = tracklet['bbx'].shape[0]

                print("base index:", base_index, "trackletLength:", trackletLength)
                for index in range(base_index, base_index + trackletLength):

                    l = final_csv[final_csv.frame == index]['optics'].values
                    if len(l) == 1 and l[0] in list(range(2)):
                        l = l[0]
                    else:
                        print("corruption happens at", index, "which is", index - base_index, "in the tracklet")
                        l = -1
                        corrupted += 1
                    pseduo_label.append(l)
                    np.savez_compressed(savePath, label=np.array(pseduo_label))
                print("bad records:", corrupted)
                print("")
            print("-------------------------------------------")

def gt_extraction(args):
    # Generates fixations.csv under each video folder. This associates a label for each frame with a gaze point computed
    # You need to run tracklet_extraction.py first
    # define the positive tag here
    positiveTag = ['host', 'host2', 'host3', 'camera']

    videoList = pd.read_csv(os.path.join(args.annotation, 'meta_list.csv'))
    for idx, row in videoList.iterrows():
        print("processing", row['celebrity_id'], row['video_name'])
        # reading annotation into a list of list
        elanTxt = os.path.join(args.annotation, row['celebrity_id'], row['video_name']+".txt")
        annotation = []
        f = open(elanTxt, 'r')
        for line in f.readlines():
            cols = line.split('\t')
            annotation.append([cols[0], cols[2], cols[3]])
        f.close()

        # reading fixation point files 
        fixationCsv = os.path.join(args.path, row['celebrity_id'], 'tracklets', row['video_name']+"_fps25", 'fixations.csv')
        fixationCsv = pd.read_csv(fixationCsv)
        fixationCsv['groundTruth'] = 0
        fixationCsv['gazeTarget'] = "None"

        for a in annotation:
            if a[0] in positiveTag:
                
                start_frame = math.ceil(float(a[1]) * 25)
                end_frame = math.floor(float(a[2]) * 25)
                for frame in range(start_frame, end_frame + 1):
                    fixationCsv.loc[fixationCsv.frame == frame, 'groundTruth'] = 1
                    fixationCsv.loc[fixationCsv.frame == frame, 'gazeTarget'] = a[0]

        fixationCsv.to_csv(os.path.join(args.path, row['celebrity_id'], 'tracklets', row['video_name']+"_fps25", 'fixations_GT.csv'), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RandomVox Pseduo Label extraction (clustering)')

    parser.add_argument('--m_eps', type=int, default=20,
                        help='max_eps value for optics clustering. Default:20')    
    parser.add_argument('--cluster_size', type=float, default=0.2,
                        help='minimum cluster size for optics (0-1). Default:0.2')   

    parser.add_argument('--skip', action='store_true',
                        help='Whether to skip those a clustered.csv has been generated. Used for resumption.')  
                  

    parser.add_argument('--path', type=str, default="randomVox", metavar='P',
                        help='path to the downloaded raw random VoxCeleb2 dataset') 
    parser.add_argument('--tpath', type=str, default="randomVox", metavar='P',
                        help='path to processed tracklets folder for training') 
    parser.add_argument('--annotation', type=str, default="Annotations", metavar='A',
                        help='path to the annotation folder')   

    parser.add_argument('--gt', action='store_true',
                        help='ground truth extraction from sec:msec interval to frame num')
    parser.add_argument('--run_train', action='store_true',
                        help='PL extractor on train set')
    parser.add_argument('--reset', action='store_true',
                        help='reset PL /gt labels (should be combined with run_test/train)')
    parser.add_argument('--associate', action='store_true',
                        help='tracklet pl assocation')

    parser.add_argument('--plot', action='store_true',
                        help='show plots after each being processed')

    args = parser.parse_args()


    if args.gt:
        print('GroundTruth Extraction and Conversion')
        gt_extraction(args)

    if args.reset:
        print("reset training tracklet PLs")
        reset_labels(args)

    if args.run_train:
        print('PL generator on training set')
        run_train(args)

    if args.associate:
        print('PL tracklet association')
        label_association_train(args)
    
            


