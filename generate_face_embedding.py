# This script aims at extracting targeted speaker's face embedding at a csv file under its root folder, face_embedding.csv.
# You need to first download VGGFace2 dataset and place it under the root.

import glob, os, argparse
import cv2, dlib, torch
import pandas as pd
import numpy as np
from tools import *

if __name__ =="__main":
    parser = argparse.ArgumentParser(description='Generating face embeddings for each celebrity')
    parser.add_argument('--num', type=int, default=30,
                            help='number of face embeddings to sample from VGGFace dataset, default: 30')
    parser.add_argument('--path', type=str, default='randomVox',
                            help='dataset path to VoxCeleb2 dataset')
    args = parser.parse_args()

    num_to_compute = args.num
    print("loading face models (detection, landmark, and recognition)")
    face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
    lmk_predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks_GTX.dat')
    # reading vox_face_conversion.csv to get a mapping between VGGface2 and VoxCeleb2 celebrities.
    # This file can be obtained from VGGFace2 dataset
    vox_face_conversion = pd.read_csv("vox_face_conversion.csv")
    # get root path
    root_path = os.getcwd()
    # getting glob information
    os.chdir(args.path)
    print("getting the list of celebrity ids...")
    directories = glob.glob("*/")
    print("...Done!")

    for id_dir in directories:
        id_name = id_dir[:-1]
        print("processing celebrity id ", id_name) 
        os.chdir(id_dir) # cd into id000xx folder 

        if os.path.isfile('./' + id_name + "_face_embedding.npz"):
            print(id_name, "already processed, go to next")
            os.chdir("../")
            continue

        if len(glob.glob("*_fps25.mkv")) + len(glob.glob("*.mkv")) + len(glob.glob("*.webm"))== 0:
            print(id_name, "no video found. Skip")
            os.chdir("../")
            continue

        embedding_set = [] 
        vox_face_mapping = vox_face_conversion[vox_face_conversion['VoxCeleb2 ID '] == id_name+" "]
        if vox_face_mapping.empty:
            print("--!! a mapping cannot be found. Skip this one")
            os.chdir("../")
            continue

        vgg_face_id = vox_face_mapping.iloc[0, 1]
        # now reading a couple images and compute their face embedding and saving into the embedding set
        vgg_face_path = root_path + '\\VGG-Face2\\data\\train\\' + vgg_face_id.strip()
        print("randomly sampling faces from", vgg_face_path)
        if not os.path.isdir(vgg_face_path):
            print("Warning:", vgg_face_path, "not exist")
            print("Trying the test set.....")
            vgg_face_path = root_path + '\\VGG-Face2\\data\\test\\' + vgg_face_id.strip()
            print("randomly sampling faces from", vgg_face_path)
            if not os.path.isdir(vgg_face_path):
                print("Warning:", vgg_face_path, "not exist")
                print("proceed to next")
                os.chdir("../")
                continue
            
        # now select randomly n images from this folder and compute their face embedding.
        trainingSamples = os.listdir(vgg_face_path)
        i = 0
        while len(embedding_set) < num_to_compute:
            face_image_path = os.path.join(vgg_face_path, trainingSamples[i] ) # This is now the first 30 faces, and is thus deterministic
            face_image = cv2.imread(face_image_path)

            #############################################################
            # modify here for face recognition.
            face_embedding = encode_face(face_image)
            #############################################################

            if face_embedding is not None:
                embedding_set.append(face_embedding)
            i += 1
            if i >= len(trainingSamples): break

        embedding_set = np.array(embedding_set)
        np.savez_compressed(id_name + "_face_embedding.npz", embedding_set)
        print("Done!")
        os.chdir("../")