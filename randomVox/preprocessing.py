import os
import glob
import pandas as pd
import subprocess
import argparse
'''
Created: 2021.08.29
Last update: 2022.03.24 for mkv 25fps support
'''

# This function seperates the downloaded mp4/mkv/webm file into mkv format, plus setting the fps to 25.
def generate_25fpsmkv():
    print("getting the list of ids...")
    directories = glob.glob("*/")    
    print("...Done!")
    for id_dir in directories:
        id_name = id_dir[:-1]
        print("processing celebrity id ", id_name) 
        os.chdir(id_dir) # cd into id000xx folder
        downloaded_mp4 = glob.glob('*.mp4')
        downloaded_mkv = glob.glob('*.mkv')
        downloaded_webm = glob.glob('*.webm')
        downloaded_video = downloaded_mkv + downloaded_mp4 + downloaded_webm
        for video in downloaded_video:
            if "_fps25.mkv" in video: continue
            ytb_id = ".".join(video.split(".")[:-1])
            print("------video", video, "id=", id_name, ytb_id)
            if os.path.isfile('./' + ytb_id + "_fps25.mkv"):
                print("------The 25fps mkv of this file already exists")
            else:
                ### convert the mp4/mkv to avi
                print("------Convert", video, "to mkv")
                command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (video, ytb_id+"_fps25.mkv")) # -y:overwrite -r frame rate -async stretch audio to sync -
                #command = ("ffmpeg -y -i %s -filter:v -r 25 %s" % (video, ytb_id+".avi")) # -y:overwrite -r frame rate -async stretch audio to sync -
                output = subprocess.call(command, shell=True, stdout=None)
        os.chdir("../")

# delete all files with specified extension 
def delete_file(extension=".csv", skip_word=None):
    print("getting the list of ids...")
    directories = glob.glob("*/")
    print("...Done!")
    directories.sort()

    for id_dir in directories:
        id_name = id_dir[:-1]
        print("processing celebrity id ", id_name) 
        os.chdir(id_dir) # cd into id000xx folder 
        files_to_delete = glob.glob('*' + extension) 
        for f in files_to_delete:
            if skip_word is not None:
                if skip_word in f:
                    continue
            print("deleting", id_name, f)
            os.remove(f)
        os.chdir("../")

# only keeps processed _fps25.mkv format videos
def clean():
    delete_file(".ytdl", skip_word = "fps25")
    delete_file(".part", skip_word = "fps25")
    delete_file(".mp4", skip_word = "fps25")
    delete_file(".avi", skip_word = "fps25")
    delete_file(".wav", skip_word = "fps25")
    delete_file(".mkv", skip_word = "fps25")
    delete_file(".webm", skip_word = "fps25")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='VoxCeleb2 Raw Videos Preprocessing')
    parser.add_argument('--convert', action='store_true',
                        help='convert all videos to 25fps mkv')
    parser.add_argument('--clean', action='store_true',
                        help='clean all files other than 25fps mkv. used after --convert')
    
    args = parser.parse_args()

    if args.convert: 
        generate_25fpsmkv()
    if args.clean:
        clean()
