import os
import glob
import pandas as pd
import subprocess
import atexit
import pathlib
'''
Requirement: yt-dlb, pandas
'''

# This class only defines a struct. 
# used by atexit only
class ind_recorder(): 
    ind = 0
    pwd = ""

# This function will be automatically triggered when on exit. (Including ctrl+c)
# It records the current ind_recorder.ind, so that next time this program can continue at the point where it ends at last time
def exit_handler():
    os.chdir(ind_recorder.pwd)
    print('Saving progress...', ind_recorder.ind)    

    f = open("download_memo.txt","w")
    f.write(str(ind_recorder.ind) + "\n")
    f.close() 
    print('Done!')  
atexit.register(exit_handler)

# The function that actual does the downloading using youtube-dl
def download_video():
    ind_recorder.pwd = os.getcwd()
    # Check if the metadata file exists
    if not os.path.isfile('./randomVoxMeta.csv'):
        return
    # Check if we have the download history
    if not os.path.isfile('./download_memo.txt'):
        ind_recorder.ind = 0
        print("New_Start : index =", ind_recorder.ind)
    else:
        f = open("./download_memo.txt", "r+")
        ind_recorder.ind = int(f.read())
        f.close()
        print("Found previous downloading history, start from index =", ind_recorder.ind)
    
    metadata = pd.read_csv('randomVoxMeta.csv')
    metadata = metadata.iloc[ind_recorder.ind:, :]

    for _, data in metadata.iterrows():
        id_name = data['celebrity_id']
        ytb_link = data['video_name']
        print("...Trying to download ", id_name, ytb_link, "||index=", ind_recorder.ind, "||")
        
        if os.path.isdir(id_name):
            os.chdir(id_name)
        else:
            p = pathlib.Path(id_name)
            p.mkdir(parents=True, exist_ok=True)
            os.chdir(id_name)
        
        if os.path.isfile('./' + ytb_link + ".mp4") or os.path.isfile('./' + ytb_link + ".mkv") or os.path.isfile('./' + ytb_link + ".avi"):
            print(" - This file has been downloaded")
        else:
            #subprocess.run(["youtube-dl", "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]", "-o", "%(id)s.%(ext)s", ytb_link])
            subprocess.run(["yt-dlp", "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]", "-o", "%(id)s.%(ext)s", ytb_link])
            print("...Done!")
        os.chdir("../")
        ind_recorder.ind += 1
        print("--------------------")
 
if __name__ == "__main__":
    download_video() 