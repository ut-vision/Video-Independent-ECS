# RandomVox Dataset

The raw Voxceleb2 Dataset we used for experiments is not provided.

However, we provide a list of videos (youtube IDs) with celebrity IDs (VoxCeleb2 dataset) in randomVoxMeta.csv.

Please first run 

> python download.py 

to collect raw videos from Youtube and then run 

> python preprocess.py --convert 

to convert all videos to 25fps in mkv format.

Finally, you can clean all redundant files with

> python preprocess.py --clean
