Eye Contact Detection on VoxCeleb2 Dataset

This dataset contains eye contact annotation for 52 videos selected from VoxCeleb2 Dataset.
Each foldername indicates the celebrity ID. 
Each celebrity's face image can be found in VGGFACE2 dataset by querying the same ID.
Each folder contains one annotation file for one video in the .txt (tab deliminated text) format created by ELAN 6.4. 
The filename of the annotation file is the youtube ID of this video converted to 25fps. 
We do not directly provide these raw videos.
These annotation files contain one-way eye contact annotation of the celebrity with the camera and other human who he interacts to, and the gaze targets are indicated as "host", "host2", "host3" and "camera".
We also assign "uncertain" segments to intervals where annotation is impossible (low resolution, invisibility of eye region, and irrelavent scenes.)