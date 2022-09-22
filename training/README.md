# Code For Experiments 

This folder contains code for reproducing the experiments after extracting tracklets.

pseudo_label_generator.py contains Pseudo Label level actions. (PL generation using OPTICS, PL level evaluation and etc).

create_dataset.py should be used after PLgenerator.py, and it generates a csv file containing a list of training and test tracklets for dataloader to refer to.

ECSTrainer.py contains code for training our proposed segmentation method.

