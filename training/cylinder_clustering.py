###################################################################
# This script contains code to perform clustering
# Given the gaze points of a video on the cylinder, 
# one should first run unrollingCylinder(), then clusterLocation()
###################################################################
#from nntplib import GroupInfo
#from platform import java_ver
#from re import T
#from typing import final
import pandas as pd
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import OPTICS
import time, os, math

def clusterLocation(csvname, csvout, cluster_size=.2, m_eps = -1, plot=False, pltname="unnamed.png"):
    # record current time 
    time_start = time.time()
    # Read in the csv file
    gaze_feature = pd.read_csv(csvname)
    gaze_virtual_location = gaze_feature[['virtual_x', 'virtual_y']] # location 2d on virtual plane
    
    print("perform OPTICS clustering now")
    p = True
    while (p):
        Optics_model_virtual = OPTICS(min_samples=5, n_jobs=-1, xi=.05, min_cluster_size=cluster_size, max_eps = m_eps).fit(gaze_virtual_location)
        m_eps += 1
        if 0 in Optics_model_virtual.labels_:
            p = False

    gaze_feature['optics'] = Optics_model_virtual.labels_
    print('Clustering done! Time elapsed: {} seconds'.format(time.time()-time_start))        

    pseudoLabels = gaze_feature.optics.to_list()

    for i in range(len(pseudoLabels)):
        pseudoLabels[i] += 1
        
    gaze_feature['optics'] = pseudoLabels

    final_df = gaze_feature
    final_df.to_csv(csvout, index = False)

    plt.figure(1)
    ax1 = sns.scatterplot(data=final_df, x="virtual_x", y="virtual_y", hue = "optics", palette="deep", s = 5, legend="full")
    ax1.set(xlim=(0, 6283))
    ax1.set(ylim=(-2000, 2000))
    ax1.invert_yaxis()
    if plot:
        plt.show()
    else:
        p = pathlib.Path("output/")
        p.mkdir(parents=True, exist_ok=True)
        fig = ax1.get_figure()
        fig.savefig("output/" + "o_"+pltname)
        plt.clf()

def unrollingCylinder(csvname, outname, radius=1000):
    ### maps 3D gaze points on the cylinder to 2D rectangluar plane

    # Read in the csv file
    gaze_feature = pd.read_csv(csvname)
    print("reading csv: ", csvname)
    print(f"number of rows {gaze_feature.shape[0]}")

    # eliminate rows with 0s
    gaze_feature = gaze_feature[gaze_feature.x != 0]
    gaze_feature = gaze_feature.reset_index(drop=True)
    print(f"number of rows after elimination {gaze_feature.shape[0]}")


    virtual_y = gaze_feature['y']
    virtual_x = gaze_feature['x'].values.tolist()
    virtual_z = gaze_feature['z'].values.tolist()

    virtual_x_prime = []
    circumference = 2 * np.pi * radius
    anchor_vec = np.array([0,1])
    anchor_vec = anchor_vec / np.linalg.norm(anchor_vec)

    for i in range(len(virtual_x)):
        # compute the unsigned angle here
        xz = np.array([virtual_x[i], virtual_z[i]])
        xz = xz / np.linalg.norm(xz)
        angle = signedAngle(anchor_vec, xz)
        x_prime = (angle / (2 * np.pi)) * circumference
        virtual_x_prime.append(x_prime)
    
    virtual_x_prime = pd.DataFrame(virtual_x_prime, columns=['virtual_x'])
    final_df = pd.concat([gaze_feature, virtual_x_prime], axis=1)
    final_df['virtual_y'] = virtual_y
    final_df.to_csv(outname, index=False)

def signedAngle(va, vb):
    angle = math.atan2( va[0]*vb[1] - va[1]*vb[0], va[0]*vb[0] + va[1]*vb[1] )
    if angle < 0:
        angle += 2 * np.pi
    return angle


