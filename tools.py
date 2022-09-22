import numpy as np
# please implement your face recognition model and 2D pose estimatior here

def encode_face():
    '''
    This function should take input a face image patch and outputs its face embedding
    '''
    face_embedding = np.array([0.1] * 512) # This is just to ensure the whole program runs without error. 
    return face_embedding

def compute_pose():
    '''
    This function should take input the whole image and output all joint keypoints of all humans. (Assuming a top-down pose estimator)
    Otherwise you can also assume a bottom-up pose estimator, but you need to modify accordingly.
    '''
    class keypoint():
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class skeleton():
        def __init__(self):
            self.body_parts = {
            0: keypoint(0.3,0.3),
            1: keypoint(0.3,0.4),
            2: keypoint(0.2,0.4),
            5: keypoint(0.4,0.4),
            }
    
    human_skeletons = [skeleton()]  # This is just to ensure the whole program runs without error. 
    return human_skeletons

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def face_similarity(known_face_encodings, face_encoding_to_check):
    # both should be np array
    res = []
    for known_encoding in known_face_encodings:
        res.append(cosin_metric(known_encoding, face_encoding_to_check))
    return res