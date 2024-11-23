import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'Images'  # Replace 'Images' with the path to your image dataset

def img(path):
    imgpaths = [os.path.join(path, f) for f in os.listdir(path)] 
    faces = []
    users = []
    for imgpath in imgpaths:
        faceimg = Image.open(imgpath).convert('L')
        facenp = np.array(faceimg, 'uint8')
        
        # Extract the last part of the filename and attempt conversion to an integer
        filename = os.path.splitext(os.path.basename(imgpath))[0]
        user_label = filename.split('_')[-1]
        
        try:
            user = int(user_label)
            faces.append(facenp)
            users.append(user)
        except ValueError:
            print(f"Skipping non-numeric user label: {user_label}")

    return users, faces

users, faces = img(path)
recognizer.train(faces, np.array(users))
recognizer.save('TrainedDataFolder/TraningData.yml')
