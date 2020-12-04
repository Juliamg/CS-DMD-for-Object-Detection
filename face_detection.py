import numpy as np
import os
import time
import cv2

"""
This script will take the frames given from DMD and detect where in the frame a face is to crop this image 
and feed it further into the src classifier which will do the classification job

Output is a folder with prepped images to be passed into src
"""

shape = (170, 170)
padding = 5
IGNORE_FILES = ['.DS_Store']

def process_DMD_snapshots(data_path): # Path to train folder residing in Data folder
    print("Running FACEDET")
    time.sleep(3)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    data_folders = os.listdir(data_path)

    for folder in data_folders:
        if folder in IGNORE_FILES:
            continue
        face_count = 0
        max_samples = 0
        for file in os.listdir(data_path + os.sep + folder):
            img_path = data_path + os.sep + folder + os.sep + file
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 8)

            if len(faces) == 0:
                os.remove(img_path) # No face detected - delete snapshot

            if max_samples >= 11:
                try:
                    os.remove(img_path)
                except:
                    pass
                continue

            # Check if faces contain a face
            for (x,y,w,h) in faces:
                face_count += 1
                max_samples += 1

                ### Crop face and create new image ###
                crop_img = img[y:y+h+padding, x:x+w+padding]
                resized_img = cv2.resize(crop_img, shape, interpolation=cv2.INTER_AREA)
                cv2.imwrite(img_path, resized_img)
                break # First detected face is probably the correct one


        print(f"Detected {face_count} faces for subject {folder}")