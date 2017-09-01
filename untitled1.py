#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:25:15 2017

@author: sreehari
"""

import numpy as np
import cv2
import face_recognition
import os

path = r'train_images'
image_path = [os.path.join(path, f) for f in os.listdir(path)]


known_faces = []
img_dict = {1:'Akshay', 2:'Anirudh', 3:'Shyam', 4:"Tamaghna"}

for i in range(len(image_path)):
    img = face_recognition.load_image_file(image_path[i])
    img_encoding = face_recognition.face_encodings(img)[0]
    known_faces.append(img_encoding)

    
cv2.namedWindow("preview")
cap = cv2.VideoCapture(0) 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)   
    if len(face_recognition.face_encodings(frame)) >= 1:       
         unknown_face_encoding = face_recognition.face_encodings(frame)
         i = 0 
         for face_location in face_locations:
            # Print the location of each face in this image
            top, right, bottom, left = face_location
            #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            ne_img = cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 5)
            #plt.imshow(ne_img)
            unknown_face_encoding_present = unknown_face_encoding[i]
            
            