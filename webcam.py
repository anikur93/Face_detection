#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:14:19 2017

@author: sreehari
"""
#modi_image = face_recognition.load_image_file("test_images/modi1.jpg")
#rahul_image = face_recognition.load_image_file("test_images/rg1.jpg")
#megan_image = face_recognition.load_image_file("test_images/megan1.jpg")
#sachin_image = face_recognition.load_image_file("test_images/st1.jpg")
#virat_image = face_recognition.load_image_file("test_images/vk1.jpg")
#
#modi_face_encoding = face_recognition.face_encodings(modi_image)[0]
#rahul_face_encoding = face_recognition.face_encodings(rahul_image)[0]
#megan_face_encoding = face_recognition.face_encodings(megan_image)[0]
#sachin_face_encoding = face_recognition.face_encodings(sachin_image)[0]
#virat_face_encoding = face_recognition.face_encodings(virat_image)[0]
#
#known_faces = [
# modi_face_encoding,
# rahul_face_encoding,
# megan_face_encoding,
# sachin_face_encoding,
# virat_face_encoding
#]

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
            #import face_recognition
            results = face_recognition.compare_faces(known_faces, unknown_face_encoding_present)
            #print(results)
            i = i +1
            
            for i in range(len(results)):
                if results[i] == True:
                    cv2.putText(ne_img,img_dict[i+1],(right,top), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                    
        
        
        # Display the resulting frame
         cv2.imshow('preview',ne_img)
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




