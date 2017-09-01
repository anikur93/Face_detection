#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:02:34 2017

@author: sreehari
"""

import numpy as np
import cv2
import face_recognition
import os

def image_capture(name,number):
    
    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(1)  
    i =0  
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        face_locations = face_recognition.face_locations(frame)   
        i =i +1
        #if len(face_recognition.face_encodings(frame)) >= 1:       
        #     unknown_face_encoding = face_recognition.face_encodings(frame)
         #    i = 0 
         
        for face_location in face_locations:
                # Print the location of each face in this image
            top, right, bottom, left = face_location
            #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            #ne_img = cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 5)
            cropped_img = frame[top:bottom, left:right]
            
                #plt.imshow(ne_img) xywh y:y+h x:x+w
                #unknown_face_encoding_present = unknown_face_encoding[i]
                #import face_recognition
                #results = face_recognition.compare_faces(known_faces, unknown_face_encoding_present)
                #print(results)
                #i = i +1
                
              #  for i in range(len(results)):
               #     if results[i] == True:
                #        cv2.putText(ne_img,img_dict[i+1],(right,top), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                        
            
            
            # Display the resulting frame
        cv2.imshow('preview',frame)
        #cv2.imshow('preview', cropped_img)
        cv2.imwrite('tamaghna'+str(i)+'.png',cropped_img)
        if (cv2.waitKey(1) & 0xFF == ord('q')) | i > 200:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
image_capture()    
    
cap.release()
cv2.destroyAllWindows()   
    