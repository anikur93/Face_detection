#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:25:31 20174
@author: sreehari
"""
import cv2
image = face_recognition.load_image_file("test_images/modi3.jpg")
plt.imshow(image)

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    ne_img = cv2.rectangle(image, (left, top), (right, bottom), (255,0,0), 2)
    plt.imshow(ne_img)

    # You can access the actual face itself like this:
    #face_image = image[top:bottom, left:right]
    #pil_image = Image.fromarray(face_image)
    #pil_image.show()
    #plt.imshow(pil_image)