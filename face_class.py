#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:11:28 2017

@author: sreehari
"""

import face_recognition

# Load the jpg files into numpy arrays
modi_image = face_recognition.load_image_file("test_images/modi1.jpg")
rahul_image = face_recognition.load_image_file("test_images/rg1.jpg")
megan_image = face_recognition.load_image_file("test_images/megan1.jpg")
unknown_image = face_recognition.load_image_file("test_images/modi2.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encordings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]
rahul_face_encoding = face_recognition.face_encodings(rahul_image)[0]
megan_face_encoding = face_recognition.face_encodings(megan_image)[0]
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

known_faces = [
    modi_face_encoding,
    rahul_face_encoding,
    megan_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of modi? {}".format(results[0]))
print("Is the unknown face a picture of rahul? {}".format(results[1]))
print("Is the unknown face a picture of megan? {}".format(results[2]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
