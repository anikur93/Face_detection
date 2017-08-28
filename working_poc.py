import matplotlib.pyplot as plt
import face_recognition
import cv2

image = face_recognition.load_image_file("test_images/modi2.jpg")
plt.imshow(image)

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)

#print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    ne_img = cv2.rectangle(image, (left, top), (right, bottom), (255,0,0), 5)
    plt.imshow(ne_img)
    
    
import face_recognition

# Load the jpg files into numpy arrays
##traning images
modi_image = face_recognition.load_image_file("test_images/modi1.jpg")
rahul_image = face_recognition.load_image_file("test_images/rg1.jpg")
megan_image = face_recognition.load_image_file("test_images/megan1.jpg")
sachin_image = face_recognition.load_image_file("test_images/st1.jpg")
virat_image = face_recognition.load_image_file("test_images/vk1.jpg")


unknown_image = image
#unknown_image = face_recognition.load_image_file("test_images/modi2.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encordings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]
rahul_face_encoding = face_recognition.face_encodings(rahul_image)[0]
megan_face_encoding = face_recognition.face_encodings(megan_image)[0]
sachin_face_encoding = face_recognition.face_encodings(sachin_image)[0]
virat_face_encoding = face_recognition.face_encodings(virat_image)[0]


unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

known_faces = [
    modi_face_encoding,
    rahul_face_encoding,
    megan_face_encoding,
    sachin_face_encoding,
    virat_face_encoding
]



    

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
    

#print("Is the unknown face a picture of modi? {}".format(results[0]))
#print("Is the unknown face a picture of rahul? {}".format(results[1]))
#print("Is the unknown face a picture of megan? {}".format(results[2]))
#print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
print(results)

# =============================================================================
# FROM HERE!!!!!!
# 
# 
# 
# 
# =============================================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:55:30 2017

@author: sreehari
"""
import numpy as np
import cv2
import face_recognition
import os

path = r'/home/sreehari/Image_project/train_images'

cap.release()
cv2.destroyAllWindows()

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)  
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
        ne_img = cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 5)
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
    cv2.imshow('preview',ne_img)
    #cv2.imshow('preview', cropped_img)
    #cv2.imwrite('akshay'+str(i)+'.png',cropped_img)
    if (cv2.waitKey(1) & 0xFF == ord('q')) | i > 150:
        break
    
cap.release()
cv2.destroyAllWindows()

from PIL import Image

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    images = []
    labels = []
    for image_path in image_paths:
        image = Image.open(image_path)
        nbr = os.path.split(image_path)[1].split('.')[0]
        label = ''.join(x for x in nbr if x.isalpha())
        images.append(image)
        labels.append(label)
        #cv2.imshow('Adding faces to training set...', image)
        cv2.waitKey(50)
    return images, labels

images, labels = get_images_and_labels(path)
    
def augment_brightness_camera_images(image):
    image_bright = []
    for i in images:
        img[i] = cv2.cvtColor(img[i],cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        #print(random_bright)
        img[i][:,:,2] = img[i][:,:,2]*random_bright
        img[i] = cv2.cvtColor(img[i],cv2.COLOR_HSV2RGB)
        image_bright.append(img[i])
    return image_bright

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    # Brightness
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
    
    for i in img:
        image[i] = cv2.warpAffine(image[i],Rot_M,(cols,rows))
        image[i] = cv2.warpAffine(image[i],Trans_M,(cols,rows))
        image[i] = cv2.warpAffine(image[i],shear_M,(cols,rows))

        if brightness == 1:
            image[i] = augment_brightness_camera_images(image[i])

    return img




# When everything done, release the capture
