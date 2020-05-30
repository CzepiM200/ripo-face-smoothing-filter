#!/usr/bin/python3

import cv2 as cv
import numpy as np
from imutils import face_utils
import argparse
import imutils
import time
import dlib

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-epredictor", required=True,
# 	help="./ibug_300W_large_face_landmark_dataset/eye_predictor.dat")
# ap.add_argument("-p", "--shape-mpredictor", required=True,
# 	help="./ibug_300W_large_face_landmark_dataset/mouth_predictor.dat")
# args = vars(ap.parse_args())


def addRed(image):
	# split the image into its BGR components
	(B, G, R) = cv.split(image)
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	#R[R < M] = 0
	G[G < 200] = 0
	B[B < 200] = 0
	# merge the channels back together and return the image
	return cv.merge([B, G, R])
    
#Wczytywanie prekyktorów oczy i ust
detector = dlib.get_frontal_face_detector()
eyesPredictor = dlib.shape_predictor('./eye_predictor.dat')
mouthPredictor = dlib.shape_predictor('./mouth_predictor.dat')

#Zmienna określająca stan włączenia filtru
filter_enabled = 0

#Funkcja przełączająca filtr
def toggle(x):
    global filter_enabled
    filter_enabled = x

#Utworzenie okna
cv.namedWindow("Face smoothing")

#Pobranie feedu z kamery
feed = cv.VideoCapture(0)

#Przełącznik filtru
cv.createTrackbar("Filtr", "Face smoothing", 0, 1, toggle)

#Czytanie obrazu z kamery dopóki jest dostępna
if feed.isOpened():
    opened, frame = feed.read()
else:
    opened = False

#Dopóki obraz z kamery jest wczytywany, wyświetlanie go
while opened:
    opened, frame = feed.read()

    #Konwersja na skalę szarości - pozwala na wykrycie twarzy
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    frame = imutils.resize(frame, width=640)
    original = frame.copy()

    #Detekcja twarzy
    rects = detector(gray, 0)

    #Rysowanie kwadratu na wykrytej twarzy
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        #Rysowane prostokąta twarzy
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #Detekja oczu
        eyes = eyesPredictor(gray, rect)
        eyes = face_utils.shape_to_np(eyes)
		
        #Rysowanie punktów oka
        for (sX, sY) in eyes:
            cv.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
        
        leftEyeImage = frame[eyes[2,1]:eyes[5,1], eyes[0,0]:eyes[3,0]].copy()
        cv.imshow("LeftEye", leftEyeImage) 
        rightEyeImage = frame[eyes[7,1]:eyes[10,1], eyes[6,0]:eyes[9,0]].copy()
        cv.imshow("RighttEye", leftEyeImage)

        #Detekcja ust
        mouths = mouthPredictor(gray, rect)
        mouths = face_utils.shape_to_np(mouths)

        #Rysowanie punktów ust
        for (mX, mY) in mouths:
            #cv.circle(frame, (mX, mY), 1, (255, 0, 0), -1)
            cv.circle(frame, (mX, mY), 1, (255, 0, 0), -1)
            
        mouthImage = frame[mouths[3,1]:mouths[9,1], mouths[1,0]:mouths[7,0]].copy()
        mouthImage = addRed(mouthImage).copy()
        cv.imshow("Mouth", mouthImage)            
                
            
            
        
        if filter_enabled == 1:
            
            # X lewy i prawy print(mouths[0,0], mouths[8,0]);
            # Y górny dolny print(mouths[0,1],mouths[6,1])

            #Filtr całej twarzy
            faceImage = frame[y:y+h, x:x+w].copy()
            faceImage = cv.GaussianBlur(faceImage,(5,5),0)
            frame[y:y+h, x:x+w] = faceImage.copy()

            frame[mouths[3,1]:mouths[9,1], mouths[1,0]:mouths[7,0]] = mouthImage.copy()
            frame[eyes[2,1]:eyes[5,1], eyes[0,0]:eyes[3,0]] = leftEyeImage.copy()
            frame[eyes[7,1]:eyes[10,1], eyes[6,0]:eyes[9,0]] = rightEyeImage.copy()


    frame = cv.GaussianBlur(frame,(5,5),0)
    cv.imshow("Face smoothing", frame)

    #Zamknięcie po wciśnięciu klawisza ESC
    key = cv.waitKey(10)
    if key == 27:
        break

#Zamknięcie okna
cv.destroyWindow("Face smoothing")

#Zwolnienie dostępu do kamery
feed.release()
