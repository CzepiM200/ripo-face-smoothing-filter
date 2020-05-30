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

    #Detekcja twarzy
    rects = detector(gray, 0)

    #Rysowanie kwadratu na wykrytej twarzy
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #Detekja oczu
        eyes = eyesPredictor(gray, rect)
        eyes = face_utils.shape_to_np(eyes)
		
        for (sX, sY) in eyes:
            cv.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
        
        #Detekcja ust
        mouths = mouthPredictor(gray, rect)
        mouths = face_utils.shape_to_np(mouths)
		
        for (mX, mY) in mouths:
            cv.circle(frame, (mX, mY), 1, (255, 0, 0), -1)

    #if filter_enabled == 1:
        #Tu miejsce na filtrację


    #frame = cv.resize(frame, (int(frame.shape[1] / frame.shape[0] * 800), 800))
    cv.imshow("Face smoothing", frame)

    #Zamknięcie po wciśnięciu klawisza ESC
    key = cv.waitKey(10)
    if key == 27:
        break

#Zamknięcie okna
cv.destroyWindow("Face smoothing")
#Zwolnienie dostępu do kamery
feed.release()
