#!/usr/bin/python3

import cv2 as cv
#Import funkcji wł/wył filtr
from smoothing_filter import toggle
import numpy as np
#Utworzenie okna
cv.namedWindow("Face smoothing")

#Pobranie feedu z kamery
feed = cv.VideoCapture(0)

#Załadowaine klasyfikatora
face_detection = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_detection = cv.CascadeClassifier("haarcascade_eye.xml")
#mouth_detection = cv.CascadeClassifier(cv.data.haarcascades + "Mouth.xml")

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
    mainFrame = frame.copy()

    #Konwersja na skalę szarości - pozwala na wykrycie twarzy
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, minNeighbors=6, minSize=(100, 100))
    

    #Rysowanie kwadratu na wykrytej twarzy
    for (x, y, width, height) in faces:
        cv.rectangle(frame, (x, y), (x + width, y + width), (0xD9, 0x67, 0x04), 2)
        face_color = frame[y:y+height, x:x+width]
        
        #twarz
        mainFace = frame[y:y+height, x:x+width].copy()
        
        #wykrywanie oczu
        eye_gray = gray[y:y+height, x:x+width]
        eye_color = frame[y:y+height, x:x+width]
        eyes = eyes_detection.detectMultiScale(eye_gray, minNeighbors=6, minSize=(30, 30))
        for (eye_x, eye_y, eye_width, eye_height) in eyes:
            #cv.rectangle(eye_color, (eye_x, eye_y), (eye_x+eye_width, eye_y+eye_height), (0xD9, 0xA2, 0x3D), 1)
            
            #nakładanie filtru na twarz
            kernel = np.ones((5,5),np.float32)/25
            mainFace = cv.filter2D(eye_color,-1,kernel)

            #nakładanie oczu bez filtru na twarz
            mainFace[eye_y:eye_y+eye_height, eye_x:eye_x+eye_width] = eye_color[eye_y:eye_y+eye_height, eye_x:eye_x+eye_width].copy()
            mainFrame[y:y+height, x:x+width] = mainFace.copy()

    # kernel = np.ones((5,5),np.float32)/25
    # mainFrame = cv.filter2D(mainFrame,-1,kernel)

    mainFrame = cv.resize(mainFrame, (int(mainFrame.shape[1] / mainFrame.shape[0] * 800), 800))
    cv.imshow("Face smoothing", mainFrame)

    #Zamknięcie po wciśnięciu klawisza ESC
    key = cv.waitKey(20)
    if key == 27:
        break

#Zamknięcie okna
cv.destroyWindow("Face smoothing")
#Zwolnienie dostępu do kamery
feed.release()
