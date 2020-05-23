#!/usr/bin/python3

import cv2 as cv
#Import funkcji wł/wył filtr
from smoothing_filter import toggle

#Utworzenie okna
cv.namedWindow("Face smoothing")

#Pobranie feedu z kamery
feed = cv.VideoCapture(0)

#Załadowaine klasyfikatora
face_detection = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

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
    faces = face_detection.detectMultiScale(gray, minNeighbors=6, minSize=(100, 100))
    for (x, y, width, height) in faces:
        cv.rectangle(frame, (x, y), (x + width, y + width), (0xDD, 0x77, 0xAA), 2)

    #Wyświetlenie obrazu w oknie o wysokości 800px
    frame = cv.resize(frame, (int(frame.shape[1] / frame.shape[0] * 800), 800))
    cv.imshow("Face smoothing", frame)
    #Zamknięcie po wciśnięciu klawisza ESC
    key = cv.waitKey(20)
    if key == 27:
        break

#Zamknięcie okna
cv.destroyWindow("Face smoothing")
#Zwolnienie dostępu do kamery
feed.release()
