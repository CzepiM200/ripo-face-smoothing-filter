#!/usr/bin/python3

import cv2 as cv

#Utworzenie okna
cv.namedWindow("Wygładzanie twarzy")

#Pobranie feedu z kamery
feed =  cv.VideoCapture(0)

if feed.isOpened():
    opened, frame = feed.read()
else:
    opened = False

while opened:
    cv.imshow("Wygładzanie twarzy", frame)
    opened, frame = feed.read()
    key = cv.waitKey(20)
    if key == 27:
        break

cv.destroyWindow("Wygładzanie twarzy")
feed.release()
