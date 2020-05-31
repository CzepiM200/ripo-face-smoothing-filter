#!/usr/bin/python3

import cv2 as cv
from imutils import face_utils
import dlib

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-epredictor", required=True,
# 	help="./ibug_300W_large_face_landmark_dataset/eye_predictor.dat")
# ap.add_argument("-p", "--shape-mpredictor", required=True,
# 	help="./ibug_300W_large_face_landmark_dataset/mouth_predictor.dat")
# args = vars(ap.parse_args())

#Detekcja twarzy
detector = dlib.get_frontal_face_detector()
#Wczytywanie prekyktorów oczy i ust
eyesPredictor = dlib.shape_predictor('./eye_predictor.dat')
mouthPredictor = dlib.shape_predictor('./mouth_predictor.dat')

#Zmienna określająca stan włączenia filtru
filter_enabled = 0

#Zmienna określająca stan włączenia elementów pomocniczych zaznaczających wykryte elementy twarzy
overlay_enabled  = 1

#Funkcja przełączająca filtr
def toggle_filter(x):
    global filter_enabled
    filter_enabled = x

def toggle_overlay(x):
    global overlay_enabled 
    overlay_enabled = x

#Utworzenie okna
cv.namedWindow("Face smoothing")

#Pobranie feedu z kamery
feed = cv.VideoCapture(0)

#Przełącznik filtru
cv.createTrackbar("Filter", "Face smoothing", 0, 1, toggle_filter)
cv.createTrackbar("Overlay", "Face smoothing", 1, 1, toggle_overlay)

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
    
    original = frame.copy()

    #Detekcja twarzy
    rects = detector(gray, 0)

    #Dla każdej z wykrytych twarzy
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        #Detekcja ust
        mouths = mouthPredictor(gray, rect)
        mouths = face_utils.shape_to_np(mouths)
        #Detekcja oczu
        eyes = eyesPredictor(gray, rect)
        eyes = face_utils.shape_to_np(eyes)
		
        if overlay_enabled == 1:
            #Rysowane prostokąta twarzy
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #Rysowanie punktów oka
            for (sX, sY) in eyes:
                cv.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
            #Rysowanie punktów ust
            for (mX, mY) in mouths:
                cv.circle(frame, (mX, mY), 1, (255, 0, 0), -1)
        
        #Zapisanie obszarów zaznaczających wykryte predyktorem oczy i usta
        leftEyeImage = frame[eyes[2,1]:eyes[5,1], eyes[0,0]:eyes[3,0]].copy()
        rightEyeImage = frame[eyes[7,1]:eyes[10,1], eyes[6,0]:eyes[9,0]].copy()
        mouthImage = frame[mouths[3,1]:mouths[9,1], mouths[1,0]:mouths[7,0]].copy()   
                
        if filter_enabled == 1:
            # X lewy i prawy print(mouths[0,0], mouths[8,0]);
            # Y górny dolny print(mouths[0,1],mouths[6,1])

            #Filtr całej twarzy
            faceImage = frame[y:y+h, x:x+w].copy()
            faceImage = cv.medianBlur(faceImage,9)
            frame[y:y+h, x:x+w] = faceImage.copy()

            frame[mouths[3,1]:mouths[9,1], mouths[1,0]:mouths[7,0]] = mouthImage.copy()
            frame[eyes[2,1]:eyes[5,1], eyes[0,0]:eyes[3,0]] = leftEyeImage.copy()
            frame[eyes[7,1]:eyes[10,1], eyes[6,0]:eyes[9,0]] = rightEyeImage.copy()


    cv.imshow("Face smoothing", frame)

    #Zamknięcie po wciśnięciu klawisza ESC
    key = cv.waitKey(10)
    if key == 27:
        break

#Zamknięcie okna
cv.destroyWindow("Face smoothing")

#Zwolnienie dostępu do kamery
feed.release()
