import cv2
import numpy as np
import time
import os
import handTracking as htm

overlayList = [cv2.imread('purple.jpg'),cv2.imread('blue.jpg'),cv2.imread('green.jpg'),cv2.imread('eraser.jpg')]

for i in range(0,4):
    overlayList[i] = cv2.resize(overlayList[i],(1280,125),interpolation=cv2.INTER_CUBIC)

# cv2.imshow('img',overlayList[3])
# cv2.waitKey(0)
header = overlayList[0]
draw = (255,0,255)
brushsize = 15


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xp,yp = 0,0

imgCanvas = np.zeros((720,1280,3),dtype='uint8')

while True:
    success , img = cap.read()
    img = cv2.flip(img,1)


    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            # print('Selection Mode')
            (xp,yp) = (0,0)
            if y1 <125:
                if 250<x1<450:
                    header = overlayList[0]
                    draw = [255,0,255]
                elif 550<x1<750:
                    header = overlayList[1]
                    draw = [255,0,0]
                elif 800<x1<950:
                    header = overlayList[2]
                    draw = [0,255,0]
                elif 1050<x1<1280:
                    header = overlayList[3]
                    draw = [0,0,0]
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), draw, cv2.FILLED)

        elif fingers[1]:
            cv2.circle(img,(x1,y1),30,draw,cv2.FILLED)
            # print('Drawing Mode')
            if (xp,yp) == (0,0):
                xp,yp = x1,y1
            if draw == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), draw, thickness=100)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), draw, thickness=100)
            else:
                cv2.line(img,(xp,yp),(x1,y1),draw,thickness=brushsize)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), draw, thickness=brushsize)
            (xp,yp) = (x1,y1)

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    img[0:125,0:1280] = header

    cv2.imshow('Image',img)
    cv2.imshow('Canvas',imgCanvas)
    cv2.waitKey(1)