import cv2
import math
import time
import numpy as np
# To import the library of hand detection
from cvzone.HandTrackingModule import HandDetector

# To open the webcam
cap = cv2.VideoCapture(0)
# To read the shape of a single hand when ( maxHands=1 ) and for the two hands when ( maxHands=2 )
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# The folder where we will save the data which we collect
folder = "data/H"
counter = 0

while True:
    # To read the video from the webcam
    success, img = cap.read()
    # To detect the hand and show the detection on the screen
    hands, img = detector.findHands(img)

    # To crop the image which we take from the webcam
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # To create a cropped image of the hand detector
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        # To create white area to show the cropped image on it
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCropShape = imgCrop.shape

        # The ratio of the size of the image
        aspectRatio = h/w

        # To make the size white area change with the size of cropped image
        if aspectRatio > 1:
            # To stretch the height of the white area
            k = imgSize/h
            wCal = math.ceil(k*w)

            # To resize the size of white area
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            # To show the cropped image on the white area
            imgResizeShape = imgResize.shape
            # To center the image of resized image on the white area
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            # To stretch the width of the white area
            k = imgSize / w
            hCal = math.ceil(k * h)

            # To resize the size of white area
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            # To show the cropped image on the white area
            imgResizeShape = imgResize.shape
            # To center the image of resized image on the white area
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize



        cv2.imshow("Image Croped", imgCrop)
        cv2.imshow("Image White", imgWhite)

    # To show the result on the screen
    cv2.imshow("Image", img)

    # To select a button which do an event when the user press on it
    key = cv2.waitKey(1)

    # To collect and save the images from the user when he presses on ( s ) button
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

    # To close the screen when the user press on ( Esc ) button
    if key & 0xFF == 27:
        break

# To close the webcam after pressing on the Esc button
cap.release()
cv2.destroyAllWindows()
