import cv2
import math
import tensorflow
from cvzone.ClassificationModule import Classifier
import numpy as np
# To import the library of hand detection
from cvzone.HandTrackingModule import HandDetector

# To open the webcam
cap = cv2.VideoCapture(0)
# To read the shape of a single hand when ( maxHands=1 ) and for the two hands when ( maxHands=2 )
detector = HandDetector(maxHands=1)
# To link the program with the model of sign language photos
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

# The folder where we will save the data which we collect
folder = "data/A"
counter = 0

labels = ["A", "B", "D", "E", "L", "R"]

while True:
    # To read the video from the webcam
    success, img = cap.read()

    # To make a copy of the original image
    imgOutput = img.copy()

    # To detect the hand and show the detection on the screen
    hands = detector.findHands(img, draw=False)

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

            # To out or print the character and it's index
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

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

            # To out or print the character and it's index
            prediction, index = classifier.getPrediction(imgWhite)


        # To print the characters in filled rectangle and the hand in another rectangle
        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x - offset + 90, y - offset - 50 + 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)


        cv2.imshow("Image Croped", imgCrop)
        cv2.imshow("Image White", imgWhite)

    # To show the result on the screen
    cv2.imshow("Image", imgOutput)

    # To select a button which do an event when the user press on it
    key = cv2.waitKey(1)

    # To close the screen when the user press on ( Esc ) button
    if key & 0xFF == 27:
        break

# To close the webcam after pressing on the Esc button
cap.release()
cv2.destroyAllWindows()
