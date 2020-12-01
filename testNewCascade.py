import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

file_face_cascade_test = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML\\idcard_cascade9.xml'


def face_detection_with_cascade(image):
    person_available = False

    face_cascade_test = cv2.CascadeClassifier(file_face_cascade_test)

    # reading the image from the computer
    img = cv2.imread(image)
    # img = cv2.resize(img, (500, 700))

    # reading the image as gray scale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.Canny(img, 100, 200)

    # cascade files, used for classifying the image types. Parameters used to recognize a face
    face = face_cascade_test.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check whether the array of co-ordinates having elements or not.
    if len(face) == 0:
        person_available = False
    else:
        person_available = True

    # draw a rectangle box surrounding the detected faces
    for x, y, w, h in face:
        gray_img = cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(img, 'ID card', (x - w, y - h), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)

    # display the image with the rectangular view
    cv2.imshow("Image", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(person_available)
    return person_available


def test():
    img = cv2.imread(r'C:\Users\h.cao\Desktop\Python\ImageRecognition\Test pictures\30056163_55115266.pdf_5.png', cv2.IMREAD_GRAYSCALE)
    _, threshold = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)

    cv2.imshow('threshold', threshold)
    cv2.imshow('image', img)
    cv2.waitKey(0)


test()
# face_detection_with_cascade(r'C:\Users\h.cao\Desktop\Python\ImageRecognition\Test pictures\cropped.png')