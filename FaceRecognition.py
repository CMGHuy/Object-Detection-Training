import PIL
import cv2
import imutils
from PIL import Image
import glob
import numpy as np
import os

# cascade file containing parameters for recognizing face

file_face_cascade_default = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML' \
                            r'\haarcascade_frontalface_default.xml '
file_face_cascade_alt = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML' \
                        r'\haarcascade_frontalface_alt.xml '
file_face_cascade_alt2 = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML' \
                         r'\haarcascade_frontalface_alt2.xml '
file_face_cascade_alt_tree = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML' \
                             r'\haarcascade_frontalface_alt_tree.xml '


def face_detection_with_cascade(image):
    person_available = False

    # create CascadeClassifier Objects
    face_cascade_default = cv2.CascadeClassifier(file_face_cascade_default)
    face_cascade_alt = cv2.CascadeClassifier(file_face_cascade_alt)
    face_cascade_alt2 = cv2.CascadeClassifier(file_face_cascade_alt2)
    face_cascade_alt_tree = cv2.CascadeClassifier(file_face_cascade_alt_tree)

    # reading the image from the computer
    img = cv2.imread(image)
    # img = imutils.resize(img, height=800)

    # reading the image as gray scale image
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = img

    # cascade files, used for classifying the image types. Parameters used to recognize a face
    faces1 = face_cascade_default.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    faces2 = face_cascade_alt.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    faces3 = face_cascade_alt2.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    faces4 = face_cascade_alt_tree.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    # check whether the array of co-ordinates having elements or not.
    if len(faces1) == 0 and len(faces2) == 0 and len(faces3) == 0 and len(faces4) == 0:
        person_available = False
    else:
        person_available = True

    # draw a rectangle box surrounding the detected faces
    for x, y, w, h in faces1:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # display the image with the rectangular view
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(person_available)
    return person_available


def face_detection_with_yolo(image):
    person_available = False

    # Load Yolo
    net = cv2.dnn.readNet(r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\training\yolov3.weights',
                          r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\training\yolov3.cfg')
    classes = []
    with open(r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\training\coco.names', "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading images
    img = cv2.imread(image)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)

    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    outs = net.forward(outputlayers)

    # Showing information on the screen
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes)
    # print(len(boxes))
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'person':
                person_available = True
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return person_available


file_id_card_cascade =  r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML' \
                             r'\id_card_left_face8.xml'
file_id_card_cascade1 =  r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML' \
                             r'\id_card_left_face7.xml'


def id_card_detection_with_cascade(image):
    id_card_available = False

    # create CascadeClassifier Objects
    face_cascade_id_card = cv2.CascadeClassifier(file_id_card_cascade)
    face_cascade_id_card1 = cv2.CascadeClassifier(file_id_card_cascade1)

    # reading the image from the computer
    img = cv2.imread(image)
    # img = imutils.resize(img, height=800)

    # reading the image as gray scale image
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = img

    rejectLevels = []
    levelWeights = []
    detection = []

    # cascade files, used for classifying the image types. Parameters used to recognize a face
    id = face_cascade_id_card.detectMultiScale(gray_img, scaleFactor=1.04, minNeighbors=5, minSize=(50,50)) #400 id face 4
    id1 = face_cascade_id_card1.detectMultiScale(gray_img, scaleFactor=1.04, minNeighbors=1, minSize=(50, 50))  # 400 id face 4

    # check whether the array of co-ordinates having elements or not.
    if len(id) == 0 and len(id1) == 0:
        id_card_available = False
    else:
        id_card_available = True

    # draw a rectangle box surrounding the detected faces
    for x, y, w, h in id:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)


    # display the image with the rectangular view
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(person_available)
    return id_card_available
