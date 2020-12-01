import numpy as np
import cv2


# def apply_invert(frame):
#     return cv2.bitwise_not(frame)
#
#
# def verify_alpha_channel(frame):
#     try:
#         frame.shape[3]  # 4th position
#     except IndexError:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
#     return frame


def face_detection_using_webcam():
    face_cascade = cv2.CascadeClassifier(
        r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML\idcard_cascade7.xml')

    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        # Capture frame-by-frame
        ret, img = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # invert = apply_invert(img)
        # sepia1 = apply_sepia(img.copy())

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

        # cv2.putText(img, 'ID card', (x - w, y - h), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


face_detection_using_webcam()
