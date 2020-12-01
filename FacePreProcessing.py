import cv2
import numpy as np

image_file = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\Test pictures\32968037_54997093.pdf_4.png'
image = cv2.imread(image_file)

cap = cv2.VideoCapture(0)


def verify_alpla_channel(frame):
    try:
        frame.shape[3]
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame


def apply_sepia(frame, intensity=0.5):
    frame = verify_alpla_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    blue = 20
    green = 66
    red = 112
    sepia_bgra = (blue, green, red, 1)
    overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


while True:
    ret, frame1 = cap.read()

    sepia = apply_sepia(frame1.copy())
    cv2.imshow('sepia', sepia)

    cv2.imshow('frame', frame1)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
