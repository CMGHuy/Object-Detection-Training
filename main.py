from PIL import Image
import PIL
import cv2
import imutils
import glob
import numpy as np
import os
import timeit
import FaceRecognition as fr

# starting time for running process
start = timeit.default_timer()

# initialize variables
PIL.Image.MAX_IMAGE_PIXELS = None
# arrays contains images detecting or not detecting face
image_detected = []
image_non_detected = []
# counting variables how many images of each type have been processed
detected = 0
not_detected = 0
count = 1

# define file directory
# the temporary jpeg file to save image for analyzing
temp_image = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\Test pictures\1.jpeg'

# folder contains all the processing data, saves the detected images and not detected images
folder_processing = glob.glob(r"C:\Users\h.cao\Desktop\Python\ImageRecognition\Test pictures\*.png")
folder_detected = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\Test pictures\Detected'
folder_not_detected = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\Test pictures\Non Detected'

file_id_card_cascade =  r'C:\Users\h.cao\Desktop\Python\ImageRecognition\venv\Lib\cascade-XML' \
                             r'\id_card_left_face8.xml'
face_cascade_id_card = cv2.CascadeClassifier(file_id_card_cascade)


# delete all files in detected and non-detected folder
def removeFile():
    file_detected_folder = glob.glob(os.path.join(folder_detected, "*.png"))
    file_non_detected_folder = glob.glob(os.path.join(folder_not_detected, "*.png"))
    for f in file_detected_folder:
        os.remove(f)

    for f in file_non_detected_folder:
        os.remove(f)


# # scan each image and apply the face recognition function to each of them
# for file_directory in folder_processing:
#     # print the order of the processing image
#     file_name = os.path.basename(file_directory)
#     print("Processing on element", count, ", file name:", file_name, ", among", len(folder_processing), "images")
#     if fr.face_detection_with_cascade(file_directory):
#         file_new_path = os.path.join(folder_detected, file_name)
#         im = Image.open(file_directory)
#         im.save(file_new_path)
#         detected += 1
#         image_detected.append(file_directory)
#         print("There is a personal image")
#     else:
#         im = Image.open(file_directory)
#         if fr.face_detection_with_yolo(file_directory):
#             detected += 1
#             found = True
#             file_new_path = os.path.join(folder_detected, file_name)
#             im.save(file_new_path)
#             image_detected.append(file_directory)
#             print("There is a personal image")
#         else:
#             not_detected += 1
#             file_new_path = os.path.join(folder_not_detected, file_name)
#             im.save(file_new_path)
#             image_non_detected.append(file_directory)
#             print("There is no personal image")
#     count += 1


# scan each image and apply the id card recognition function to each of them
# removeFile()
# for file_directory in folder_processing:
#     # print the order of the processing image
#     file_name = os.path.basename(file_directory)
#     print("Processing on element", count, ", file name:", file_name, ", among", len(folder_processing), "images")
#     if fr.id_card_detection_with_cascade(file_directory):
#         file_new_path = os.path.join(folder_detected, file_name)
#         im = Image.open(file_directory)
#         im.save(file_new_path)
#         detected += 1
#         image_detected.append(file_directory)
#         print("There is a personal image")
#     else:
#         im = Image.open(file_directory)
#         not_detected += 1
#         file_new_path = os.path.join(folder_not_detected, file_name)
#         im.save(file_new_path)
#         image_non_detected.append(file_directory)
#         print("There is no personal image")
#     count += 1


# scan each image and apply the id card recognition function to each of them
# and draw a rectangle on detecting images
removeFile()
for file_directory in folder_processing:
    # print the order of the processing image
    file_name = os.path.basename(file_directory)
    print("Processing on element", count, ", file name:", file_name, ", among", len(folder_processing), "images")
    if fr.id_card_detection_with_cascade(file_directory):
        file_new_path = os.path.join(folder_detected, file_name)

        img = cv2.imread(file_directory)

        # reading the image as gray scale image
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = img

        # cascade files, used for classifying the image types. Parameters used to recognize a face
        id = face_cascade_id_card.detectMultiScale(gray_img, scaleFactor=1.04, minNeighbors=5, minSize=(50,50)) #1.3, 5, minSize(50,50)  #1.04, 400, (50,50)

        # draw a rectangle box surrounding the detected faces
        for x, y, w, h in id:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.imwrite(file_new_path, img)
        detected += 1
        image_detected.append(file_directory)
        print("There is a personal image")
    else:
        im = Image.open(file_directory)
        not_detected += 1
        file_new_path = os.path.join(folder_not_detected, file_name)
        im.save(file_new_path)
        image_non_detected.append(file_directory)
        print("There is no personal image")
    count += 1

# stopping time of running process
stop = timeit.default_timer()

# print the summary of the process
print('\n************************************\n'
      'Summary: \n',
      detected, 'images detect person.\n',
      not_detected, 'images do not detect person.\n'
      'In total there are', detected + not_detected, 'images\n'
      'Processing time: ', stop - start,
      '\n************************************')