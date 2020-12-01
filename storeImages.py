import urllib.request
import cv2
import os
import glob
from PIL import Image
import sys


# Step 1
# Download negative images from image-net.org
def store_neg_raw_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01317541'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 5457

    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/" + str(pic_num) + ".jpg")
            img = cv2.imread("neg/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            # resize the downloaded image to the wanted size
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/" + str(pic_num) + ".jpg", resized_image)
            # cv2.imwrite("neg/" + str(pic_num) + ".jpg", img)
            pic_num += 1

        except Exception as e:
            print(str(e))


# Step 2
# Create the txt file having the reference to all the negative images
def create_pos_n_neg():
    for file_type in ['neg']:

        for img in os.listdir(file_type):
            if file_type == 'neg':
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)

            elif file_type == 'pos':
                line = file_type + '/' + img + ' 1 0 0 50 50\n'
                with open('info.dat', 'a') as f:
                    f.write(line)


# # Change color images to gray scale images. In case of using images from other sources
# def change_raw_to_gray_neg_images():
#     pic_num = 1
#     # Folder containing color negative images
#     folder_processing = glob.glob(r"C:\Users\h.cao\Desktop\Python\ImageRecognition\draft\*.png")
#     # Folder containing gray scale negative images
#     folder_result = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\neg'
#
#     if not os.path.exists('draft'):
#         os.makedirs('draft')
#
#     for file_directory in folder_processing:
#         # print the order of the processing image
#         file_name = os.path.basename(file_directory)
#         file_new_path = os.path.join(folder_result, file_name)
#         img = cv2.imread(file_directory)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (600, 850))
#         # cv2.imshow("Image", img)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         cv2.imwrite(os.path.join(folder_result, file_name), img)


# Step 3
# Resize the raw positive images to the required size
def resize_positive_images():
    # all types of objects
    classified_folder = ['american_passport', 'back_side_aufenhaltstitel', 'bank_card', 'id_card_left_and_right_faces',
                         'id_card_left_face', 'id_card_left_multiple_faces', 'id_card_right_face', 'other', 'passport',
                         'passport_address', 'all']

    unresized_images_location = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\pos_draft'
    resized_images_location = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\pos'

    for folder in classified_folder:

        unresized_folder = unresized_images_location + '\\' + folder + '\*.png'
        resized_folder = resized_images_location + '\\' + folder

        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)

        folder_processing = glob.glob(unresized_folder)
        folder_result = resized_folder

        for file_directory in folder_processing:
            # print the order of the processing image
            file_name = os.path.basename(file_directory)
            file_new_path = os.path.join(folder_result, file_name)
            img = cv2.imread(file_directory)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (350, 215)) #width x height
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(os.path.join(folder_result, file_name), img)

        files = os.listdir(resized_folder)

        # Rename all files in folder in incrementing order
        for index, file in enumerate(files):
            os.rename(os.path.join(resized_folder, file), os.path.join(resized_folder, 'a'.join([str(index), '.png'])))

        files = os.listdir(resized_folder)
        for index, file in enumerate(files):
            os.rename(os.path.join(resized_folder, file), os.path.join(resized_folder, ''.join([str(index), '.png'])))


# Step 4
# Create positive samples scripts
def create_runCreateSamples_content():
    start_num = 0
    end_num = 1000  # number of positive images = the largest number of the image name + 1
    process_num = 50  # number of images of each runCreateSamples file contains/number of lines in one runCreateSamples file
    num_file = round(end_num // process_num)  # number of runCreateSamples created
    num_pos_samples = 30  # number of negative images that 1 positive image want to superposition on.
    folder_location = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\runCreateSample'

    for num_createSamples in range(0, num_file):
        f = open(folder_location + "\createSamples" + str(num_createSamples) + ".txt", "w+")
        for i in range(start_num, start_num + process_num):
            # change the wanted folder in here
            f.write('opencv_createsamples -img pos/all/' + str(i) + '.png -bg bg.txt -info info/info' + str(i) +
                    '.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num ' + str(num_pos_samples) +
                    '\n')
        start_num += process_num

    f = open(folder_location + "\script.txt", "w+")
    for i in range(0, num_file-1):
        f.write('bash createSamples' + str(i) + '.txt &&\n')

    f.write('bash createSamples' + str(num_file-1) + '.txt')


def change_name_in_order_increment():
    resized_folder = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\pos\test'

    files = os.listdir(resized_folder)
    # Rename all files in folder in incrementing order
    for index, file in enumerate(files):
        os.rename(os.path.join(resized_folder, file), os.path.join(resized_folder, 'a'.join([str(index), '.png'])))

    files = os.listdir(resized_folder)
    for index, file in enumerate(files):
        os.rename(os.path.join(resized_folder, file), os.path.join(resized_folder, ''.join([str(index), '.png'])))


def resize_negative_images():
    # all types of objects

    unresized_folder = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\neg_draft\*.png'
    resized_folder = r'C:\Users\h.cao\Desktop\Python\ImageRecognition\neg'

    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)

    folder_processing = glob.glob(unresized_folder)
    folder_result = resized_folder

    for file_directory in folder_processing:
        # print the order of the processing image
        file_name = os.path.basename(file_directory)
        file_new_path = os.path.join(folder_result, file_name)
        img = cv2.imread(file_directory)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (595, 841)) #100 100
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(folder_result, file_name), img)


create_runCreateSamples_content()