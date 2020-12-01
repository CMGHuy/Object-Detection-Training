# Import libraries
import PIL
import fitz
import cv2
import numpy as np
import FaceRecognition as fr
import ImageDetectionTraining as idt
from PIL import Image

# -----------------------------------------
# Initialization
# -----------------------------------------

# Path of the processing pdf
pdf_attachment = fitz.open(r'C:\Users\h.cao\Desktop\Python\Image Processing\venv\Lib\pdf\4.pdf')

# Name of the temporary images file for processing Face Recognition
image_link = r'C:\Users\h.cao\Desktop\Python\Image Processing\venv\Lib\images\1.png'

# List of the remained pages
remained_pages = []

# Set the maximum image pixel to infinity (risk of DoS attack)
# Reference: https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
PIL.Image.MAX_IMAGE_PIXELS = None

# -----------------------------------------
# Converting PDF to images
# -----------------------------------------

# Iterate through all the pages stored above
for page_number in range(pdf_attachment.pageCount):
    page = pdf_attachment.loadPage(page_number)
    page_as_image = page.getPixmap()
    page_as_image.writePNG(image_link)

    # Initiate the face detection process
    # fr.face_detection(image_link)

    # Store the images, which do not have personal photo, in an array
    if not fr.face_detection(image_link):
        remained_pages.append(page)


# im1 = remained_pages.__getitem__(0)
# im1 = np.array(im1)
# cv2.imshow("abc", im1)
# im1.save(r'C:\Users\h.cao\Desktop\Python\Image Processing\venv\Lib\pdf\merged.pdf',save_all=True, append_images=remained_pages)

