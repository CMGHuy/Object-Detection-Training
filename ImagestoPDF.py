# Reference: https://datatofish.com/images-to-pdf-python/
from PIL import Image

image1 = Image.open(r'C:\Users\h.cao\Desktop\Python\Image Processing\page_1.jpg')
image2 = Image.open(r'C:\Users\h.cao\Desktop\Python\Image Processing\page_2.jpg')
image3 = Image.open(r'C:\Users\h.cao\Desktop\Python\Image Processing\page_3.jpg')
image4 = Image.open(r'C:\Users\h.cao\Desktop\Python\Image Processing\page_4.jpg')

im1 = image1.convert('RGB')
im2 = image2.convert('RGB')
im3 = image3.convert('RGB')
im4 = image4.convert('RGB')

imagelist = [im2,im3,im4]

im1.save(r'C:\Users\h.cao\Desktop\Python\Image Processing\10.pdf',save_all=True, append_images=imagelist)