Reference: https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/

Steps:

-- Create negative files
Run function create_pos_n_neg() in file storeImages

-- Open ubuntu for Windows 
cd /mnt/c/Users/h.cao/Desktop/Python/ImageRecognition/

-- Change bg.txt to unix type
dos2unix bg.txt

-- Resize positive images
Run function resize_positive_images() in file storeImages

-- Create positive samples
opencv_createsamples -img pos/1.png -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1950

-- Create runCreateSamples file
Run function create_runCreateSamples_content in file storeImages

-- Run script runCreateSamples to create multiple samples based on multiple positive images:
bash script.txt

-- Run script combineAllInfoFile to combine all the created info files. Remember to delete the last row of the merged files
copy *.lst info.lst

cd info
cat *.lst > info.lst


-- Create positive vectors
cd ..
opencv_createsamples -info info/info.lst -num 8000 -w 25 -h 25 -vec positives.vec

-- Train cascade file
opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 8000 -numNeg 4000 -numStages 15 -w 25 -h 25