#importing libraries
#to run this file execute following command
#python NumberPlateRecognition.py --image input/<imgname.jpg> --output output/<imgname.jpg>
import numpy as np
import cv2
import  imutils
import sys
import pytesseract
import pandas as pd
import time
import argparse

pytesseract.pytesseract.tesseract_cmd = r"C:\users\dhruv\appdata\local\tesseract.exe"

#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input car image")
ap.add_argument("-o", "--output", required=True, help="path to output car image")
args = vars(ap.parse_args())

print("[Info] loading image...")
image = cv2.imread(args["image"])
cv2.imshow("Original Image", image)

print("[Info] preprocess image...")
image = imutils.resize(image, width=800)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Conversion", gray)
#cv2.cvtColor() method is used to convert an image from one color space to another.
#There are more than 150 color-space conversion methods available in OpenCV
#here we are converting from BGR format to gray scale format

gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral Filter", gray)
#A bilateral filter is used for smoothening images and reducing noise, while preserving 
#edges.However, these convolutions often result in a loss of important edge information, 
#since they blur out everything, irrespective of it being noise or an edge. 
#To counter this problem, the non-linear bilateral filter was introduced.

edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Canny Edges", edged)
#OpenCV has in-built function cv2.Canny() which takes our input image as first argument 
#and its aperture size(min value and max value) as last two arguments. This is a 
#simple example of how to detect edges in Python.

cv2.waitKey(0)
cv2.destroyAllWindows()

print("[Info] finding contours...")
(cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#The cv2.findContours method is destructive (meaning it manipulates the image you pass in) 
#so if you plan on using that image again later, be sure to clone it.

#cv2.RETR_LIST retrieves all of the contours without establishing any 
#hierarchical relationships.

#cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and 
#leaves only their end points. For example, an up-right rectangular contour is encoded 
#with 4 points.

#Contours are defined as the line joining all the points along the boundary of an image 
#that are having the same intensity. Contours come handy in shape analysis, finding the 
#size of the object of interest, and object detection. OpenCV has findContour() function 
#that helps in extracting the contours from the image. 

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:20]
NumberPlateCnt = None 

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        #cv2.arcLength() Calculates a contour perimeter or a curve length.
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #cv2.approxPolyDP(): This is used to obtain the number of points found in the 
        #figure. epsilon is an accuracy parameter. A wise selection of epsilon is needed 
        #to get the correct output.
        #closed – If true, the approximated curve is closed (its first and last 
        #vertices are connected). Otherwise, it is not closed.
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break

#Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt], 0, 255, -1)
cv2.imshow("Contours", new_image)

#To draw the contours, cv2.drawContours function is used. It can also be used to draw 
#any shape provided you have its boundary points. Its first argument is source image, 
#second argument is the contours which should be passed as a Python list, third argument 
#is index of contours (useful when drawing individual contour. To draw all contours, pass 
#-1) and remaining arguments are color, thickness etc.

new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.imshow("masking",new_image)

cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)

print("[Info] printing number plate characters...")

#Configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

#Run tesseract OCR on image
text = pytesseract.image_to_string(new_image, config=config)

#Data is stored in CSV file
raw_data = {'date':[time.asctime( time.localtime(time.time()))],'':[text]}
#raw_data = [time.asctime( time.localtime(time.time()))],[text]   

df = pd.DataFrame(raw_data)
df.to_csv('data.csv',mode='a')

#Print recognized text
print(text)
cv2.putText(new_image,text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,2)
cv2.imshow("Final_image",new_image)
cv2.imwrite(args["output"], new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()