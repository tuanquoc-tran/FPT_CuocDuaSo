import cv2 
import matplotlib.pyplot as plt
import imutils
import numpy as np

imgDepth =cv2.imread("E:/TaiLieuChuyenNganh/Code_CuocDuaSo/DataCDS/depth/432.jpg")
imgRGB =cv2.imread("E:/TaiLieuChuyenNganh/Code_CuocDuaSo/DataCDS/rgb/432.jpg")

plt.subplot(2,2,1)
plt.imshow(imgDepth)
plt.title("depth")

plt.subplot(2,2,2)
plt.title("rgb")
plt.imshow(imutils.opencv2matplotlib(imgRGB))

height = imgDepth.shape[0]
width = imgDepth.shape[1]

hog = cv2.HOGDescriptor()
rect = hog.detect(imgDepth,Size(3,3),Size(3,3),(10,10))
print(rect)
plt.subplot(2,2,3)

plt.show()
