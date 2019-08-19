import cv2
import pywt
import os
import math
import numpy as np


def mkdir_os(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def dwt_(org_gray, comp_type):
    LL, (LH, HL, HH) = pywt.dwt2(org_gray, comp_type, 'smooth')
    LL_threshold = pywt.threshold(LL,60,"soft",10)

    idwt_img = pywt.idwt(LL_threshold,HH,comp_type,'smooth')
    idwt_img = cv2.resize(idwt_img, (512, 512))

    avg = np.average(idwt_img)
    psnr_i = psnr(org_gray,idwt_img)

    print("{} Average : {}".format(comp_type, avg))
    print("{} PSNR : {}".format(comp_type, psnr_i))
    print("{} Shape : {}".format(comp_type, idwt_img.shape))
    cv2.imwrite("GenImg/" +comp_type + ".jpg", idwt_img)



mkdir_os("GenImg")
org_img = cv2.imread("standard_test_images/cameraman.tif")
org_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_cameraman.jpg", org_gray)

print("Original Image type : {}".format(type(org_gray)))
print("Orignal Shape : {}".format(org_gray.shape))

dwt_comp = ['db4', 'haar', 'bior1.3']
for c in dwt_comp:
    print("*" * 30)
    print("Trying {}".format(c))
    dwt_(org_gray, c)



# Caluclate image quality
