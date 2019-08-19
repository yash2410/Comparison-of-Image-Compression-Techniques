import pywt
import cv2
import math
import os
import numpy as np


def mkdir_os(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def psnr(img1, img2):
    mse = math.sqrt(np.mean((img2 - img1) ** 2))/100
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr_i =20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr_i,mse


def seprate_channel(org_img):
    b_org, r_org, g_org = cv2.split(org_img)
    return [b_org, r_org, g_org]


def compression_ratio(o_img, n_img):
    x_o, y_o, z_o = o_img.shape
    x_n, y_n, z_n = n_img.shape

    o_size = x_o * y_o * z_o * 8
    n_size = x_n * y_n * z_n * 8
    cr = ((o_size - n_size) / o_size) * 100
    return cr


def dwt_(img, comp_type):

    x,y,z = img.shape
    img_seprate = seprate_channel(img)
    new_img = list()
    for i in img_seprate:
        LL, (LH, HL, HH) = pywt.dwt2(i, comp_type, 'smooth')
        LL_threshold = pywt.threshold(LL, 70, "soft", 20)

        idwt_img = pywt.idwt(LL_threshold, HH, comp_type, 'smooth')
        new_img.append(idwt_img)
    merged_img = cv2.merge((new_img[0], new_img[1], new_img[2]))

    merged_img = cv2.resize(merged_img,(x,y))

    avg = np.average(merged_img)
    psnr_i,mse = psnr(img, merged_img)

    print("Average : {}".format(avg))
    print("PSNR : {}".format(psnr_i))
    print("MSE : {}".format(mse))
    cv2.imwrite("CompareImg/" + comp_type + ".jpg", merged_img)

def get_dwt(org_img):
    dwt_comp = ['db4', 'haar', 'bior1.3','coif4','dmey','rbio1.3','sym4']
    for c in dwt_comp:
        print("*" * 30)
        print("Trying {}".format(c))
        dwt_(org_img, c)

def get_dct(img):
    pass

def main():
    mkdir_os("CompareImg")
    org_img = cv2.imread("standard_test_images/lena_color_512.tif")
    cv2.imwrite("lena_color_512.jpg", org_img)

    print("Orignal Shape : {}".format(org_img.shape))
    get_dwt(org_img)
    get_dct(org_img)



main()