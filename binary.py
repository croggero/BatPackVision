import cv2
import numpy as np
from matplotlib import pyplot as plt


def pipeline(img, s_thresh=(0, 60), sx_thresh=(10, 100), r_thresh=(100, 255), sobel_kernel=3):
    dst = np.copy(img)

    # dst = cv2.pyrMeanShiftFiltering(dst, 20, 45, 3)
    # cv2.imshow("Test", dst)
    dst = cv2.GaussianBlur(dst, (15, 15), 0)

    cv2.imshow('frame', dst)
    r_channel = dst[:, :, 0]
    g_channel = dst[:, :, 1]
    b_channel = dst[:, :, 2]

    hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 255

    ret, otsu = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('frame2', otsu)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

    test = np.zeros_like(s_binary)
    test[:,:] = 255
    if(np.array_equal(s_binary, test)):
        return np.zeros_like(dst)

    combined = np.zeros_like(s_binary)
    combined[(s_binary == 255) | (otsu == 255)] = 255

    combined = cv2.bitwise_not(otsu)

    contour_image, contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    binary_contour_image = np.zeros_like(contour_image)
    cv2.drawContours(binary_contour_image, [c], -1, (255, 255, 255), 2)

    return binary_contour_image

def select_white_yellow(image):
    # white color mask
    lower = np.uint8([  0, 50, 70])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([ 50, 200, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

cap = cv2.VideoCapture('Sequence3.mp4')

# NOTE: this function expects color images
ret, frame = cap.read()
while ret:
    # frame = cv2.imread('153.jpg')
    ret, frame = cap.read()
    cv2.imshow('frame1', frame)
    image = pipeline(frame)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()