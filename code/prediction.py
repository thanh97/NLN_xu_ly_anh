from __future__ import unicode_literals
from keras.models import load_model
import cv2
import numpy as np
from keras.utils.generic_utils import CustomObjectScope
import keras
import os


# xoay anh
def Image_Alignment(img_chuan, img_canxoay):
    # Convert to grayscale.
    img1 = cv2.cvtColor(img_canxoay, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img_chuan, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(7400)

    # (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))

    return (transformed_img)

def row_line(im_th):
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return im_floodfill_inv


def prdiect(img_da_xoay):
    # load model
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        model = load_model("./models/7/weights.02.h5")

    img_resize = cv2.resize(img_da_xoay, (1654, 1166), interpolation=cv2.INTER_AREA)
    img_convert = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # Set values below 128 to 255.
    th, im_th = cv2.threshold(img_convert, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Đảo ngược ảnh
    img_bin = 255 - row_line(im_th)

    # Độ dài của kernel
    kernel_length = np.array(img_bin).shape[1] // 200

    # A verticle kernel of (1 X kernel_0length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    # Tìm các đường thẳng nằm dọc
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)

    # Tìm các đường thẳng nằm ngang
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)

    # Tham số để quyết định số lượng hình ảnh kết hợp tạp thành ô vuông
    alpha = 0.5
    beta = 1 - alpha

    # Kết hợp tạo các ô vuông
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0)
    img_final_bin = cv2.erode(img_final_bin, kernel, iterations=1)

    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # cv2.imshow("anh_1", verticle_kernel)
    # cv2.imshow("anh_2", verticle_lines_img)
    # cv2.imshow("anh_3", horizontal_lines_img)
    # cv2.imshow("anh_4", img_final_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # # cv2.imwrite("img_final_bin.jpg", img_final_bin)
    # #
    # Áp dụng phương pháp tìm biên sẽ phát hiện được các hình vuông và kết hợp lại để tạo hình chữ nhật lớn
    _, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các hình chữ nhật theo thứ tự từ trên xuống
    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = False
        # handle if we are sorting against the y-coordinate rather than

        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to

        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    tmp_anh = []
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        # print(w)
        # print(h)
        # print("het !!!!")
        if (200 < w < 700 and 47 < h < 73 or w > 800 and 47 < h < 73):
            # if (200 < w and  h < 90):
            idx += 1
            new_img = img_resize[y:y + h, x:x + w]
            new_img_convert = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            tmp_anh.append(new_img_convert)

    nguong_pixel = 128
    for h in range(len(tmp_anh)):
        sum_index = 0
        tmp_sum = []
        for i in tmp_anh[h]:
            for j in i:
                if (j < nguong_pixel):
                    sum_index += 1
                else:
                    continue
        tmp_sum.append(sum_index)

        for k in tmp_sum:
            # print(k)
            if (k < 10000):

                img_resize = cv2.resize(tmp_anh[h], (900, 100), interpolation=cv2.INTER_AREA)

                # Threshold.
                th, im_th = cv2.threshold(img_resize, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                image = np.invert(im_th)

                # Invert the image
                img_bin = 255 - image

                # Defining a kernel length
                kernel_length = np.array(img_bin).shape[1] // 150
                # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
                verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
                # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
                hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                # A kernel of (3 X 3) ones.
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

                # Morphological operation to detect vertical lines from an image
                img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
                verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)

                # Morphological operation to detect horizontal lines from an image
                img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
                horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)

                # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
                alpha = 0.5
                beta = 1 - alpha

                # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
                img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0)
                img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)

                (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # cv2.imshow("anh_1",verticle_lines_img)
                # cv2.imshow("anh_2",horizontal_lines_img)
                # cv2.imshow("anh_3",img_final_bin)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

                # Find contours for image, which will detect all the boxes
                _, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Sort all the contours by top to bottom.
                def sort_contours(cnts, method="left-to-right"):
                    # initialize the reverse flag and sort index
                    reverse = False
                    i = 0
                    # handle if we need to sort in reverse
                    if method == "right-to-left" or method == "bottom-to-top":
                        reverse = False
                    # handle if we are sorting against the y-coordinate rather than

                    # the x-coordinate of the bounding box
                    if method == "top-to-bottom" or method == "bottom-to-top":
                        i = 1
                    # construct the list of bounding boxes and sort them from top to

                    # bottom
                    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

                    (cnts, boundingBoxes) = zip(
                        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

                    # return the list of sorted contours and bounding boxes
                    return (cnts, boundingBoxes)

                (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")

                idx = 0
                tmp_ket_qua=[]
                for c in contours:
                    # Returns the location and width, height for every contour
                    x, y, w, h = cv2.boundingRect(c)
                    # print(w)
                    # print(h)
                    # print("het!!!!")
                    if (24 < w < 110 and 66 < h < 90):
                        idx += 1
                        # img_cut = img_resize[y:y+h,x:x+w]
                        new_img = img_resize[y + 4:(y - 1) + h, x + 1:(x - 2) + w]

                        # plt.imshow(new_img)
                        # plt.show()
                        # y+3 dich chuyen len:(y-5) dich chuyen xuong
                        # x-1 di chuyen qua phai0
                        img_color = cv2.cvtColor(new_img, 1)
                        img_resize_lan_1 = cv2.resize(img_color, (128, 128),
                                                      interpolation=cv2.INTER_AREA)  # chuyển kích thước ảnh về 128x128
                        cropped_image = img_resize_lan_1[10:128, 10:128]

                        img_resize_lan_2 = cv2.resize(cropped_image, (128, 128),
                                                      interpolation=cv2.INTER_AREA).flatten()  # chuyển kích thước ảnh về 128x128
                        img = cv2.medianBlur(img_resize_lan_2, 9)

                        thresh = 200
                        # (thresh, img_bin) = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                        img_agray = cv2.threshold(img, thresh, maxval=255, type=cv2.THRESH_BINARY_INV)[1]  # chuyển ảnh về dạng trắng đen
                        (thresh, img_bin) = cv2.threshold(img_agray, 128, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
                        image = np.invert(img_bin)
                        img_color = 255 - image

                        img_color = cv2.cvtColor(img_color, 1)

                        names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
                                 10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
                                 19: "J",
                                 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S",
                                 29: "T", 30: "U",
                                 31: "V", 32: 'W', 33: "X", 34: "Y", 35: "Z"}

                        # dự đoán
                        prediction = model.predict(img_color.reshape(-1, 128, 128, 3))

                        # # # in ra mãng các giá trị dự đoán
                        # print(prediction)
                        #
                        # # lấy phần tử có giá trị lớn nhất
                        predict_img = np.argmax(prediction, axis=-1)

                        tmp_ket_qua.append(names.get(predict_img[0]))

            else:
                continue

    return (tmp_ket_qua)


if __name__ == '__main__':
    # # Open the image files.
    img2_color = cv2.imread("../data/anh_chuan/8.jpg")  # Reference image.
    img1_color = cv2.imread("../data/xu_ly_anh/1.jpg")  # Image to be aligned.

    img_alight = Image_Alignment(img2_color, img1_color)

    tmp_ket_qua=prdiect(img_alight)
    # for ls in tmp_ket_qua:
    #     print(ls)
