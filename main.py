import cv2

def displayImg(img):
    cv2.imshow("Test Display", img)
    cv2.waitKey(0)

#OpenCV Version : opencv_python-4.5.4.58
image = cv2.imread("images/fourn.png", cv2.IMREAD_GRAYSCALE)
displayImg(image)

#1. Filtrage Gaussien (en cas de bruit ou de d ́etails très fins)
blur_x = 3
blur_y = 3
blurredImg = cv2.blur(image, (blur_x, blur_y))
displayImg(blurredImg)