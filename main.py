import matplotlib.pyplot as plt
import numpy as np
import cv2

def displayImg(img):
    cv2.imshow("Test Display", img)
    cv2.waitKey(0)

#OpenCV Version : opencv_python-4.5.4.58
image = cv2.imread("images/fourn.png", cv2.IMREAD_GRAYSCALE)
#displayImg(image)

#1.  Filtrage Gaussien (en cas de bruit ou de d ́etails très fins)
blur_x = 5
blur_y = 5
blurredImg = cv2.blur(image, (blur_x, blur_y))
#displayImg(blurredImg)

#2.  Filtrage de Sobel, calcul de la magnitude de gradient Imag dans chaque
#pixel
#TODO
#sobel_x = cv2.Sobel(blurredImg, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)

sobelX = cv2.Sobel(blurredImg, cv2.CV_64F, 1, 0, ksize=5)  # x
#displayImg(sobelX)
sobelY = cv2.Sobel(blurredImg, cv2.CV_64F, 0, 1, ksize=5)  # y
#displayImg(sobelY)

magnitude = np.sqrt((sobelX ** 2) + (sobelY ** 2))
magnitude = magnitude / np.max(magnitude) 
displayImg(magnitude)

#plt.subplot(2,2,3),plt.imshow(sobelX, cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobelY, cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,1),plt.imshow(magnitude, cmap = 'gray')
#plt.title('Magnitude'), plt.xticks([]), plt.yticks([])

plt.show()

# Question 3

# Fraction t
t = 800

# Pour copier les valeurs > t (seuil) dans une matrice de copie
matriceCopieSuperieur = np.ones_like(magnitude)

s = 0.0
for i in range(0, magnitude.shape[0]):
    for j in range(0, magnitude.shape[1]):
        #print(magnitude[i, j])
        if magnitude[i, j] > t:
            magnitude[i, j] = 255
            matriceCopieSuperieur[i, j] = magnitude[i, j]
        else:
            magnitude[i, j] = 0
        s = s + magnitude[i, j]
    #print("Somme = ", s)

plt.subplot(2,2,1),plt.imshow(magnitude, cmap = 'gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])

# Une autre façon d'afficher :
#plt.imshow(magnitude)

plt.show()



### Mettre dans le comtpe-rendu qu'on a dut modifier le filtre gaussien car il y avait trop de bruit (3, 3) -> (5, 5)

       

