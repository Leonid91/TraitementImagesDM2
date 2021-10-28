import matplotlib.pyplot as plt # A commenter si utilisation seulement de displayImg() définie ci-dessous.
import numpy as np
import math
import cv2

def displayImg(img):
    cv2.imshow("Test Display", img)
    cv2.waitKey(0)

#OpenCV Version : opencv_python-4.5.4.58
image = cv2.imread("images/fourn.png", cv2.IMREAD_GRAYSCALE)
#displayImg(image)

#1. Filtrage Gaussien (en cas de bruit ou de d ́etails très fins)
blur_x = 5
blur_y = 5
blurredImg = cv2.GaussianBlur(image, (blur_x, blur_y), 0)
#displayImg(blurredImg)

#2. Filtrage de Sobel, calcul de la magnitude de gradient Imag dans chaque pixel
sobelX = cv2.Sobel(blurredImg, cv2.CV_64F, 1, 0, ksize=5)  # X
#displayImg(sobelX)
sobelY = cv2.Sobel(blurredImg, cv2.CV_64F, 0, 1, ksize=5)  # Y
#displayImg(sobelY)

magnitude = np.sqrt((sobelX ** 2) + (sobelY ** 2))
magnitude = magnitude / np.max(magnitude) # Pour ramener les valeurs des pixels dans l'intervalle [0, 1], pour que celà soit utilisable avec displayImg
#displayImg(magnitude)

# Une autre façon d'afficher les images : (à paraméter pour l'intervalle [0, 1]

#plt.subplot(2,2,3),plt.imshow(sobelX, cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobelY, cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,1),plt.imshow(magnitude, cmap = 'gray')
#plt.title('Magnitude'), plt.xticks([]), plt.yticks([])

#plt.show()

#3. Mise en contraste des contours

# Fraction t
t = 0.26

# Pour copier les valeurs > t (seuil) dans une matrice de copie
matriceCopieSuperieur = np.ones_like(magnitude)

#s = 0.0
for i in range(0, magnitude.shape[0]):
    for j in range(0, magnitude.shape[1]):
        if magnitude[i, j] > t:
            magnitude[i, j] = 255
            matriceCopieSuperieur[i, j] = magnitude[i, j]
        else:
            magnitude[i, j] = 0
        #s = s + magnitude[i, j]

#displayImg(magnitude)

# Une autre façon d'afficher :
#plt.subplot(2,2,1),plt.imshow(magnitude, cmap = 'gray')
#plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
# Ou
#plt.imshow(magnitude)

#plt.show()

### Mettre dans le comtpe-rendu qu'on a dut modifier le filtre gaussien car il y avait trop de bruit (3, 3) -> (5, 5)


#4. Initialisation de toutes les valeurs de l'accumulateur acc à 0

# FAUX : tableau tridimensionnel
#taille1, taille2 = 3, magnitude.size # Potentiellement les centres des cercles peuvent êtres partout d'ans l'image, même si ce sont des cercles incomplets
## taille de l'image <=> nombre de cercles
#acc = [[0 for x in range(taille1)] for y in range(taille2)]

i = magnitude.size
j = magnitude.size
k = 3 # Car r, c puis rad

acc = np.zeros((i, j, k))


#5. Calcul du rayon rad pour que le cercle situé en (r, c) passe par le pixel respectif, incrementation dans l'accumulateur de la case qui correspond ) (r, c, rad)
for row in range(0, magnitude.shape[0]):
    for col in range(0, magnitude.shape[1]):
        xi = magnitude[0][row]
        yi = magnitude[col][0]

        rad = math.sqrt((xi - row)**2 + (yi - col)**2)
        acc[row][col][int(t)] +=1




       

