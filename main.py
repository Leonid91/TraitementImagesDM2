#OpenCV Version : opencv_python-4.5.4.58
#import matplotlib.pyplot as plt # A commenter si utilisation seulement de displayImg() définie ci-dessous.
import numpy as np
import math
import cv2
import scipy.ndimage as ndimage

def displayImg(img):
    cv2.imshow("Display", img)
    cv2.waitKey(0)

def sobel(img):
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # X
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Y

    magnitude = np.sqrt((sobelX ** 2) + (sobelY ** 2))
    magnitude = magnitude / np.max(magnitude) # Pour ramener les valeurs des pixels dans l'intervalle [0, 1], pour que celà soit utilisable avec displayImg

    # Une autre façon d'afficher les images : (à paraméter pour l'intervalle [0, 1]

    #plt.subplot(2,2,3),plt.imshow(sobelX, cmap = 'gray')
    #plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,4),plt.imshow(sobelY, cmap = 'gray')
    #plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,1),plt.imshow(magnitude, cmap = 'gray')
    #plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    #plt.show()
    return magnitude

def contrast(seuil):
    # Fraction t, fixée à partir de plusieurs tests
    t = seuil

    # Pour copier les valeurs > t (seuil) dans une matrice de copie
    matriceCopieSuperieur = np.zeros_like(magnitude)

    #s = 0.0
    for i in range(0, magnitude.shape[0]):
        for j in range(0, magnitude.shape[1]):
            if magnitude[i, j] > t:
                #magnitude[i, j] = 255
                matriceCopieSuperieur[i, j] = 1
            #else:
                #magnitude[i, j] = 0
            #s = s + magnitude[i, j]

    #displayImg(matriceCopieSuperieur)

    # Une autre façon d'afficher :
    #plt.subplot(2,2,1),plt.imshow(magnitude, cmap = 'gray')
    #plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    # Ou
    #plt.imshow(magnitude)

    #plt.show()
    return matriceCopieSuperieur

def classic_hough(I, J, K, matrix):
    for row in range(0, I):
        for col in range(0, J):
            for xi in range (0, matrix.shape[0]):
                for yi in range(0, matrix.shape[1]):
                    if matrix[xi][yi] > 0:
                        rad = math.sqrt((xi - row)**2 + (yi - col)**2) ### On l'as vu dans le cours et je l'ai également trouvé sur internet cette formule
                        acc[row][col][int(rad)] +=1
    return acc

def astuce1():
    #On réduit la taille de l'image
    ratio = 0.25
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def normalize(I,J,K, locMax):
    locMaxNorm = np.copy(locMax)

    for i in range(I):
        for j in range(J):
            for k in range(K):
                if k!= 0: # On divise par le rayon si le rayon n'est pas nul <=> le cercle existe
                    temp = locMax[i][j][k]
                    if temp != 0:
                        locMaxNorm[i][j][k] = temp / k

                        #test non normalisé (il y aura que le gros cercle)
                        if temp == locMax.max():
                            circles.append((i, j, k, temp))                        
    return locMaxNorm

############# DEBUT ############
image = cv2.imread("images/fourn.png", cv2.IMREAD_GRAYSCALE)

#1. Filtrage Gaussien (en cas de bruit ou de d ́etails très fins)
blurredImg = cv2.GaussianBlur(image, (5, 5), 0)

#2. Filtrage de Sobel, calcul de la magnitude de gradient Imag dans chaque pixel
magnitude = sobel(blurredImg)

#3. Mise en contraste des contours
tresholdedImg = contrast(0.26)

#4. Initialisation de toutes les valeurs de l'accumulateur acc à 0
I = image.shape[0] # Height
J = image.shape[1] # Width
K = int(math.sqrt(I**2 + J**2)) # De 1 à maxRadius. Donc jusqu'au sqrt(rows**2 + cols**2)
acc = np.zeros((I, J, int(K)))

#5. Calcul du rayon rad pour que le cercle situé en (r, c) passe par le pixel respectif, incrementation dans l'accumulateur de la case qui correspond ) (r, c, rad)
acc = classic_hough(I, J, K, tresholdedImg)

#6 On cherche les maximums locaux
locMax = ndimage.maximum_filter(acc, mode='constant', size=(1,1,1)) # dans un rayon d'un cube on a 26 cases voisines

#7 Normalisation et visualisation
circles = []
circlesNormalized = []
locMaxNorm = normalize(I, J, K, locMax)

#à l'arrache (pas opti) : on prend que les cercles qui sont près du max
for x in range(locMaxNorm.shape[0]):
    for y in range(locMaxNorm.shape[1]):
        for z in range(locMaxNorm.shape[2]):
            if locMaxNorm[x][y][z] >= 8:
                circlesNormalized.append((x,y,z, locMaxNorm[x][y][z]))

for c in range(0, len(circlesNormalized)):
    cv2.circle(image, (circlesNormalized[c][1], circlesNormalized[c][0]), circlesNormalized[c][2], (0, 0, 255), 1) # cv2.circle(image, center_coordinates, radius, color, thickness)

displayImg(image)
