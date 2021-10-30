#OpenCV Version : opencv_python-4.5.4.58
#import matplotlib.pyplot as plt # A commenter si utilisation seulement de displayImg() définie ci-dessous.
import numpy as np
import math
import cv2
import scipy.ndimage as ndimage

def displayImg(img):
    cv2.imshow("Test Display", img)
    cv2.waitKey(0)

def sobel():
    sobelX = cv2.Sobel(blurredImg, cv2.CV_64F, 1, 0, ksize=5)  # X
    sobelY = cv2.Sobel(blurredImg, cv2.CV_64F, 0, 1, ksize=5)  # Y

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

def classic_hough(I, J, K):
    for row in range(0, I):
        for col in range(0, J):
            # Pas très sur si j'ai bien sélectionné les x et les y, il est possible que je les ai inversés, mais ça n'as peut-être pas d'importance pour le calcul
            #xi = magnitude[0][row]
            #yi = magnitude[col][0]
            for xi in range (0, matriceContrast.shape[0]):
                for yi in range(0, matriceContrast.shape[1]):
                    if matriceContrast[xi][yi] > 0:
                        rad = math.sqrt((xi - row)**2 + (yi - col)**2) ### On l'as vu dans le cours et je l'ai également trouvé sur internet cette formule
                        acc[row][col][int(rad)] +=1
    return acc

def normalize(I,J,K, locMax):
    locMaxNorm = np.copy(locMax)

    for i in range(I):
        for j in range(J):
            for k in range(K):
                if k!= 0: # On divise par le rayon si le rayon n'est pas nul <=> le cercle existe
                    locMaxNorm[i][j][k] = locMax[i][j][k] / k
    return locMaxNorm

############# DEBUT ############

image = cv2.imread("images/fourn.png", cv2.IMREAD_GRAYSCALE)

#1. Filtrage Gaussien (en cas de bruit ou de d ́etails très fins)
blurredImg = cv2.GaussianBlur(image, (5, 5), 0)

#2. Filtrage de Sobel, calcul de la magnitude de gradient Imag dans chaque pixel
magnitude = sobel()

#3. Mise en contraste des contours
matriceContrast = contrast(0.26)

#4. Initialisation de toutes les valeurs de l'accumulateur acc à 0
I = image.shape[0] # Height
J = image.shape[1] # Width
K = int(math.sqrt(I**2 + J**2)) # De 1 à maxRadius. Donc jusqu'au sqrt(rows**2 + cols**2)
acc = np.zeros((I, J, int(K)))

#5. Calcul du rayon rad pour que le cercle situé en (r, c) passe par le pixel respectif, incrementation dans l'accumulateur de la case qui correspond ) (r, c, rad)
acc = classic_hough(I, J, K)

#6 On cherche les maximums locaux
locMax = ndimage.maximum_filter(acc, size=(1,1,1)) # dans un rayon d'un cube on a 26 cases voisines
#print(locMax)
#print("localMax.shape \n")
#print(locMax.shape)
#print("\n")
#print("acc.shape \n")
#print(acc.shape)

#7 Normalization, votes et visualisation
locMaxNorm = normalize(I, J, K, locMax)

# Nombre de plus hautes valeurs qu'on veut sélectionner
N = 10
selectedValues = np.argsort(locMaxNorm)[-N:, -N:, -N:] # Pour sélectionner N plus hautes valeurs triés par ordre décroissant
#print(selectedValues) # Test...

result = np.zeros_like(image)

for x in range(selectedValues.shape[0]):
    for y in range(selectedValues.shape[1]):
        for z in range(selectedValues.shape[2]):
            a = int(selectedValues[0][x][0])
            b = int(selectedValues[y][0][0])
            rad = int(selectedValues[0][0][z])
            cv2.circle(result, (a, b), rad, (0, 1, 0))

displayImg(result)