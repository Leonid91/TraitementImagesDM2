import matplotlib.pyplot as plt # A commenter si utilisation seulement de displayImg() définie ci-dessous.
import numpy as np
import math
import cv2
import scipy.ndimage as ndimage

def displayImg(img):
    cv2.imshow("Test Display", img)
    cv2.waitKey(0)

def simpleDetection():

    image = cv2.imread("images/fourn.png")
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 35, minRadius = 5, maxRadius = 30)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

	    # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (255, 0, 0), 2)
            #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.imshow("Output", output)
        plt.imshow(output)
        plt.show()
        cv2.waitKey(0)

#OpenCV Version : opencv_python-4.5.4.58
image = cv2.imread("images/fourn.png", cv2.IMREAD_GRAYSCALE)
#displayImg(image)

#1.  Filtrage Gaussien (en cas de bruit ou de d ́etails très fins)
blur_x = 5
blur_y = 5
blurredImg = cv2.GaussianBlur(image, (blur_x, blur_y), 0)
#displayImg(blurredImg)

#2.  Filtrage de Sobel, calcul de la magnitude de gradient Imag dans chaque
#pixel
sobelX = cv2.Sobel(blurredImg, cv2.CV_64F, 1, 0, ksize=5)  # X
#displayImg(sobelX)
sobelY = cv2.Sobel(blurredImg, cv2.CV_64F, 0, 1, ksize=5)  # Y
#displayImg(sobelY)
magnitude = np.sqrt((sobelX ** 2) + (sobelY ** 2))
magnitude = magnitude / np.max(magnitude) # Pour ramener les valeurs des pixels dans l'intervalle [0, 1], pour que celà
                                          # soit utilisable avec displayImg
#displayImg(magnitude)

# Une autre façon d'afficher les images : (à paraméter pour l'intervalle [0, 1]

#plt.subplot(2,2,3),plt.imshow(sobelX, cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobelY, cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,1),plt.imshow(magnitude, cmap = 'gray')
#plt.title('Magnitude'), plt.xticks([]), plt.yticks([])

#plt.show()

#3.  Mise en contraste des contours
# Fraction t, fixée à partir de plusieurs tests
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


#4.  Initialisation de toutes les valeurs de l'accumulateur acc à 0
I = magnitude.shape[0] # Height
J = magnitude.shape[1] # Width
K = int(math.sqrt(I ** 2 + J ** 2)) # De 1 à maxRadius.  Donc jusqu'au sqrt(rows**2 + cols**2)
acc = np.zeros((I, J, K))


#5.  Calcul du rayon rad pour que le cercle situé en (r, c) passe par le pixel
#respectif, incrementation dans l'accumulateur de la case qui correspond ) (r,
#c, rad)
for row in range(0, magnitude.shape[0]):
    for col in range(0, magnitude.shape[1]):
        # Pas très sur si j'ai bien sélectionné les x et les y, il est possible
        # que je les ai inversés, mais ça n'as peut-être pas d'importance pour
        # le calcul
        xi = magnitude[col][0]
        yi = magnitude[0][row]

        rad = math.sqrt((xi - row) ** 2 + (yi - col) ** 2) ### On l'as vu dans le cours et je l'ai également trouvé sur internet cette
                                                           ### formule
        acc[row][col][int(rad)] +=1

#print(acc)
print("Maximum acc : ", np.max(acc))
print("\n")

#6 On cherche les maximums locaux
locMax = ndimage.maximum_filter(acc, size=(1,1,1)) # dans un rayon d'un cube on a 26 cases voisines
#print("Maximum locMax : ", np.max(locMax))
                                                  #print("\n")
                                                  #print(locMax)


#print("localMax.shape \n")
#print(locMax.shape)
#print("\n")
#print("acc.shape \n")
#print(acc.shape)

#7 Normalization, votes et visualisation

### Pour normaliser en divisant par le rayon.  MARCHE PAS
locMaxNorm = np.copy(locMax == acc)


for i in range(I):
    for j in range(J):
        for k in range(K):
            if k != 0 and k > 1: # On divise par le rayon si le rayon n'est pas nul et sup à 1 <=> le cercle
                                 # existe
                locMaxNorm[i][j][k] = locMax[i][j][k] / k

print("Maximum locMaxNorm : ", np.max(locMaxNorm))
#print(locMaxNorm)

# Nombre de plus hautes valeurs qu'on veut sélectionner
N = 3
#selectedValues = np.argsort(locMaxNorm)[-N:, -N:, -N:] # Pour sélectionner N
#plus hautes valeurs triés par ordre décroissant
selectedValues = np.argsort(locMax)[-N:, -N:, -N:] # Pour sélectionner N plus hautes valeurs triés par ordre décroissant
#print(selectedValues) # Test...
                                                  #print(selectedValues.shape) # Test...

for x in range(selectedValues.shape[0]):
    for y in range(selectedValues.shape[1]):
        for z in range(selectedValues.shape[2]):
            a = int(selectedValues[x][0][0])
            b = int(selectedValues[0][y][0])
            rad = int(selectedValues[0][0][z])
            cv2.circle(image, (a, b), rad, (0, 255, 0))

#displayImg(image)

#plt.imshow(image)
#plt.show()


simpleDetection()
