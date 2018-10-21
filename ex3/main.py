import cv2
import numpy as np
import glob

def select_image():
	lst = glob.glob("../res/*.png")
	lst += glob.glob("../res/*.jpg")
	for i, image in enumerate(lst):
		image_name = image.split('/')
		print(str(i) + " - " + image_name[-1])
	index = int(input('Selecciona el índice de la imagen a utilizar: '))
	return lst[index]

path = select_image()
img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print('0 - Prewitt')
print('1 - Sobel')
print('2 - Canny')
print('3 - Roberts')
filter_index = int(input('Selecciona el índice del filtro que deseas: '))

if filter_index == 0:
	# prewitt
	gaussian_size = int(input('Selecciona el tamaño del kernel gaussiano (debe ser impar, recomendado: 9): '))
	img_gaussian = cv2.GaussianBlur(img_gray,(gaussian_size,gaussian_size),0)

	prewitt_size = int(input('Selecciona el tamaño del kernel de Prewitt (debe ser impar, recomendado: 3): '))

	kernel_x = np.zeros((prewitt_size,prewitt_size), np.float32)
	kernel_y = np.zeros((prewitt_size,prewitt_size), np.float32)

	kernel_x[0:prewitt_size, 0] = 1
	kernel_x[0:prewitt_size, prewitt_size-1] = -1

	kernel_y[0, 0:prewitt_size] = -1
	kernel_y[prewitt_size-1, 0:prewitt_size] = 1

	# convolution
	prewitt_x = cv2.filter2D(img_gaussian, -1, kernel_x)
	prewitt_y = cv2.filter2D(img_gaussian, -1, kernel_y)

	img_prewitt = prewitt_x + prewitt_y

	cv2.imshow("Prewitt", img_prewitt)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif filter_index == 1:
	# sobel
	gaussian_size = int(input('Selecciona el tamaño del kernel gaussiano (debe ser impar, recomendado: 3): '))
	img_gaussian = cv2.GaussianBlur(img_gray,(gaussian_size,gaussian_size),0)

	sobel_size = int(input('Selecciona el tamaño del kernel de Sobel (debe ser impar, recomendado: 3): '))

	img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=sobel_size)
	img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=sobel_size)
	img_sobel = img_sobelx + img_sobely

	cv2.imshow("Sobel", img_sobel)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif filter_index == 2:
	canny_x = int(input('Selecciona el tamaño en x del kernel de Canny (recomendado: 100): '))
	canny_y = int(input('Selecciona el tamaño en x del kernel de Canny (recomendado: 100): '))
	img_canny = cv2.Canny(img,canny_x,canny_y)

	cv2.imshow("Sobel", img_canny)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif filter_index == 3:
	# roberts
	gaussian_size = int(input('Selecciona el tamaño del kernel gaussiano (debe ser impar, recomendado: 3): '))
	img_gaussian = cv2.GaussianBlur(img_gray,(gaussian_size,gaussian_size),0)

	kernel_x = np.zeros((2,2), np.float32)
	kernel_y = np.zeros((2,2), np.float32)

	kernel_x[0,0] = 1
	kernel_y[1,0] = 1
	kernel_x[1,1] = -1
	kernel_y[0,1] = -1

	# convolution
	roberts_x = cv2.filter2D(img_gaussian, -1, kernel_x)
	roberts_y = cv2.filter2D(img_gaussian, -1, kernel_y)

	img_roberts = roberts_x + roberts_y

	cv2.imshow("Roberts", img_roberts)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

else:
	print('Opción inválida...')

