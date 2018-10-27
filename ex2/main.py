
import cv2
import numpy as np
import glob

def select_image_background():
	lst = glob.glob("../res/*background.png")
	lst += glob.glob("../res/*background.jpg")
	lst += glob.glob("../res/*background.jpeg")
	for i, image in enumerate(lst):
		image_name = image.split('/')
		print(str(i) + " - " + image_name[-1])
	index = int(input('Selecciona el índice de la imagen de fondo a utilizar: '))
	return lst[index]

def select_image():
	lst = glob.glob("../res/*greenscreen.png")
	lst += glob.glob("../res/*greenscreen.jpg")
	lst += glob.glob("../res/*greenscreen.jpeg")
	for i, image in enumerate(lst):
		image_name = image.split('/')
		print(str(i) + " - " + image_name[-1])
	index = int(input('Selecciona el índice de la imagen de pantalla verde a utilizar: '))
	return lst[index]

path = select_image_background()
bg = cv2.imread(path)

path = select_image()
img = cv2.imread(path)

blur_quantity = int(input('Ingresa el valor de degradacion del fondo (recomendado: 17): '))
h = np.ones((blur_quantity,blur_quantity), np.float32)/(blur_quantity*blur_quantity)

bg_height, bg_width, _ = bg.shape

height, width, _ = img.shape
dim = (width, height)

bg = cv2.resize(bg, dim, interpolation = cv2.INTER_AREA)

if blur_quantity != 0:
	bg = cv2.filter2D(bg,-1, h)

lower_color_bound = np.array([0,10,0])
upper_color_bound = np.array([255,205,255])

frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(frame_hsv, lower_color_bound, upper_color_bound)
median = cv2.medianBlur(mask, 15)

frame_filtered = cv2.bitwise_and(img, img, mask = median)

for i in range(0, width):
	for j in range(0, height):
		if np.any(frame_filtered[j,i] == 0) :
			frame_filtered[j,i] = bg[j,i]




cv2.imshow("filter", frame_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows() 

