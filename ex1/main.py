import cv2
import numpy as np
import glob

def select_image():
    lst = glob.glob('../res/*.png')
    counter = 0
    for image in lst:
        image_name = image.split('/')
        print(str(counter) + '.- ' + image_name[-1])
        counter = counter + 1
    index = int(input('Selecciona Indice de Imagen a utilizar: '))
    return lst[index]

uri = select_image()

img = cv2.imread(uri)

size = int(input('Ingrese la variable d:'))

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# applying the kernel to the input image
output = cv2.filter2D(img, -1, kernel_motion_blur)

cv2.imshow('Motion Blur', output)
cv2.waitKey(0)