import cv2
import numpy as np
import glob
import math

HORIZONTAL = LINEAL = 1
VERTICAL = 2
DIAGONAL = 3
def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)
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

size = int(input('Ingrese la magnitud :'))

print("1.- Lineal")
print("2.- Radial")
print("3.- Zoom")

degradacion = int(input("Seleccione el tipo de degradacion: "))

if degradacion == LINEAL:


    print("1.- Horizontal")
    print("2.- Vertical")
    print("3.- Diagonal")

    direction = int(input("Seleccione la direccion del kernel: "))

    kernel_motion_blur = np.zeros((size, size))

    if direction == HORIZONTAL:
            
        # generating the kernel
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

    elif direction == VERTICAL:
        # generating the kernel
        kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
    elif direction == DIAGONAL:
        #angle = int(input("Ingrese angulo: "))
        np.fill_diagonal(kernel_motion_blur, 1)
        kernel_motion_blur = kernel_motion_blur / size

        
    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)

    cv2.imshow('Motion Blur', output)
    cv2.waitKey(0)

if degradacion == 2:
    kernel_motion_blur = np.zeros((size, size))

    
    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)

    cv2.imshow('Motion Blur', output)
    cv2.waitKey(0)