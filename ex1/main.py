import cv2
import numpy as np
import glob
import math

HORIZONTAL = LINEAL = 1
VERTICAL = RADIAL = 2
DIAGONAL = ZOOM = 3

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

if degradacion == RADIAL:

    kernel_motion_blur = np.zeros((size, size))

    img64_float = img.copy()

    Mvalue = np.sqrt(((img64_float.shape[0]/2.0)**2.0)+((img64_float.shape[1]/2.0)**2.0))


    ploar_image = cv2.linearPolar(img64_float,(img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue,cv2.WARP_FILL_OUTLIERS)


    kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    ploar_image = cv2.filter2D(ploar_image, -1, kernel_motion_blur)

    cartisian_image = cv2.linearPolar(ploar_image, (img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue, cv2.WARP_INVERSE_MAP)

    cartisian_image = cartisian_image/200
    ploar_image = ploar_image/255

    cv2.imshow('Radial Blur', cartisian_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if degradacion == ZOOM:

    kernel_motion_blur = np.zeros((size, size))

    img64_float = img.copy()

    Mvalue = np.sqrt(((img64_float.shape[0]/2.0)**2.0)+((img64_float.shape[1]/2.0)**2.0))


    ploar_image = cv2.linearPolar(img64_float,(img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue,cv2.WARP_FILL_OUTLIERS)


    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    ploar_image = cv2.filter2D(ploar_image, -1, kernel_motion_blur)

    cartisian_image = cv2.linearPolar(ploar_image, (img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue, cv2.WARP_INVERSE_MAP)

    cartisian_image = cartisian_image/200
    ploar_image = ploar_image/255

    cv2.imshow('Radial Blur', cartisian_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()