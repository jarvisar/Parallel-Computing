#Adam Jarvis

import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
from numba import jit, njit, prange
from math import pi, sqrt, exp
import time

#@jit(nopython=True)
#Generates 2d Guassian kernel
def kernel(size,s):
    center=(int)(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            diff=np.sqrt((i-center)**2+(j-center)**2)
            kernel[i,j]=np.exp(-(diff**2)/(2*s**2))
    return kernel/np.sum(kernel)

@jit(nopython=True)
def gauss_filter(in_img, out_img, kernel): 
    kernel_len = len(kernel)
    kernel_center = int((kernel_len-1)/2)
    for c in range((in_img.shape[2])):
        for  x in range(in_img.shape[1]): 
            for y in range(in_img.shape[0]):
                val = 0
                for i in range(kernel_len): 
                    for j in range(kernel_len):
                        if (x+i < in_img.shape[1]) and (x+i >= 0) and (y+j < in_img.shape[0] ) and (y+j >=0 ):
                            val += (int(in_img[y+j-kernel_center,x+i-kernel_center,c] * kernel[j][i]))
                out_img[y,x,c] = val
                  
img = np.array(Image.open('noisy1.jpg')) 
print(img.shape) 
imgblur= img.copy() 
#blurfilter(img, imgblur)

begin = time.time()
gauss_filter(img, imgblur, kernel(11,1))
end = time.time()
print("Elapsed time = ", end - begin)

# Display and save blurred image 
fig = plt.figure() 
ax = fig.add_subplot(1, 2, 1) 
imgplot = plt.imshow(img) 
ax.set_title('Before') 
ax = fig.add_subplot(1, 2, 2) 
imgplot = plt.imshow(imgblur) 
ax.set_title('After') 
img2= Image.fromarray(imgblur) 
img2.save('blurred.jpg') 
