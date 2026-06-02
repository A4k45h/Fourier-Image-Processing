print('hlo wrld')

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def fourier_convolution(image_path, kernel_type='gb', n=0):
    """
    Applies spatial domain filters in the frequency domain using 2D FFT.
    Demonstrates circular convolution properties and magnitude/phase splits.
    """
    try:
        img_raw = Image.open(image_path).convert('L') #Loading image in grayscale(for easier processing)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    a = np.array(img_raw, dtype=np.float32) #Changing into array form
    rows, cols = a.shape

    # Forward Fourier Transform of the image
    A = np.fft.fft2(a) #A 2D discrete fourier transform
    A_shift = np.fft.fftshift(A) #This just reorganizes the trasnform and makes it centralized( without shifting the data remains unitelligible)
    
    # Define standard kernels
    kernels = {
        'gb':  np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32) / 256, #Blurs using weights from a gaussian distribution
        's': np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 20, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0]
], dtype=np.float32), #Sharpens the image
        's_x': np.array([
    [-1, -2, 0, 2, 1],
    [-2, -4, 0, 4, 2],
    [-3, -6, 0, 6, 3],
    [-2, -4, 0, 4, 2],
    [-1, -2, 0, 2, 1]
], dtype=np.float32), #Detects edges on x-axis
        's_y': np.array([
    [-1, -2, -3, -2, -1],
    [-2, -4, -6, -4, -2],
    [ 0,  0,  0,  0,  0],
    [ 2,  4,  6,  4,  2],
    [ 1,  2,  3,  2,  1]
], dtype=np.float32), #Detects edges on y-axis
        'l': np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 24, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
], dtype=np.float32), #Detects all edges
        'b': np.ones((5, 5), dtype=np.float32) / 25 #This just set the value of the middle pixel to the avegrage of neighbouring pixels acheiving a blurrinhg effect
    }
    
    k = kernels.get(kernel_type, kernels['gb'])

    # Padding spatial kernels to match image matrix size
    k_padded = np.zeros((rows, cols), dtype=np.float32)
    kr, kc = k.shape
    k_padded[:kr, :kc] = k # we fill the empty spots with zeros to pad
    
    # Align kernel phase relative to origin (circular convolution theorem)
    k_padded = np.roll(k_padded, -(kr // 2), axis=0) 
    k_padded = np.roll(k_padded, -(kc // 2), axis=1) # Just circular shifting around our axes
    
    # Transform kernel to frequency spectrum
    K = np.fft.fft2(k_padded)
    K_shift = np.fft.fftshift(K) 

    # Frequency Multiplication
    G = A * K 
    G_shift = np.fft.fftshift(G)

    # Inverse FFT back to Spatial Domain
    g = np.abs(np.fft.ifft2(G)) # Here we are taking the magnitude of the inverse trnsform( inverse transform is an array of complex numbers) since we need the real values for the image. Alternatively, we can also take the real parts of the inverse transform if the imaginary parts are small..but taking the magnitude is common practice.
    
    
    def display(arr, title=''):
        img_display = np.clip(arr, 0, 255).astype(np.uint8)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_display, cmap='gray')
        plt.title(str(title))
        plt.axis('off')
        plt.show()
    
    # Display results
    if n==1:
         display(a, 'Original Image')
    elif n==2:
         display(k_padded, 'Kernel Image')
    elif n==3:
         display(g, f'Final Image - {kernel_type} kernel')
    elif n==4:
         display(np.abs(A_shift), 'FFT Magnitude Spectrum')
    elif n==5:
         display(np.abs(K_shift), 'Kernel FFT Magnitude')
    elif n==6:
         display(np.abs(G_shift), 'Filtered FFT Magnitude')
    elif n==0:
         display(a, 'Original Image')
         display(k_padded, 'Kernel Image')
         display(g, f'Final Image - {kernel_type} kernel')
         display(np.abs(A_shift), 'FFT Magnitude Spectrum')
         display(np.abs(K_shift), 'Kernel FFT Magnitude')
         display(np.abs(G_shift), 'Filtered FFT Magnitude')

#testing on images
test_i = np.zeros((200, 200), dtype=np.uint8)
test_i[50:150, 50:150] = 255  # White square in middle
Image.fromarray(test_i).save("test_s.png")
x, y = np.ogrid[:200, :200]
cx, cy = 100, 100
radius = 50
test_c = ((x - cx)**2 + (y - cy)**2 <= radius**2) * 255 # Circle in middle
Image.fromarray(test_c.astype(np.uint8)).save("test_c.png")

fourier_convolution("test_s.png", kernel_type='s_y', n=0)