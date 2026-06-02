print('hello world')

import os
import numpy as np
from PIL import Image

def noise_reduction(image_path, threshold_factor=3.5, protect_radius=10):
    """
    Removes high-amplitude periodic noise spikes from the image spectrum.
    Noise take up 
    """
    try:
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # Forward FFT and shift low frequencies to center
    a = np.array(img, dtype=np.float64) 
    A = np.fft.fft2(a)
    A_shift = np.fft.fftshift(A)
    
    # Magnitude Spectrum under a Log transform for spike detection
    magnitude_spectrum = np.abs(A_shift)
    log_spectrum = 20 * np.log1p(magnitude_spectrum) # 20 is there just to scale the values...better scalars can be chosen..but here i just eyeballed

    rows, cols = a.shape
    crow, ccol = rows // 2, cols // 2
    
    # Threshold masking
    thresh = np.mean(log_spectrum) + threshold_factor * np.std(log_spectrum) # Here, we are marking the outliers which have more amplitude than others( the Z-score is our threshold factor)
    peaks = (log_spectrum > thresh) 
    '''
    we need to be careful here. The centre elements of the magnitude spectrum carry the overall luminescence of the image and thus have high amplitude and thus are marked as outliers.
    So we need to introduce another radius,r, and only the outliers outside of 'r' will be considered noise spikes.
    '''
    
    # Protect the central low-frequency (DC)
    r = protect_radius
    peaks[max(0, crow-r):min(rows, crow+r), max(0, ccol-r):min(cols, ccol+r)] = False # Removing center components from 'peaks'
    
    '''
    we have more than one option here, we can set the peaks to zero, or, we can also set them to the average of their non-peak neighbours.
    I am choosing the easier route here(results are only slighty worse)
    '''
    A_shift[peaks] = 0

    # Return matrix to normal alignment and apply Inverse FFT
    A_ishift = np.fft.ifftshift(A_shift)
    a_filtered = np.abs(np.fft.ifft2(A_ishift))

    # Normalize floating artifacts safely and save
    a_filtered = np.clip(a_filtered, 0, 255).astype(np.uint8)
    a_filtered_img = Image.fromarray(a_filtered)