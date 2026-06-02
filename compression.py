import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def fourier_compress(image_path, tau, n=5):
    """
    Compresses an image by discarding low-energy Fourier coefficients.
    """
    try:
        img_raw = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    a = np.array(img_raw)

    name = os.path.splitext(os.path.basename(image_path))[0] # Extracting name from the path

    # Forward FFT
    A = np.fft.fft2(a)
    amplitude = np.abs(A)

    # Thresholding
    A_filtered = np.copy(A)
    A_filtered[np.square(amplitude) < tau] = 0  # Zero out elements where energy (|A|^2) < tau
    A_threshed = np.copy(A)
    A_threshed[np.square(amplitude) >= tau] = 0 # Saving it for visualization

    # Inverse FFT
    a_compressed = np.abs(np.fft.ifft2(A_filtered))
    
    def display(arr, title=''):
        img_display = np.clip(arr, 0, 255).astype(np.uint8)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_display, cmap='gray')
        plt.title(str(title))
        plt.axis('off')
        plt.show()
    
    # Display results
    if n == 1:
        display(a, 'Original Image')
    elif n == 3:
        display(np.abs(A_filtered), f"Magnitude spectrum of {name}")
    elif n == 5:
        display(a_compressed, f'{name} compressed')
    elif n == 4:
        display(np.abs(A)-np.abs(A_filtered), 'Deleted frequencies')
    elif n == 2:
        display(np.abs(A), f"Magnitude spectrum of {name} after filtering the frequencies")
    elif n == 0:
        display(a, 'Original Image')
        display(np.abs(A), f"Magnitude spectrum of {name}")
        display(np.abs(A_filtered), f"Magnitude spectrum of {name} after filtering the frequencies")
        display(np.abs(A_threshed), 'Deleted frequencies')
        display(a_compressed, f'{name} compressed')

    original_size = os.path.getsize(image_path)
    compressed_img = np.clip(a_compressed, 0, 255).astype(np.uint8)
    compressed_size = compressed_img.nbytes  # This is memory size, not file size
    
    output_path = f"{name}_compressed.png"
    Image.fromarray(compressed_img).save(output_path)
    compressed_file_size = os.path.getsize(output_path) # This is to get the size of the compressed image
    
    total_coeffs = A.size
    non_zero_coeffs = np.count_nonzero(A_filtered)
    sparsity = (1.0 - (non_zero_coeffs / total_coeffs)) * 100  # Measure of how much we have removed from the image(%)

    print(f"Threshold (tau): {tau}")
    print(f"Original File Size    : {original_size / 1024:.2f} KB")  # Sizes were in 'bytes' so we change into 'KB' by dividing by 1024
    print(f"Compressed Memory Size: {compressed_size / 1024:.2f} KB (in memory)")
    print(f"Compression Ratio     : {original_size / compressed_size:.2f}x")
    print(f"Mathematical Sparsity : {sparsity:.2f}% of coefficients zeroed out.")
    print(f"Non-zero coefficients : {non_zero_coeffs:,} / {total_coeffs:,}")

# image for testing
test_img = np.zeros((200, 200), dtype=np.uint8)
test_img[50:150, 50:150] = 255  # White square in middle
Image.fromarray(test_img).save("test_square.png")
x, y = np.ogrid[:200, :200]
cx, cy = 100, 100
radius = 50
test_circle = ((x - cx)**2 + (y - cy)**2 <= radius**2) * 255 # White circle in middle
Image.fromarray(test_circle.astype(np.uint8)).save("test_circle.png")

fourier_compress("test_circle.png", tau=100, n=0)