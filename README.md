# Fourier-Image-Processing
A Python CLI toolkit demonstrating image processing using 2D Discrete Fourier Transform(DFT). Features frequency-domain convolution(Gaussian, Laplacian, Sobel), Image compression, and log-spectrum thresholding with neighbourhood interpolation
for perodic noise mitigation.

By shifting the spatial data into frequency components via FFT, this project provides clean, reproducible implementations of:
1. Analytical Convolution: Instead of applying filters via direct matrix convoltions we use the fact that T[f*g]=T[f].T[g]( T[.]:= Fourier transform, * :=Convolution). So, we save processing time by multiplying the frequency spectrums instead of directly convolving.
2. Energy based Compression: By setting the coefficiens of low spatial frequencies to 0. We use the fact that the human eye is more sensitive to low spatial frequencies when detecting color patterns.
3. Perodic Noise Restoration: Perodic noises take up the form of high amplitude noise spikes in the frequency domain. We detect those spikes using statistical threshold masks and smoothing them by simply setting it to zero.

