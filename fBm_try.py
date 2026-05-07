import numpy as np
import matplotlib.pyplot as plt
import signalz

def fbm_2d(n, H, seed=None):
    """
    Generates a 2D fractional Brownian motion (fBm) field.

    Parameters:
    n : int
        Grid size (n x n).
    H : float
        Hurst exponent (0 < H < 1, noisy to smooth).
    seed : int or None
        Seed for the random number generator.

    Returns:
    2D np.ndarray, or a real-valued 2D fBm field.
    """
    if seed is not None:
        np.random.seed(seed)

    kx = np.fft.fftfreq(n)
    ky = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0  # avoid division by zero

    gaussian_noise = np.random.normal(size=(n, n)) # + 1j * np.random.normal(size=(n, n))  # Generates Gaussian noise

    fBm_map = np.fft.fft2(gaussian_noise)

    freq_spectrum = fBm_map * (k ** -(H + 1))
    freq_spectrum[0, 0] = 0.0  # zero out frequency 0 component

    fBm_map = np.fft.ifft2(freq_spectrum).real

    fBm_map = (1 / np.max(fBm_map)) * fBm_map # Normalize the maximum value in the fbm_map to 1

    return fBm_map

def fLm_2d(n, H, alpha=1.5, beta=0.5, sigma = 1.0, pos=0, seed=0, iterations = 0, persistence=0.5, H_scaling=1, alpha_scaling=1, sigma_scaling=1):
    np.random.seed(seed)

    kx = np.fft.fftfreq(n)
    ky = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0  # avoid division by zero

    a_stable_array = np.empty((n, n))
    for col in range(n):
        a_stable_array[col] = signalz.levy_noise(n, alpha, beta, sigma, pos) # + 1j * signalz.levy_noise(n, alpha, beta, sigma, pos))

    #a_stable_noise = (signalz.levy_noise(n*n, alpha, beta, sigma, pos) + 1j * signalz.levy_noise(n*n, alpha, beta, sigma, pos)).reshape(n, n)


    fLm_map = np.fft.fft2(a_stable_array)

    freq_spectrum = fLm_map * (k ** -(H + 1))
    freq_spectrum[0, 0] = 0.0  # zero out frequency 0 component
    fLm_map = np.fft.ifft2(freq_spectrum).real
    #fLm_map = (1 / np.max(fLm_map)) * fLm_map # Normalize the maximum value in the fbm_map to 1

    alpha *= alpha_scaling

    if alpha > 2:
        alpha = 2


    if iterations > 0:
        fLm_map += persistence * fLm_2d(n, H * H_scaling, alpha, beta, sigma * sigma_scaling, pos, seed + 1, iterations - 1, persistence, H_scaling, alpha_scaling, sigma_scaling)

    return fLm_map





if __name__ == "__main__":
    n = 2000
    H = 0.7
    seed = 6835

    alpha = 1.95
    beta = 0
    sigma = 1
    pos = 0
    octaves = 0
    persistence = 1.0
    H_scaling = 1.0
    alpha_scaling = 1.0
    sigma_scaling = 1.0


    fBm_field = fbm_2d(n, H, seed)
    fLm_field = fLm_2d(n, 0.1, 1.96, beta, sigma, pos, seed, octaves, persistence, H_scaling, alpha_scaling, sigma_scaling)

    plt.imshow(fBm_field, cmap='magma')
    plt.title(f"2D fLm (H={H}, alpha = {1.96}, beta = {beta})")
    #plt.savefig(f"frame-{a_iter * 11 + iter + 198}.png")
    plt.show()