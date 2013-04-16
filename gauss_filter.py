"""
Second attempt at a high-pass filter

This time we smoothe the data with a Gaussian
and subtract that from the original

"""

from scipy.ndimage.filters import gaussian_filter1d


def gauss_highpass_filter(data, smooth_scale, fs, mode="wrap"):
    smooth_data = gaussian_filter1d(data, smooth_scale*fs, mode=mode)
    return data - smooth_data


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 200.0
    highcut = 1250.0

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 300.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 112 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    x += 0.3*np.exp(-0.5*(t-0.025)**2 / (0.0003)**2)

    plt.plot(t, x, label='Noisy signal')

    y = gauss_highpass_filter(x, 1./lowcut, fs, mode="reflect")

    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.savefig("gauss-filter-example.pdf")
