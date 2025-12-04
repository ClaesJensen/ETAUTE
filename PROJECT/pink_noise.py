import numpy as np
import soundfile as sf

from matplotlib import pyplot as plt
from scipy.signal import welch


def main():
    fs = 48000
    pink = generate_pink_noise(duration_s=1000.0, fs=fs, rms=0.05, random_state=42)

    plt.style.use("seaborn-v0_8")

    # Welch parameters
    nperseg = int(2**12)

    f, Pxx = welch(
        pink,
        fs=fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nperseg,
        scaling="density",
        detrend="constant",
    )

    # PSD in dB/Hz
    Pxx_dB = 10 * np.log10(Pxx)

    plt.figure()
    plt.semilogx(f, Pxx_dB)
    plt.grid(True)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.title("Welch PSD estimate of pink noise")
    plt.show()
    sf.write("pink_noise.wav", pink, fs)


def generate_pink_noise(duration_s=5.0, fs=48000, rms=0.1, random_state=None):
    """
    Generate pink (1/f) noise using frequency-domain shaping.

    Parameters
    ----------
    duration_s : float
        Length of the signal in seconds.
    fs : int
        Sample rate in Hz.
    rms : float
        Target RMS level of the output signal.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    pink : np.ndarray
        1D array of pink noise samples (float32).
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    N = int(duration_s * fs)

    # White noise
    white = rng.standard_normal(N)

    # FFT of white noise
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    White = np.fft.rfft(white)

    # Build 1/sqrt(f) filter (pink). Avoid division by zero at DC.
    # Set DC (freq=0) to 0 so we don't get a huge DC component.
    S = np.ones_like(freqs)
    S[1:] = 1.0 / np.sqrt(freqs[1:])
    S[0] = 0.0

    Pink = White * S

    # Back to time domain
    pink = np.fft.irfft(Pink, n=N)

    # Normalize to desired RMS
    current_rms = np.sqrt(np.mean(pink**2))
    if current_rms > 0:
        pink = pink * (rms / current_rms)

    return pink.astype(np.float32)


if __name__ == "__main__":
    main()
