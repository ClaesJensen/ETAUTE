import soundfile as sf
import numpy as np
import scipy.signal as sig

# Load reference (unfiltered) and the *filtered file you actually played*
ref, fs = sf.read("./lastlast_random_pink_noise_20hzfilter.wav")
played, fs_p = sf.read("./new/pink_noise_filter_applied.wav")

assert fs == fs_p
# Convert to mono if stereo
if ref.ndim > 1:
    ref = ref[:, 0]
if played.ndim > 1:
    played = played[:, 0]

corr = sig.correlate(played, ref, mode="full", method="fft")
lag = np.argmax(np.abs(corr))
max_corr_norm = corr[lag] / (np.linalg.norm(ref) * np.linalg.norm(played))
print("Max *normalized* correlation:", np.abs(max_corr_norm))
lags = sig.correlation_lags(len(played), len(ref))
print("Max correlation:", np.max(np.abs(corr)) / len(ref))
