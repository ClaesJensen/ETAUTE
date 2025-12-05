import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt

# --- 1. Load Data (Raw) ---
data_high, sr_high = sf.read("./ETAUTE-MÅLINGER test.wav")
data_ref, sr_ref = sf.read("./PINK_NOISE_REFERENCE.wav")

# Ensure 1D arrays
ref = data_ref[:, 0] if data_ref.ndim > 1 else data_ref
ref = ref[: 480000 - int(sr_ref * 2)]
low = data_high[:, 0] if data_high.ndim > 1 else data_high

# --- 2. Compute Lag (WITHOUT manual slicing) ---
# We pass the FULL arrays to correlate. It finds the global maximum match.
print("Calculating correlation...")
corr = sig.correlate(low, ref, mode="full", method="fft")
lags = sig.correlation_lags(len(low), len(ref), mode="full")
best_lag = lags[np.argmax(corr)] - 140

print(f"Best Lag: {best_lag} samples")

# --- 3. Align ---
if best_lag > 0:
    # Recording is delayed. Slice 'low' to match 'ref' start.
    low_aligned = low[best_lag:]
    ref_aligned = ref
elif best_lag < 0:
    # Recording started early. Slice 'ref' to match 'low' start.
    low_aligned = low
    ref_aligned = ref[abs(best_lag) :]
else:
    low_aligned = low
    ref_aligned = ref

# --- 4. Trim to match lengths ---
min_len = min(len(low_aligned), len(ref_aligned))
low_final = low_aligned[:min_len]
ref_final = ref_aligned[:min_len]


# --- 5. Verify ---
# Normalize for plotting
def normalize(arr):
    return arr / np.max(np.abs(arr))


low_norm = normalize(low_final)
ref_norm = normalize(ref_final)

f, Cxy = sig.coherence(low_final, ref_final, fs=sr_ref, nperseg=2048)
# Create a figure with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# --- Subplot 1: Visual Alignment Check (Zoomed) ---
start = 10000
window = 20000  # Small window to see phase alignment

ax1.set_title(f"Alignment Check (Lag: {best_lag} samples)")
ax1.plot(ref_norm[start : start + window], label="Reference", alpha=0.8)
ax1.plot(low_norm[start : start + window], label="Recording", alpha=0.8, linestyle="--")
ax1.legend(loc="upper right")
ax1.grid(True)
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Samples (Relative to window start)")

# --- Subplot 2: Coherence Check ---
# (Calculate coherence if you haven't already, otherwise just plot)
# f, Cxy = sig.coherence(low_final, ref_final, fs=sr_ref, nperseg=2048)

ax2.semilogx(f, Cxy)
ax2.set_title(f"Magnitude Squared Coherence (Mean Score: {np.mean(Cxy):.3f})")
ax2.grid(True, which="both")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Coherence (0 to 1)")

# Adjust spacing so titles don't overlap
plt.tight_layout()
plt.show()


# Configuration
fs = sr_ref  # Sampling rate (e.g., 48000)
nperseg = 4096  # Frequency resolution (higher = better bass detail)

# --- 1. Calculate Power Spectral Densities (CSD and PSD) ---
# Pxy: Cross Spectral Density (Relationship between Source and Rec)
f, Pxy = sig.csd(ref_final, low_final, fs=fs, nperseg=nperseg)

# Pxx: Power Spectral Density of the Source
f, Pxx = sig.welch(ref_final, fs=fs, nperseg=nperseg)

# --- 2. Calculate H1 Transfer Function ---
# H = Output / Input
H = Pxy / Pxx

# --- 3. Extract Magnitude and Phase ---
# Magnitude in dB
mag_db = 20 * np.log10(np.abs(H) + 1e-12)  # +epsilon to avoid log(0)

# Phase in Degrees
# We "unwrap" the phase to prevent it from jumping instantly from +180 to -180
phase_rad = np.unwrap(np.angle(H))
phase_deg = np.rad2deg(phase_rad)

# --- 4. Plotting (Bode Plot) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot Magnitude
ax1.semilogx(f, mag_db, color="blue")
ax1.set_title("Frequency Response (Magnitude)")
ax1.set_ylabel("Magnitude (dB)")
ax1.grid(True, which="both")
ax1.set_xlim(20, 20000)  # Standard Audio Range

# Plot Phase
ax2.semilogx(f, phase_deg, color="orange")
ax2.set_title("Phase Response")
ax2.set_ylabel("Phase (Degrees)")
ax2.set_xlabel("Frequency (Hz)")
ax2.grid(True, which="both")
ax2.set_xlim(20, 20000)

plt.tight_layout()
plt.show()
