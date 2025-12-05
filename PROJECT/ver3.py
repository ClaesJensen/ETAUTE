import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def calculate_lag(ref: np.ndarray, measured: np.ndarray) -> int:
    """Calculates the lag between two signals using FFT correlation."""
    print("Calculating correlation...")
    # Mode='full' is required to find the offset
    corr = sig.correlate(measured, ref, mode="full", method="fft")
    lags = sig.correlation_lags(len(measured), len(ref), mode="full")
    best_lag = lags[np.argmax(corr)]
    print(f"Best Lag: {best_lag} samples")
    return best_lag


def apply_alignment(
    ref: np.ndarray, measured: np.ndarray, lag: int, fine_tune: int = 0
):
    """
    Shifts signals.
    fine_tune: Positive int shifts 'measured' to the RIGHT (adds delay).
               Negative int shifts 'measured' to the LEFT (removes delay).
    """
    adjusted_lag = lag - fine_tune

    if adjusted_lag > 0:
        measured_aligned = measured[adjusted_lag:]
        ref_aligned = ref
    elif adjusted_lag < 0:
        measured_aligned = measured
        ref_aligned = ref[abs(adjusted_lag) :]
    else:
        measured_aligned = measured
        ref_aligned = ref

    min_len = min(len(measured_aligned), len(ref_aligned))
    return ref_aligned[:min_len], measured_aligned[:min_len]


# def compute_transfer_function(ref, measured, fs, nperseg=4096):
#     """
#     Computes the H1 Transfer Function (H = Pxy / Pxx) and Coherence.
#     """
#     # Pxy: Cross Spectral Density
#     f, Pxy = sig.csd(ref, measured, fs=fs, nperseg=nperseg)
#     # Pxx: Power Spectral Density of Source
#     _, Pxx = sig.welch(ref, fs=fs, nperseg=nperseg)
#     # Coherence
#     _, Cxy = sig.coherence(ref, measured, fs=fs, nperseg=nperseg)
#
#     # H1 Estimate
#     H = Pxy / Pxx
#     return f, H, Cxy
#
def compute_transfer_function(ref, measured, fs, nperseg=65536):  # Changed from 4096
    print(f"Computing Transfer Function (Resolution: {fs / nperseg:.2f} Hz)...")

    # Use overlap to reduce variance (noverlap = nperseg // 2 is standard)
    f, Pxy = sig.csd(ref, measured, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, Pxx = sig.welch(ref, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, Cxy = sig.coherence(ref, measured, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    H = Pxy / Pxx
    return f, H, Cxy


def plot_time_alignment(ref, measured, lag):
    """Visualizes the time-domain alignment in its own figure."""

    # Normalize for plotting
    def normalize(arr):
        return arr / (np.max(np.abs(arr)) + 1e-12)

    ref_norm = normalize(ref)
    meas_norm = normalize(measured)

    # Create window for zoomed view (center of signal usually good)
    mid_point = len(ref) // 2
    window = 1000  # 1000 samples zoom
    start = max(0, mid_point - window // 2)
    end = min(len(ref), start + window)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Time Alignment Check (Lag: {lag} samples)")
    ax.plot(ref_norm[start:end], label="Reference", alpha=0.8)
    ax.plot(meas_norm[start:end], label="Measured", alpha=0.8, linestyle="--")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Samples (Zoomed Window)")
    ax.set_ylabel("Normalized Amplitude")
    plt.tight_layout()


def plot_coherence(f, Cxy):
    """Visualizes the Coherence in its own figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.semilogx(f, Cxy, color="green")
    ax.set_title(f"Coherence (Mean: {np.mean(Cxy):.3f})")
    ax.set_ylabel("Coherence (0-1)")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(20, 20000)
    plt.tight_layout()


def plot_bode(f, H):
    """Plots Magnitude and Phase in their own figure."""
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    phase_deg = np.rad2deg(np.unwrap(np.angle(H)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 1. Magnitude
    ax1.semilogx(f, mag_db, color="blue")
    ax1.set_title("Frequency Response (Magnitude)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both")
    ax1.set_xlim(20, 20000)

    # 2. Phase
    ax2.semilogx(f, phase_deg, color="orange")
    ax2.set_title("Phase Response")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both")

    plt.tight_layout()


def design_flattening_filter(
    f, H, fs, taps=4095, limit_db=15, min_freq=40, max_freq=22050
):
    """
    Creates an FIR filter to flatten response, restricted to a specific frequency range.
    """
    # 1. Extract Magnitude in dB
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)

    # 2. Smooth the response
    smoothed_db = gaussian_filter1d(mag_db, sigma=1)

    # 3. Normalize around the average volume
    avg_level = np.mean(smoothed_db)
    variations = smoothed_db - avg_level

    # 4. Invert variations to get target
    target_curve_db = -variations

    out_of_band_mask = (f < min_freq) | (f > max_freq)

    # Smoothly fade out the correction at the edges?
    # For now, we will simply zero out the correction outside the band.
    target_curve_db[out_of_band_mask] = 0

    # 5. Safety Clamping
    # target_curve_db = np.clip(target_curve_db, -limit_db, limit_db)

    # 6. Convert to Linear Gain
    target_gain = 10 ** (target_curve_db / 20)

    # 7. Prepare Frequency Grid (Hz)
    freq_grid = f.copy()
    gain_grid = target_gain.copy()

    # Ensure 0 Hz included
    if freq_grid[0] != 0:
        freq_grid = np.insert(freq_grid, 0, 0)
        gain_grid = np.insert(gain_grid, 0, 1.0)  # 1.0 gain (0dB) at DC

    # Ensure Nyquist included
    nyquist = fs / 2
    if freq_grid[-1] < nyquist:
        freq_grid = np.append(freq_grid, nyquist)
        gain_grid = np.append(gain_grid, 1.0)  # 1.0 gain (0dB) at Nyquist
    elif freq_grid[-1] > nyquist:
        freq_grid[-1] = nyquist

    # 8. Generate FIR Coefficients
    fir_coeffs = sig.firwin2(taps, freq_grid, gain_grid, fs=fs)

    return fir_coeffs, target_curve_db


def verify_correction(f, H, fir_coeffs, fs):
    """
    Simulates the result of applying the filter.
    """
    # Calculate Frequency Response of the FIR filter
    w, h_filter = sig.freqz(fir_coeffs, worN=f, fs=fs)

    # "Predicted" System = Original System * Correction Filter
    # (Multiplication in Freq Domain = Convolution in Time Domain)
    H_corrected = H * h_filter

    mag_orig = 20 * np.log10(np.abs(H) + 1e-12)
    mag_corr = 20 * np.log10(np.abs(H_corrected) + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.semilogx(f, mag_orig, label="Original Magnitude", alpha=0.5)
    plt.semilogx(
        f, mag_corr, label="Predicted Corrected Magnitude", color="black", linewidth=2
    )
    plt.title(f"Correction Result (Filter Length: {len(fir_coeffs)} taps)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both")
    plt.legend()
    plt.xlim(20, 20000)
    plt.ylim(-60, -15)
    plt.tight_layout()

    return H_corrected


def fractional_octave_smoothing(f, mag_db, fraction=12):
    """
    Applies fractional octave smoothing (e.g., 1/12th octave) to a dB magnitude curve.
    Mimics REW smoothing.
    """
    # 1. Define the bandwidth for the smoothing window at every frequency
    # Bandwidth factor for 1/N octave
    factor = 2 ** (1 / (2 * fraction))

    smoothed = np.zeros_like(mag_db)

    # This loop is slow in pure Python, but easy to understand.
    # For <100k points it's acceptable (~1-2 seconds).
    for i in range(len(f)):
        freq = f[i]

        # Skip DC component
        if freq == 0:
            smoothed[i] = mag_db[i]
            continue

        # Determine the window limits for this specific frequency
        f_lower = freq / factor
        f_upper = freq * factor

        # Find indices in the array that fall into this window
        # We assume 'f' is sorted (standard FFT output)
        idx_start = np.searchsorted(f, f_lower)
        idx_end = np.searchsorted(f, f_upper)

        # Ensure we have at least one point
        if idx_start == idx_end:
            idx_end += 1

        # Average the POWER, not the dB (standard audio practice)
        # 1. Convert window slice to linear power
        slice_db = mag_db[idx_start:idx_end]
        slice_power = 10 ** (slice_db / 10)

        # 2. Average power
        avg_power = np.mean(slice_power)

        # 3. Convert back to dB
        smoothed[i] = 10 * np.log10(avg_power + 1e-12)

    return smoothed


def main():
    # --- 1. Load Data ---
    print("Loading audio...")
    # measured_raw, sr_meas = sf.read("./ETAUTE-MÅLINGER høj.wav")
    measured_raw, sr_meas = sf.read("./ETAUTE-MÅLINGER mellem.wav")
    ref_raw, sr_ref = sf.read("./PINK_NOISE_REFERENCE.wav")

    print(sr_ref)
    assert sr_ref == sr_meas, "Sampling rates do not match!"

    # --- 2. Pre-process (Ensure Mono) ---
    # Take first channel if stereo
    ref = ref_raw[:, 0] if ref_raw.ndim > 1 else ref_raw
    measured = measured_raw[:, 0] if measured_raw.ndim > 1 else measured_raw

    #  Crop Reference logic (reference is 10 seconds but recording is 8)
    ref = ref[: 480000 - int(sr_ref * 2)]

    # --- 3. Compute & Align ---
    lag = calculate_lag(ref, measured)
    ref_aligned, measured_aligned = apply_alignment(ref, measured, lag, fine_tune=10)

    f, H, Cxy = compute_transfer_function(
        ref_aligned, measured_aligned, fs=sr_ref, nperseg=65536
    )  # --- 4. System Identification ---
    # 2. Calculate Raw Magnitude
    mag_raw = 20 * np.log10(np.abs(H) + 1e-12)

    # 3. Apply REW-style Smoothing (e.g., 1/12th Octave)
    # This cleans up the high freq noise while preserving the bass detail
    print("Applying 1/12 octave smoothing...")
    mag_smoothed = fractional_octave_smoothing(f, mag_raw, fraction=12)

    # 4. Design Filter (Use the SMOOTHED version for the design)
    # Note: We reconstruct a "smoothed H" effectively for the design function
    # Or just modify design_flattening_filter to take the magnitude array directly.

    # For visualization:
    plt.figure(figsize=(10, 6))
    plt.semilogx(f, mag_raw, color="lightgray", alpha=0.5, label="Raw (High Res)")
    plt.semilogx(f, mag_smoothed, color="blue", label="1/12 Oct Smoothed")
    plt.xlim(20, 24000)
    plt.grid(which="both")
    plt.legend()
    plt.show()
    # --- 6. Design Correction Filter ---
    print("Designing inverse filter...")
    # Taps: Higher = better bass resolution, but more latency.
    # 4095 is decent for 48kHz.
    correction_filter, target_curve = design_flattening_filter(
        f, H, fs=sr_ref, taps=int(2**16) - 1
    )

    # --- 7. Verify Result ---
    verify_correction(f, H, correction_filter, sr_ref)

    plt.show()


if __name__ == "__main__":
    main()
