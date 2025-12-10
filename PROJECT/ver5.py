import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pyfar as pf


def check_signal_health(name, audio):
    """Diagnoses if a signal is compressed, clipped, or healthy."""
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    crest_factor = peak / rms
    crest_factor_db = 20 * np.log10(crest_factor)

    print(f"--- DIAGNOSTIC: {name} ---")
    print(f"  Peak Amplitude: {peak:.4f}")
    print(f"  RMS Level:      {rms:.4f}")
    print(f"  Crest Factor:   {crest_factor:.2f} (raw) | {crest_factor_db:.1f} dB")

    if crest_factor_db < 10:
        print("  [FAIL] Signal is HEAVILY COMPRESSED or CLIPPED.")
        print("         Pink noise should have a crest factor > 12 dB.")
    elif crest_factor_db < 12:
        print("  [WARNING] Signal shows signs of limiting/compression.")
    else:
        print("  [PASS] Signal dynamic range looks healthy.")
    print("-" * 30)


def clean_signal(audio, fs, cutoff=20):
    """
    Removes DC offset and low-frequency rumble (<20Hz) to prevent
    analysis errors and 'fake' clipping.
    """
    print(f"Cleaning signal (DC removal + {cutoff}Hz High-pass)...")

    # 1. Remove DC Offset (Center the waveform)
    audio = audio - np.mean(audio)

    # 2. High-pass filter (Butterworth 4th order) to remove rumble
    sos = sig.butter(4, cutoff, "hp", fs=fs, output="sos")
    audio_filtered = sig.sosfilt(sos, audio)

    return audio_filtered


def calculate_lag(ref: np.ndarray, measured: np.ndarray) -> int:
    """Calculates the lag between two signals using FFT correlation."""
    print("Calculating correlation...")
    # Mode='full' is required to find the offset
    corr = sig.correlate(measured, ref, mode="full")
    lags = sig.correlation_lags(len(measured), len(ref))
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
        rer = ref
    elif adjusted_lag < 0:
        measured_aligned = measured
        rer = ref[abs(adjusted_lag) :]
    else:
        print("3")
        measured_aligned = measured
        rer = ref

    min_len = min(len(measured_aligned), len(rer))
    return rer[:min_len], measured_aligned[:min_len]


def compute_transfer_function(ref, measured, fs, nperseg=4096):
    """
    Computes the H1 Transfer Function (H = Pxy / Pxx) and Coherence.
    """
    # Pxy: Cross Spectral Density
    f, Pxy = sig.csd(ref, measured, fs=fs, nperseg=nperseg)
    # Pxx: Power Spectral Density of Source
    _, Pxx = sig.welch(ref, fs=fs, nperseg=nperseg)
    # Coherence
    _, Cxy = sig.coherence(ref, measured, fs=fs, nperseg=nperseg)

    # H1 Estimate
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
    window = len(ref)  # 1000 samples zoom
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


def apply_filter_to_file(inverse_filter, input_file, output_file):
    print(f"Applying filter to {input_file}...")

    # 1. Load the music/audio you want to correct
    data, fs = sf.read(input_file)

    # Check Sampling Rate (Must match the filter!)
    if fs != inverse_filter.sampling_rate:
        raise ValueError(
            f"Sampling rate mismatch! Audio: {fs}, Filter: {inverse_filter.sampling_rate}"
        )

    # 2. Get the filter coefficients (Impulse Response)
    # inverse_filter.time is shape (1, n_samples), we need 1D array
    ir = inverse_filter.time[0]

    # 3. Perform Convolution (Filtering)
    # We use fftconvolve because it is much faster for long IRs
    # If stereo, we apply to both channels
    if data.ndim == 2:
        # Process Left
        filtered_L = sig.fftconvolve(data[:, 0], ir, mode="same")
        # Process Right
        filtered_R = sig.fftconvolve(data[:, 1], ir, mode="same")
        filtered_audio = np.vstack((filtered_L, filtered_R)).T
    else:
        # Mono
        filtered_audio = sig.fftconvolve(data, ir, mode="same")

    # 4. Normalize to prevent clipping
    # Filters often boost frequencies, pushing levels above 1.0
    max_val = np.max(np.abs(filtered_audio))
    if max_val > 1.0:
        print(f"Warning: Signal clipped (Max: {max_val:.2f}). Normalizing to -0.1 dB.")
        filtered_audio = filtered_audio / max_val * 0.99

    # 5. Save

    sf.write(output_file, filtered_audio, fs)
    print(f"Saved filtered audio to: {output_file}")
    return (data, filtered_audio, fs)


def plot_spectrum_comparison(original, filtered, fs):
    """
    Plots the frequency spectrum (PSD) of the original vs filtered signal.
    """
    print("Computing spectrum...")

    # Ensure signals are the same length for fair comparison (optional but good practice)
    min_len = min(len(original), len(filtered))
    original = original[:min_len]
    filtered = filtered[:min_len]

    # --- Use Welch's Method for a clean "Average Frequency Response" ---
    # nperseg=4096 gives good low-frequency resolution
    f_orig, p_orig = sig.welch(original, fs=fs, nperseg=8192)
    # f_filt, p_filt = sig.welch(filtered, fs=fs, nperseg=8192)

    # Convert to dB
    # We add 1e-12 to avoid log(0) errors
    db_orig = 10 * np.log10(p_orig + 1e-12)
    # db_filt = 10 * np.log10(p_filt + 1e-12)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot Original
    plt.semilogx(f_orig, db_orig, label="Original Signal", alpha=0.6, color="gray")

    # Plot Filtered
    # plt.semilogx(f_filt, db_filt, label="Filtered (Corrected)", alpha=0.9, color="blue")

    plt.title("Spectrum Analysis: Before vs. After Correction")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)

    # Limit view to audible range
    plt.xlim(20, 20000)

    # Auto-scale Y-axis to focus on the data
    # max_db = max(np.max(db_orig))
    # plt.ylim(max_db - 60, max_db + 5)  # Show top 60dB range

    plt.tight_layout()
    plt.show()


def plot_raw_fft(original, filtered, fs):
    """
    Computes and plots the raw FFT magnitude of the original and filtered signals.
    """
    print("Computing raw FFT...")

    # 1. Compute FFT (rfft is faster for real-valued audio)
    # We use the length of the signal as the FFT size
    n_orig = len(original)
    n_filt = len(filtered)

    fft_orig = np.fft.rfft(original)
    fft_filt = np.fft.rfft(filtered)

    # 2. Get Frequency Axis
    freqs_orig = np.fft.rfftfreq(n_orig, d=1 / fs)
    freqs_filt = np.fft.rfftfreq(n_filt, d=1 / fs)

    # 3. Calculate Magnitude in dB
    # We normalize by the signal length to keep magnitudes comparable
    mag_orig_db = 20 * np.log10(np.abs(fft_orig) / n_orig + 1e-12)
    mag_filt_db = 20 * np.log10(np.abs(fft_filt) / n_filt + 1e-12)

    # 4. Plot
    plt.figure(figsize=(12, 6))

    # Plot Original
    plt.semilogx(
        freqs_orig,
        mag_orig_db,
        label="Original Signal",
        alpha=0.5,
        color="gray",
        linewidth=0.5,
    )

    # Plot Filtered
    plt.semilogx(
        freqs_filt,
        mag_filt_db,
        label="Filtered Signal",
        alpha=0.8,
        color="blue",
        linewidth=0.5,
    )

    plt.title("Raw FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.xlim(0, 20000)  # Limit to audible range

    # Set reasonable Y-limits based on the data
    # (ignoring DC offset at index 0 which can be huge)
    max_val = np.max(mag_filt_db[1:])
    plt.ylim(max_val - 80, max_val + 10)

    plt.tight_layout()
    plt.show()


def main():
    # --- 1. Load Data ---
    # measured_raw, sr_meas = sf.read("./ETAUTE-MÅLINGER høj.wav")
    # ref_raw, sr_ref = sf.read("./PINK_NOISE_REFERENCE.wav")
    #
    # measured_raw, sr_meas = sf.read("./ETAUTE-MÅLINGER lav.wav")
    #
    # measured_raw, sr_meas = sf.read("./prefiltered_measurement_not_normalized.wav")
    # ref_raw, sr_ref = sf.read("./PINK_NOISE_NEW_REFERENCE_PREFILTERED.wav")
    # Bæææm
    # measured_raw, sr_meas = sf.read("./20hz22khz_measurement.wav")
    # ref_raw, sr_ref = sf.read("./Pink_20_22000_-12_dBV_48k_Float_LR.wav")
    # Boooom
    # measured_raw, sr_meas = sf.read("./hopefully_last_fl_measurement.wav")
    # ref_raw, sr_ref = sf.read("./hopefully_last_pink_noise.wav")
    # Verification
    measured_raw, sr_meas = sf.read("./new/unfiltered_measurement_original_noise.wav")
    ref_raw, sr_ref = sf.read("./new/Pink_20_24000_-12_dBV_48k_Float_LR.wav")

    print(sr_ref)
    assert sr_ref == sr_meas

    # --- 2. Pre-process (Ensure Mono) ---
    ref = ref_raw[:, 0] if ref_raw.ndim > 1 else ref_raw
    measured = measured_raw[:, 0] if measured_raw.ndim > 1 else measured_raw
    # --- NEW STEP: Clean Signals (DC Offset & Rumble) ---
    # measured = clean_signal(measured, sr_meas)
    # We also clean the reference just to be safe/consistent
    # ref = clean_signal(ref, sr_ref)
    #  Crop Reference logic (reference is 10 seconds but recording is 8)
    # ref = ref[: 480000 - int(sr_ref * 2)]
    check_signal_health("Reference (Source)", ref)
    check_signal_health("Measured (Mic)", measured)
    plt.plot(ref)
    plt.show()

    def rms(x):
        return np.sqrt(np.mean(x**2))

    print("RMS ref:", rms(ref), "RMS measured:", rms(measured))
    # --- 3. Compute & Align ---
    lag = calculate_lag(ref, measured)
    # lag = lag - 6000
    # measured_aligned = measured[lag : len(ref) + lag]

    # Use your general alignment function
    ref, measured_aligned = apply_alignment(ref, measured, lag)

    # (Optional) if you want to see the aligned time signals:
    plot_time_alignment(ref, measured_aligned, lag)

    # --- 4. System Identification ---
    f, H, Cxy = compute_transfer_function(
        ref, measured_aligned, fs=sr_ref, nperseg=int(2**14)
    )

    # --- 5. Plotting (Separated Figures) ---
    plot_time_alignment(ref, measured_aligned, lag)
    plot_coherence(f, Cxy)
    plot_bode(f, H)

    # Show all plots at once

    # A. Convert your Scipy data to a Pyfar FrequencyData object
    # pyfar needs to know the complex data and the frequency bins
    n_fft = (len(H) - 1) * 2
    print(n_fft)

    # --- Create the correct pyfar Signal object ---
    # domain='freq' tells pyfar this is FFT data, not raw audio
    h_pyfar = pf.Signal(H, sr_ref, n_samples=n_fft, domain="freq")

    # --- Inspect the pyfar inputs & outputs ---
    # h_pyfar was created as: pf.Signal(H, sr_ref, n_samples=n_fft, domain="freq")

    # 1) Shape & simple stats of your H (from SciPy)
    print("H (scipy) length:", len(H))
    print(
        "H (scipy) mag dB: min {:.1f}, max {:.1f}, mean {:.1f}".format(
            20 * np.log10(np.abs(H).min() + 1e-30),
            20 * np.log10(np.abs(H).max() + 1e-30),
            20 * np.log10(np.abs(H).mean() + 1e-30),
        )
    )

    # 2) pyfar freq-domain data inside the Signal
    print("h_pyfar.freq shape:", h_pyfar.freq.shape)
    print(
        "h_pyfar.freq magnitude dB: min {:.1f}, max {:.1f}, mean {:.1f}".format(
            20 * np.log10(np.abs(h_pyfar.freq).min() + 1e-30),
            20 * np.log10(np.abs(h_pyfar.freq).max() + 1e-30),
            20 * np.log10(np.abs(h_pyfar.freq).mean() + 1e-30),
        )
    )

    # B. Define the frequency range you want to correct
    # It is dangerous to correct < 40Hz or > 18kHz usually
    safe_range = [60, 21000]
    # C. Calculate the Inverse Filter (The "Farina" Magic)
    # This function performs the Kirkeby regularization, IFFT, and Windowing automatically.

    inverse_filter = pf.dsp.regularized_spectrum_inversion(
        signal=h_pyfar,
        frequency_range=safe_range,
        regu_outside=1.0,  # Don't boost/cut outside the range (0dB)
        regu_inside=10 ** (-40 / 20),  # -40dB regularization.
        # A good balance between flat response and low ringing.
        # If you get "pre-echo", increase this (e.g. -30/20).
        normalized=True,  # Maximize volume to 0dBFS
    )

    # 2. Normalize to avoid clipping in REW (optional but recommended)
    # We scale the peak to -1 dBFS so REW reads it clearly

    # 3. Save as a standard WAV file

    # 3) Inspect inverse_filter in pyfar freq-domain directly (do NOT rfft the time-domain)
    print("inverse_filter.freq shape:", inverse_filter.freq.shape)
    print(
        "inverse_filter.freq magnitude dB: min {:.1f}, max {:.1f}, mean {:.1f}".format(
            20 * np.log10(np.abs(inverse_filter.freq).min() + 1e-30),
            20 * np.log10(np.abs(inverse_filter.freq).max() + 1e-30),
            20 * np.log10(np.abs(inverse_filter.freq).mean() + 1e-30),
        )
    )

    attenuation_dB = -6.0
    gain_linear = 10 ** (attenuation_dB / 20)
    inverse_filter = inverse_filter * gain_linear
    ir_data = inverse_filter.time[0]
    output_filename = "my_inverse_filter_REW2.wav"
    sf.write(output_filename, ir_data, inverse_filter.sampling_rate)
    print(f"Exported filter to {output_filename}")
    # --- 7. Visualize the Inverse Filter ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Time Domain (Impulse Response of the Filter)
    # We plot the result to ensure the peak is centered (it should be)
    ax[0].plot(inverse_filter.times, inverse_filter.time[0])
    ax[0].set_title("Inverse Filter Impulse Response (Correction Filter)")
    ax[0].set_xlabel("Time (s)")
    ax[0].grid(True)

    # Frequency Domain (The Filter's EQ Curve)
    # We plot the magnitude to see the "Inverse" EQ curve
    freqs_inv = np.fft.rfftfreq(
        inverse_filter.n_samples, d=1 / inverse_filter.sampling_rate
    )
    mag_inv = 20 * np.log10(np.abs(np.fft.rfft(inverse_filter.time[0])) + 1e-12)

    ax[1].semilogx(freqs_inv, mag_inv)
    ax[1].set_title("Inverse Filter Frequency Response")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Gain (dB)")
    ax[1].set_xlim(20, 20000)
    ax[1].grid(True, which="both")

    plt.tight_layout()
    plt.show()

    simulated_response = h_pyfar * inverse_filter

    # --- 10. Visual Comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # A. Plot Original Response (Magnitude)
    # We use pyfar's plotting utilities or manual matplotlib.
    # Let's stick to matplotlib to keep it consistent with your previous code.

    # Get frequency axes
    f_axis = simulated_response.frequencies

    # Calculate Magnitudes in dB
    mag_orig = 20 * np.log10(np.abs(h_pyfar.freq) + 1e-12).flatten()
    mag_inv = 20 * np.log10(np.abs(inverse_filter.freq) + 1e-12).flatten()
    mag_sim = 20 * np.log10(np.abs(simulated_response.freq) + 1e-12).flatten()

    # Plot
    ax.semilogx(f_axis, mag_orig, label="Original Measurement", alpha=0.5, color="gray")
    ax.semilogx(
        f_axis,
        mag_inv,
        label="Inverse Filter (The Correction)",
        alpha=0.7,
        linestyle="--",
        color="orange",
    )
    ax.semilogx(f_axis, mag_sim, label="Simulated Result", linewidth=2, color="blue")

    ax.set_title("Predicted System Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(20, 20000)
    ax.set_ylim(-60, 60)
    ax.grid(True, which="both")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Plot Impulse Response of the Simulation
    plt.figure(figsize=(10, 5))
    # Normalize for viewing
    ir_sim = simulated_response.time[0]
    ir_sim = ir_sim / np.max(np.abs(ir_sim))

    # Create a time axis centered around the peak
    t_axis = np.linspace(0, len(ir_sim) / sr_ref, len(ir_sim))

    plt.plot(t_axis, ir_sim)
    plt.title("Simulated Impulse Response (Check for Pre-ringing)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    # Zoom in very close to the main spike
    peak_idx = np.argmax(np.abs(ir_sim))
    plt.xlim(t_axis[peak_idx] - 0.005, t_axis[peak_idx] + 0.005)  # +/- 5ms window
    plt.show()

    (original, filtered, fs) = apply_filter_to_file(
        inverse_filter,
        "./new/NEW_Pink_20_24000_-12_dBV_48k_Float_LR.wav",
        "./new/NEW_Pink_20_24000_-12_dBV_48k_Float_LR_filtered.wav",
    )
    plot_spectrum_comparison(original[:, 0], filtered[:, 0], fs)
    plt.show()

    plot_raw_fft(original[:, 0], filtered[:, 0], fs)


if __name__ == "__main__":
    main()
