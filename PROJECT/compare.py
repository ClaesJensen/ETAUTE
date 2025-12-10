 
    # --- 5. Plotting (Separated Figures) ---
    plot_time_alignment(ref, measured_aligned, lag)
    plot_coherence(f, Cxy)
    plot_bode(f, H)

    # Show all plots at once

    # A. Convert your Scipy data to a Pyfar FrequencyData object
    # pyfar needs to know the complex data and the frequency bins
    n_fft = (len(H) - 1) * 2

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
    safe_range = [20, 17000]
    # C. Calculate the Inverse Filter (The "Farina" Magic)
    # This function performs the Kirkeby regularization, IFFT, and Windowing automatically.

    inverse_filter = pf.dsp.regularized_spectrum_inversion(
        signal=h_pyfar,
        frequency_range=safe_range,
        regu_outside=1.0,  # Don't boost/cut outside the range (0dB)
        regu_inside=10 ** (-30 / 20),  # -40dB regularization.
        # A good balance between flat response and low ringing.
        # If you get "pre-echo", increase this (e.g. -30/20).
        normalized=True,  # Maximize volume to 0dBFS
    )
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
        "./PINK_NOISE_NEW_REFERENCE_PREFILTERED.wav",
        "./pink_noise_test-prefiltered.wav",
    )
    # plt.plot(np.fft(filtered))
    plot_spectrum_comparison(original[:, 0], filtered[:, 0], fs)
    plt.show()

    plot_raw_fft(original[:, 0], filtered[:, 0], fs)



