import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import freqz, tf2zpk, group_delay, dimpulse
import matplotlib


def plot_filter(a, b):
    # Pole-Zero Plot
    z, p, _ = tf2zpk(b, a)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(np.real(z), np.imag(z), "o", label="Zeros")
    plt.plot(np.real(p), np.imag(p), "x", label="Poles")
    plt.plot(
        np.cos(np.linspace(0, 2 * np.pi, 100)),
        np.sin(np.linspace(0, 2 * np.pi, 100)),
        "b",
    )  # Unit circle
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Pole-Zero Plot")
    plt.legend()
    plt.grid()

    # Impulse Response
    plt.subplot(2, 2, 2)
    n, h = dimpulse((b, a, 1), n=25)
    plt.stem(n, np.squeeze(h))
    plt.xlabel("n")
    plt.ylabel("h[n]")
    plt.title("Impulse Response")
    plt.grid()

    # Magnitude Response
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 2, 3)
    plt.plot(w - np.pi, 20 * np.log10(np.abs(h)))
    plt.title("Magnitude Response")
    plt.ylim([-5, 5])
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$|H(e^{j\omega})|$ (dB)")
    plt.grid()

    # Group Delay
    w, gd = group_delay((b, a), whole=True)
    plt.subplot(2, 2, 4)
    plt.plot(w - np.pi, gd)
    plt.title("Group Delay")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\tau_{grp}(\omega)$ (Samples)")
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_signal_time(signal, fs=1.0, title="Signal in Time Domain"):
    n = len(signal)
    t = np.arange(n) / fs

    # Create the Figure and Axes explicitly
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot on the specific axes object
    ax.plot(t, signal)

    # Set labels using the .set_ method
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    ax.grid(True)
    fig.tight_layout()

    # Return the figure object
    return fig


if __name__ == "__main__":
    plt.plot([1, 2, 3, 4, 5, 6])

    print(matplotlib.get_backend())
    plt.show()
