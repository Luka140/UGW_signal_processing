import numpy as np
import scipy.fft as fft
import scipy.signal as spsignal

class Signal:

    def __init__(self, time: np.ndarray, data: np.ndarray, t_unit=None, d_unit=None):

        self.data = data
        self.time = time
        self.sample_frequency = self._calc_sample_frequency()
        self.t_unit, self.d_unit = t_unit, d_unit

    def _calc_sample_frequency(self):
        avg_sample_interval = np.mean(self.time[1:] - self.time[:-1])
        sample_frequency = 1 / avg_sample_interval
        return sample_frequency

    def get_fft(self, positive_half: bool = True):
        signal_samples = len(self.data)  # Number of data points
        avg_sample_interval = 1 / self.sample_frequency

        # Compute FFT
        fft_output = fft.fft(self.data)
        fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT
        fft_freq = fft.fftfreq(signal_samples, avg_sample_interval)  # Frequency axis

        if positive_half:
            mask = fft_freq >= 0
            fft_freq = fft_freq[mask]
            fft_magnitude = fft_magnitude[mask]
        return fft_freq, fft_magnitude

    def get_signal_envelope(self):
        analytic_signal = spsignal.hilbert(self.data)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sample_frequency
        return amplitude_envelope, instantaneous_frequency, instantaneous_phase



