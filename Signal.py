import numpy as np
import scipy.fft as fft
import scipy.signal as spsignal
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from typing import Iterable


class Signal:

    def __init__(self, time: np.ndarray, data: np.ndarray, t_unit=None, d_unit=None):

        self.data = data
        self.time = time
        self.sample_frequency = self._calc_sample_frequency()
        self.t_unit, self.d_unit = t_unit, d_unit

        # These are initially None. Will be calculated on first call to the getter and cached
        self._fft_frequency, self._fft_magnitude = None, None
        self._amplitude_envelope, self._instant_phase = None, None
        self._peak_time, self._peak_amplitude = None, None

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

    @property
    def fft_frequency(self):
        if self._fft_frequency is None:
            self._fft_frequency, self._fft_magnitude = self.get_fft()
        return self._fft_frequency

    @property
    def fft_magnitude(self):
        if self._fft_magnitude is None:
            self._fft_frequency, self._fft_magnitude = self.get_fft()
        return self._fft_magnitude

    def get_signal_envelope(self):
        analytic_signal = spsignal.hilbert(self.data)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sample_frequency
        return amplitude_envelope, instantaneous_frequency, instantaneous_phase

    @property
    def amplitude_envelope(self):
        if self._amplitude_envelope is None:
            self._amplitude_envelope, self._instant_phase, _ = self.get_signal_envelope()
        return self._amplitude_envelope
    @property
    def instant_phase(self):
        if self._instant_phase is None:
            self._amplitude_envelope, self._instant_phase, _ = self.get_signal_envelope()
        return self._instant_phase

    def _get_envelope_peaks(self):
        peak_idx, _ = spsignal.find_peaks(self.amplitude_envelope)
        return self.time[peak_idx], self.data[peak_idx]

    @property
    def peak_time(self):
        if self._peak_time is None:
            self._peak_time, self._peak_amplitude = self._get_envelope_peaks()
        return self._peak_time

    @property
    def peak_amplitude(self):
        if self._peak_amplitude is None:
            self._peak_time, self._peak_amplitude = self._get_envelope_peaks()
        return self._peak_amplitude

    def get_trimmed_signal(self, start_time, end_time):
        try:
            trim_index_1 = np.argwhere(self.time <= start_time)[-1][0]
        except IndexError:
            raise IndexError(f"Trim index out of range for start_time. Got {start_time}, min: {self.time[0]}")
        try:
            trim_index_2 = np.argwhere(self.time > end_time)[0][0]
        except IndexError:
            raise IndexError(f"Trim index out of range for end_time. Got {end_time}, max: {self.time[-1]}")
        return Signal(self.time[trim_index_1:trim_index_2], self.data[trim_index_1:trim_index_2], self.t_unit, self.d_unit)

    def plot(self, tlim=None):
        fig, (axt, axf) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
        axt.set_ylabel("Amplitude")
        axt.plot(self.time, self.data, label='Signal')
        axt.plot(self.time, self.amplitude_envelope, label='Envelope')
        axt.plot([self.peak_time, self.peak_time], [-self.peak_amplitude, self.peak_amplitude], '--', color='red')
        axt.set(xlabel=f'Time ({self.t_unit})', ylabel=f'Signal ({self.d_unit})')
        if tlim is not None:
            axt.xlim(tlim)
        axt.legend()

        axf.set(xlabel=f"Frequency", ylabel="Magnitude")
        axf.plot(self.fft_frequency, self.fft_magnitude, label='FFT')

        plt.show()

    @staticmethod
    def bandpass(signal: 'Signal', lowcut: float, highcut: float, order: int=2):
        bandpass_filter = spsignal.butter(order, Wn=(lowcut, highcut), btype='bandpass', output='sos', fs=signal.sample_frequency)
        filtered_signal = spsignal.sosfilt(bandpass_filter, signal.data)
        return Signal(signal.time, filtered_signal, signal.t_unit, signal.d_unit)



if __name__ == '__main__':
    ...