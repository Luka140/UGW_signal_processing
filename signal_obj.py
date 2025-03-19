import numpy as np
import scipy.fft as fft
import scipy.signal as spsignal
import matplotlib.pyplot as plt


class Signal:

    def __init__(self, time: np.ndarray, data: np.ndarray, t_unit=None, d_unit=None):

        self.data = data.flatten()
        self.time = time.flatten()
        self.t_unit, self.d_unit = t_unit, d_unit
        self.sample_frequency = self._calc_sample_frequency()

        # These are initially None. Will be calculated on first call to the getter and cached
        # These are accessed through the corresponding property (same name without leading underscore)
        self._fft_output, self._fft_frequency, self._fft_magnitude  = None, None, None 
        self._amplitude_envelope, self._instant_phase               = None, None
        self._peak_time, self._peak_amplitude                       = None, None

        self._peak_n            = 5         # Number of peaks to find    
        self._recalc_peak_flag  = False     # Signals that _peak_n changed -> recaulculate peaks

    def _calc_sample_frequency(self):
        avg_sample_interval = np.mean(self.time[1:] - self.time[:-1])
        sample_frequency = 1 / avg_sample_interval
        return sample_frequency

    def get_fft(self, positive_half: bool = True):
        signal_samples = len(self.data)  # Number of data points
        avg_sample_interval = 1 / self.sample_frequency

        fft_output = fft.fft(self.data)
        fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT
        fft_freq = fft.fftfreq(signal_samples, avg_sample_interval)  # Frequency axis

        if positive_half:
            mask = fft_freq >= 0
            fft_freq = fft_freq[mask]
            fft_magnitude = fft_magnitude[mask]
        return fft_output, fft_freq, fft_magnitude

    @property
    def fft_frequency(self):
        if self._fft_frequency is None:
            self._fft_output, self._fft_frequency, self._fft_magnitude = self.get_fft()
        return self._fft_frequency

    @property
    def fft_magnitude(self):
        if self._fft_magnitude is None:
            self._fft_output, self._fft_frequency, self._fft_magnitude = self.get_fft()
        return self._fft_magnitude

    @property
    def fft_output(self):
        if self._fft_output is None:
            self._fft_output, self._fft_frequency, self._fft_magnitude = self.get_fft()
        return self._fft_output


    def _get_signal_envelope(self):
        analytic_signal = spsignal.hilbert(self.data)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sample_frequency
        return amplitude_envelope, instantaneous_frequency, instantaneous_phase

    @property
    def amplitude_envelope(self):
        if self._amplitude_envelope is None:
            self._amplitude_envelope, self._instant_phase, _ = self._get_signal_envelope()
        return self._amplitude_envelope
    @property
    def instant_phase(self):
        if self._instant_phase is None:
            self._amplitude_envelope, self._instant_phase, _ = self._get_signal_envelope()
        return self._instant_phase

    def _get_envelope_peaks(self):
        self._recalc_peak_flag = False
        peak_idx, peak_properties = spsignal.find_peaks(self.amplitude_envelope)
        prominence, *_ = spsignal.peak_prominences(self.amplitude_envelope, peak_idx)
        
        # These are the idx in the prominence/peak array, not in the data array! [0] is the first peak idx, not data[0]
        n_most_prominent_peak_idx = np.argsort(prominence)[-self._peak_n:][::-1] # Sorted from most prominent to least prominent

        return self.time[peak_idx[n_most_prominent_peak_idx]], self.amplitude_envelope[peak_idx[n_most_prominent_peak_idx]]

    @property
    def peak_time(self):
        if self._peak_time is None or self._recalc_peak_flag:
            self._peak_time, self._peak_amplitude = self._get_envelope_peaks()
        return self._peak_time

    @property
    def peak_amplitude(self):
        if self._peak_amplitude is None or self._recalc_peak_flag:
            self._peak_time, self._peak_amplitude = self._get_envelope_peaks()
        return self._peak_amplitude
    
    def set_max_peak_number(self, n):
        self._peak_n = n
        self._recalc_peak_flag = True

    
    def get_stfft(self, window_length_time, plot=True):
        """_summary_

        Args:
            window_length_time (_type_): The window length in time units. The frequency resolution of the method will scale as 1/window_time
            plot (bool, optional): Show the spectogram. Defaults to True.

        Returns:
            _type_: Spectogram of the signal.
        """
        fft_length_to_window_factor = 2
        window_length_samples = round(window_length_time * self.sample_frequency)
        window = spsignal.windows.hann(window_length_samples, sym=True)
        SFT = spsignal.ShortTimeFFT(win=window, hop=window_length_samples//2, fs=self.sample_frequency, mfft=window_length_samples*fft_length_to_window_factor)
        N = len(self.data)
        t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
        Sx2 = SFT.spectrogram(self.data)
        if plot:
            fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit

            t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
            ax1.set_title(rf"Spectrogram ({SFT.m_num*SFT.T:g})$\,s$ Hann ")
            ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " + rf"$\Delta t = {SFT.delta_t:g}\,$s)",ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " + rf"$\Delta f = {SFT.delta_f:g}\,$Hz)", xlim=(t_lo, t_hi))

            Sx_dB = 10 * np.log10(np.fmax(Sx2, 1e-4))  # limit range to -40 dB
            im1 = ax1.imshow(Sx_dB, origin='lower', aspect='auto', extent=SFT.extent(N), cmap='magma')
            fig1.colorbar(im1, label='Power Spectral Density ' + r"$20\,\log_{10}|S_x(t, f)|$ in dB")

            # Shade areas where window slices stick out to the side:
            for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T), (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
                ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.3)
            for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line
                ax1.axvline(t_, color='c', linestyle='--', alpha=0.5)
            ax1.legend()
            fig1.tight_layout()
            plt.show()

        return Sx2

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
        for (time, amplitude) in zip(self.peak_time, self.peak_amplitude):
            axt.text(time, -amplitude*1.1, s=f"{time:.2e}")
        axt.set(xlabel=f'Time ({self.t_unit})', ylabel=f'Signal ({self.d_unit})')
        if tlim is not None:
            axt.xlim(tlim)
        axt.legend()

        axf.set(xlabel=f"Frequency", ylabel="Magnitude")
        axf.plot(self.fft_frequency, self.fft_magnitude, label='FFT')

        plt.show()

    def bandpass(self, lowcut: float, highcut: float, order: int=2):
        bandpass_filter = spsignal.butter(order, Wn=(lowcut, highcut), btype='bandpass', output='sos', fs=self.sample_frequency)
        filtered_signal = spsignal.sosfilt(bandpass_filter, self.data)
        return Signal(self.time, filtered_signal, self.t_unit, self.d_unit)



if __name__ == '__main__':
    ...