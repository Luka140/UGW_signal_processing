import numpy as np
import scipy.fft as fft
import scipy.signal as spsignal
import scipy.signal.windows as spwindow
import matplotlib.pyplot as plt
from typing import Iterable, Union, List

class Signal:

    def __init__(self, time: np.ndarray, data: np.ndarray, t_unit=None, d_unit=None):

        self.data = data.flatten()
        self.time = time.flatten()
        self.t_unit, self.d_unit = t_unit, d_unit
        self.sample_frequency = self._calc_sample_frequency()

        # These are initially None. Will be calculated on first call to the getter and cached
        # These are accessed through the corresponding property (same name without leading underscore)
        self._fft_output, self._fft_frequency, self._fft_magnitude  = None, None, None 
        self._fft_start_index                                       = None
        self._amplitude_envelope, self._instant_phase               = None, None
        self._peak_indices, self._peak_width                        = None, None

        self._fft_pad_times     = 4         # Number of signal durations to zero pad for fft 
        self._peak_n            = 5         # Number of peaks to find    
        self._fft_recalc_flag   = False 
        self._recalc_peak_flag  = False     # Signals that _peak_n changed -> recaulculate peaks

    def _calc_sample_frequency(self):
        avg_sample_interval = np.mean(self.time[1:] - self.time[:-1])
        sample_frequency = 1 / avg_sample_interval
        return sample_frequency

    def get_fft(self, positive_half: bool = True):
        self._fft_recalc_flag = False
        avg_sample_interval = 1 / self.sample_frequency

        # Clip off start for better fit
        relative_threshold = 0.02
        zero_centered_data = self.data - np.mean(self.data)
        fft_start_index = np.where(np.abs(zero_centered_data) > (np.max(np.abs(zero_centered_data)) * relative_threshold))[0][0]
        
        # Detect last zero crossing before wavepacket. Otherwise start signal at zero regularly
        
        if fft_start_index > 0.05 * self.data.size:
            try:
                zero_crossing = np.where(zero_centered_data[:fft_start_index-1] * zero_centered_data[1:fft_start_index] < 0)[0][-1]
            except IndexError:
                print("could not find zero crossing point for FFT start index")
                zero_crossing = 0
            interval_data = self.data[zero_crossing:]
        else: 
            zero_crossing = 0
            interval_data = self.data 


        # Zero padding increases the granularity of the frequency domain (not actual resolution)
        if self._fft_pad_times > 1: 
            interval_data = np.pad(interval_data, (0, (self._fft_pad_times - 1) * len(interval_data)), 'constant')
            # interval_data = np.concat((interval_data, np.zeros(interval_data.size * self._fft_pad_times)))

        fft_output = fft.fft(interval_data)
        clipped_samples = len(interval_data)

        fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT
        fft_freq = fft.fftfreq(clipped_samples, avg_sample_interval)

        if positive_half:
            mask = fft_freq >= 0
            fft_freq = fft_freq[mask]
            fft_magnitude = fft_magnitude[mask]
        return fft_output, fft_freq, fft_magnitude, zero_crossing

    @property
    def fft_frequency(self):
        if self._fft_frequency is None or self._fft_recalc_flag:
            self._fft_output, self._fft_frequency, self._fft_magnitude, self._fft_start_index = self.get_fft()
        return self._fft_frequency

    @property
    def fft_magnitude(self):
        if self._fft_magnitude is None or self._fft_recalc_flag:
            self._fft_output, self._fft_frequency, self._fft_magnitude, self._fft_start_index = self.get_fft()
        return self._fft_magnitude
    
    @property
    def fft_start_index(self):
        if self._fft_output is None or self._fft_recalc_flag:
            self._fft_output, self._fft_frequency, self._fft_magnitude, self._fft_start_index = self.get_fft()
        return self._fft_start_index

    @property
    def fft_output(self):
        if self._fft_output is None or self._fft_recalc_flag:
            self._fft_output, self._fft_frequency, self._fft_magnitude, self._fft_start_index = self.get_fft()
        return self._fft_output
    
    @property
    def characteristic_frequency(self):
        return np.sum(self.fft_magnitude * self.fft_frequency) / np.sum(self.fft_magnitude)

    
    def set_fft_pad_times(self, n):
        self._fft_pad_times = n
        self._fft_recalc_flag = True 
    
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

    def _get_envelope_peak_idxs(self):
        self._recalc_peak_flag = False
        peak_idx, peak_properties = spsignal.find_peaks(self.amplitude_envelope)
        prominence, *_ = spsignal.peak_prominences(self.amplitude_envelope, peak_idx)
        
        # These are the idx in the prominence/peak array, not in the data array! [0] is the first peak idx, not data[0]
        n_most_prominent_peak_idx = np.argsort(prominence)[-self._peak_n:][::-1] # Sorted from most prominent to least prominent

        return peak_idx[n_most_prominent_peak_idx]

    @property
    def peak_time(self):
        if self._peak_indices is None or self._recalc_peak_flag:
            self._peak_indices = self._get_envelope_peak_idxs()
        return self.time[self._peak_indices]

    @property
    def peak_amplitude(self):
        if self._peak_indices is None or self._recalc_peak_flag:
            self._peak_indices = self._get_envelope_peak_idxs()
        return self.amplitude_envelope[self._peak_indices]
    
    @property 
    def peak_width(self):
        if self._peak_width is None or self._recalc_peak_flag:
            self._peak_width = spsignal.peak_widths(self.amplitude_envelope, self._peak_indices, rel_height=.7)
        return self._peak_width

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

    def bandpass(self, lowcut: float, highcut: float, order: int=2):
        bandpass_filter = spsignal.butter(order, Wn=(lowcut, highcut), btype='bandpass', output='sos', fs=self.sample_frequency)
        # filtered_signal = spsignal.sosfilt(bandpass_filter, self.data)
        filtered_zero_phase = spsignal.sosfiltfilt(bandpass_filter, self.data)
        return Signal(self.time, filtered_zero_phase, self.t_unit, self.d_unit)       
    
    def zero_average_signal(self):
        return Signal(self.time, self.data - np.mean(self.data), self.t_unit, self.d_unit)

    def to_array(self):
        """
        Returns a Numpy array representation of the signal.
        Column 0 is time, column 1 is data [[t1, d1], [t2, d2], ...]
        """
        return np.array([self.time, self.data]).T

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
        fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
        self._plot_helper(self, axtime, axfrequency, tlim)
        plt.show()

    def get_time_reversed_section(self, t_lower, t_upper):
        trimmed_signal = self.get_trimmed_signal(t_lower, t_upper)
        windowed_data = spwindow.tukey(len(trimmed_signal.data), alpha=0.1) * trimmed_signal.data
        zero_start_time = trimmed_signal.time - trimmed_signal.time[0]
        return Signal(zero_start_time, windowed_data[::-1], trimmed_signal.t_unit, trimmed_signal.d_unit)

    @staticmethod
    def _plot_helper(signal: "Signal", axt, axf, tlim=None, label="", colors=None, plot_envelope=True, plot_waveform=True, plot_peak_width=False):

        # Set default colors if not provided
        if plot_envelope and plot_waveform:
            waveform_color = colors[0] if colors else None
            envelope_color = colors[1] if colors else None
        else:
            waveform_color = colors[0] if colors else None
            envelope_color = colors[0] if colors else None

        axt.set_ylabel("Amplitude")
        if plot_waveform:
            axt.plot(signal.time, signal.data, label=f'{label} Signal', alpha=0.7, color=waveform_color)
        if plot_envelope:
            axt.plot(signal.time, signal.amplitude_envelope, label=f'{label} Envelope', color=envelope_color)

        # # -------- Plot signal peaks # TODO remove, this was replaced by interactive SignalPlot object 
        # axt.plot([signal.peak_time, signal.peak_time], [-signal.peak_amplitude, signal.peak_amplitude], '--', color='red')
        # for (time, amplitude) in zip(signal.peak_time, signal.peak_amplitude):
        #     axt.text(time, -amplitude*1.1, s=f"{time:.2e}")
        
        # -------- Plot start time used for FFT
        axt.plot([signal.time[signal.fft_start_index], signal.time[signal.fft_start_index]], 
                 [-np.max(signal.peak_amplitude), np.max(signal.peak_amplitude)], '-.', color='black', label=f'{label} FFT start time', alpha=0.5)
        axt.set(xlabel=f'Time ({signal.t_unit})', ylabel=f'{label} Signal ({signal.d_unit})')
        axt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # ------- Plot peak widths 
        if plot_peak_width:
            axt.hlines(signal.peak_width[1], signal.time[np.int32(signal.peak_width[2])], signal.time[np.int32(signal.peak_width[3])])

        if tlim is not None:
            axt.xlim(tlim)
        axt.legend()

        # -------- Plot frequency spectrum
        axf.set(xlabel=f"Frequency", ylabel="Magnitude")
        axf.plot(signal.fft_frequency, signal.fft_magnitude, label=f'{label} FFT', marker='.')
        axf.ticklabel_format(style='sci', axis='x')
        axf.set(xlim=(0, 2 * signal.fft_frequency[np.argmax(signal.fft_magnitude)]))
        # axf.xlim
        return axt, axf
    
    
class SignalPlot:
    def __init__(self):
        self.fig, (self.axtime, self.axfrequency) = plt.subplots(
            nrows=2, sharex='none', tight_layout=True
        )
        self.signals = []
        self.click_markers = []  # For auto-generated markers from clicks
        self.manual_markers = []  # For manually added markers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def add_signal(self, signal: Signal, label="", colors=None, 
                   plot_envelope=True, plot_waveform=True):
        """Add a signal to the plot"""
        self.signals.append({
            'signal': signal,
            'label': label,
            'colors': colors,
            'plot_envelope': plot_envelope,
            'plot_waveform': plot_waveform
        })
        self._redraw()
    
    def add_manual_marker(self, x_position, label=None, color='red', 
                          linestyle='--', linewidth=2, alpha=0.7):
        """
        Add a manual vertical marker to the plot
        
        Parameters:
        -----------
        x_position : float
            Time position for the vertical line
        label : str, optional
            Text label to display at the marker
        color : str, optional
            Color of the marker line and text
        linestyle : str, optional
            Line style for the marker
        linewidth : float, optional
            Width of the marker line
        alpha : float, optional
            Transparency of the marker
        """
        marker = {
            'type': 'manual',
            'x': x_position,
            'style': {
                'color': color,
                'linestyle': linestyle,
                'linewidth': linewidth,
                'alpha': alpha
            },
            'label': label
        }
        self.manual_markers.append(marker)
        self._redraw()
        return marker  # Return marker reference for potential removal
    
    def remove_marker(self, marker_reference):
        """Remove a specific marker from the plot"""
        if marker_reference in self.click_markers:
            self.click_markers.remove(marker_reference)
        elif marker_reference in self.manual_markers:
            self.manual_markers.remove(marker_reference)
        self._redraw()
    
    def clear_markers(self, manual_only=False):
        """Clear all markers from the plot"""
        if manual_only:
            self.manual_markers.clear()
        else:
            self.click_markers.clear()
            self.manual_markers.clear()
        self._redraw()
    
    def _find_closest_curve(self, click_time, click_y):
        """Determine which signal's curve is closest to the click position"""
        closest_signal = None
        min_distance = float('inf')
        active_curve_type = None  # 'waveform' or 'envelope'
        
        for sig_data in self.signals:
            signal = sig_data['signal']
            time = signal.time
            idx = np.searchsorted(time, click_time)
            idx = max(0, min(idx, len(time)-1))  # Ensure within bounds
            
            # Check waveform if plotted
            if sig_data['plot_waveform']:
                waveform_dist = abs(signal.data[idx] - click_y)
                if waveform_dist < min_distance:
                    min_distance = waveform_dist
                    closest_signal = sig_data
                    active_curve_type = 'waveform'
            
            # Check envelope if plotted
            if sig_data['plot_envelope']:
                envelope_dist = abs(signal.amplitude_envelope[idx] - click_y)
                if envelope_dist < min_distance:
                    min_distance = envelope_dist
                    closest_signal = sig_data
                    active_curve_type = 'envelope'
        
        return closest_signal, active_curve_type
    
    def _find_closest_peak(self, signal_data, curve_type, click_time, search_window=0.1):
        """
        Find the closest peak to the clicked time on the specified curve of the signal
        """
        signal = signal_data['signal']
        idx = np.searchsorted(signal.time, click_time)
        window_samples = int(search_window * signal.sample_frequency)
        
        start_idx = max(0, idx - window_samples)
        end_idx = min(len(signal.time)-1, idx + window_samples)
        
        # Determine which data to use based on clicked curve type
        if curve_type == 'envelope':
            data = signal.amplitude_envelope
            min_height = 0.1 * np.max(data)
        else:  # waveform
            data = signal.data
            min_height = 0.2 * (np.max(data) - np.min(data))
        
        # Find peaks with adaptive parameters
        peak_indices, _ = spsignal.find_peaks(
            data[start_idx:end_idx],
            height=min_height,
            prominence=0.3,
            width=3,
            distance=10
        )
        peak_indices += start_idx
        
        if len(peak_indices) == 0:
            # If no peaks found, use maximum point in window
            peak_indices = [np.argmax(data[start_idx:end_idx]) + start_idx]
        
        # Find closest peak to click
        closest_idx = peak_indices[np.argmin(np.abs(signal.time[peak_indices] - click_time))]
        return signal.time[closest_idx], data[closest_idx], signal
    
    def _on_click(self, event):
        if event.inaxes != self.axtime:
            return
        
        click_time = event.xdata
        click_y = event.ydata
        
        # First find which curve is closest to the click
        closest_signal_data, curve_type = self._find_closest_curve(click_time, click_y)
        if closest_signal_data is None:
            return
        
        # Then find the closest peak on that specific curve
        try:
            peak_time, peak_amp, signal = self._find_closest_peak(
                closest_signal_data, curve_type, click_time
            )
        except Exception as e:
            print(f"Error finding peak: {e}")
            return
        
        # Clear previous auto markers (keep manual ones)
        self.click_markers.clear()
        
        # Create marker style based on the signal
        marker_color = closest_signal_data['colors'][0] if closest_signal_data['colors'] else 'red'
        
        # Add vertical line marker
        line_style = {
            'color': marker_color,
            'linestyle': '--',
            'linewidth': 2,
            'alpha': 0.7
        }
        self.axtime.axvline(peak_time, **line_style)
        self.click_markers.append({
            'type': 'line',
            'x': peak_time,
            'style': line_style
        })
        
        # Add text annotation
        text_style = {
            's': f"{closest_signal_data['label']}\nt={peak_time:.2e}\nA={peak_amp:.2e}",
            'ha': 'center',
            'va': 'bottom',
            'color': marker_color,
            'alpha': 0.9,
            'bbox': dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7)
        }
        self.axtime.text(peak_time, peak_amp, **text_style)
        self.click_markers.append({
            'type': 'text',
            'x': peak_time,
            'y': peak_amp,
            'style': text_style
        })
        
        self.fig.canvas.draw()
    
    def _redraw(self):
        """Clear and redraw all signals and markers"""
        self.axtime.clear()
        self.axfrequency.clear()
        
        # Redraw signals
        for sig_data in self.signals:
            Signal._plot_helper(
                sig_data['signal'],
                self.axtime,
                self.axfrequency,
                label=sig_data['label'],
                colors=sig_data['colors'],
                plot_envelope=sig_data['plot_envelope'],
                plot_waveform=sig_data['plot_waveform']
            )
        
        # Redraw all markers (both click and manual)
        for marker in self.click_markers + self.manual_markers:
            if marker['type'] in ('line', 'manual'):
                self.axtime.axvline(marker['x'], **marker['style'])
                if 'label' in marker and marker['label']:
                    self.axtime.text(
                        marker['x'], 
                        self.axtime.get_ylim()[1] * 0.9,  # Place near top
                        marker['label'],
                        ha='center',
                        color=marker['style']['color'],
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                    )
            elif marker['type'] == 'text':
                self.axtime.text(marker['x'], marker['y'], **marker['style'])
        
        self.fig.canvas.draw()
    
    def show(self):
        plt.show()

        
if __name__ == '__main__':
    ...