import pathlib
import pandas as pd
import scipy.signal as spsignal
import scipy.interpolate as interpolate
import scipy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Iterable

from data_loading import load_signals_labview


def average_signals(signals: Iterable[np.ndarray]) -> list[np.ndarray]:
    min_time, max_time = min([sig[0, 0] for sig in signals]), max([sig[-1, 0] for sig in signals])
    common_time = np.linspace(min_time, max_time, int(signals[0].shape[0]))

    # Check nr of channels by checking dimensions of first signal -1 for time axis
    channels = signals[0].shape[1] - 1
    averaged_channels = []
    for j in range(channels):
        interpolated_signals = []
        for sig in signals:
            time    = sig[:, 0]
            voltage = sig[:, 1+j]

            # Interpolate the signal to the common time points, and store the interpolated points
            interp_func = interpolate.interp1d(time, voltage, kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolated_voltage = interp_func(common_time)
            interpolated_signals.append(interpolated_voltage)

        avg_signal = np.mean(np.stack(interpolated_signals, axis=0), axis=0)
        averaged_channels.append(np.stack((common_time, avg_signal)).T)
    return averaged_channels

def get_sample_frequency(signal: np.ndarray) -> tuple[float, float | np.floating]:
    avg_sample_interval = np.mean(signal[1:, 0] - signal[:-1, 0])
    sample_frequency = 1 / avg_sample_interval
    return sample_frequency, avg_sample_interval


def get_fft(signal: np.ndarray, positive_half: bool=True):
    voltage_data = signal[:, 1]  # Extract voltage data (second column)
    signal_samples = len(voltage_data)  # Number of data points

    _, avg_sample_interval = get_sample_frequency(signal)

    # Compute FFT
    fft_output = fft.fft(voltage_data)
    fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT
    fft_freq = fft.fftfreq(signal_samples, avg_sample_interval)  # Frequency axis

    if positive_half:
        mask = fft_freq >= 0
        fft_freq = fft_freq[mask]
        fft_magnitude = fft_magnitude[mask]
    return fft_freq, fft_magnitude



def get_signal_envelope(signal: np.ndarray, sample_frequency: None | float =None):
    if sample_frequency is None:
        sample_frequency, _ = get_sample_frequency(signal)

    analytic_signal = spsignal.hilbert(signal[:,1])
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_frequency
    return amplitude_envelope, instantaneous_frequency, instantaneous_phase


def apply_bandpass(signal, lower_f, upper_f):
    sample_freq, _ = get_sample_frequency(signal)
    bandpass_filter = spsignal.butter(2, Wn=(lower_f, upper_f), btype='bandpass', output='sos', fs=sample_freq)
    if signal.ndim > 1:
        filtered_signal = np.stack((signal[:,0], spsignal.sosfilt(bandpass_filter, signal[:,1]))).T
    else:
        filtered_signal = spsignal.sosfilt(bandpass_filter, signal)

    return filtered_signal


if __name__ == '__main__':
    data_dir = pathlib.Path(__file__).parent / 'data' / 'test_Abishay_example'
    signal_list, signal_units = load_signals_labview(data_dir)
    avg_channels = average_signals(signal_list)

    # Relative duration of shown signal after the largest amplitude peak
    times_peak = 1.5
    # Get index of t=0 uglily
    start_index = np.where(avg_channels[0][:,0]>= 0)[0][0]

    for i, channel in enumerate(avg_channels):

        peak_index = np.argmax(np.abs(channel[:,1]))
        end_index = int(min(peak_index * times_peak, channel.shape[0]))

        plt.plot(channel[start_index:end_index,0], channel[start_index:end_index,1])
        xlab = list(signal_units.index)[0]
        ylab = list(signal_units.index)[i+1]
        plt.xlabel(f"{xlab} {signal_units[xlab]}")
        plt.ylabel(f"{ylab} {signal_units[ylab]}")

        plt.show()


    freq, mag = get_fft(avg_channels[1])
    bandpassed_signal = apply_bandpass(avg_channels[1],0.9e3, 1.1e3)
    freq_bandpassed, mag_bandpassed = get_fft(bandpassed_signal)

    plt.plot(freq, mag, label="Original")
    plt.plot(freq_bandpassed, mag_bandpassed, label="Bandpassed")
    plt.xlim(-100, 2000)
    plt.legend()
    plt.show()

    amp_env, sign_freq, sig_phase = get_signal_envelope(bandpassed_signal)
    plt.plot(bandpassed_signal[:500, 0], amp_env[:500])
    plt.plot(bandpassed_signal[:500, 0], bandpassed_signal[:500, 1])
    plt.show()

    plt.plot(bandpassed_signal[1:, 0], sign_freq/(2*np.pi))
    plt.show()

    plt.plot(bandpassed_signal[start_index:start_index+300, 0], sig_phase[start_index:start_index+300])
    plt.show()

    # TODO get ToF
    # TODO implement comparison with dispersion curve
    # TODO give measurements a position

