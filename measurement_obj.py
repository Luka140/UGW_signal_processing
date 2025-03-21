import pathlib
import pandas as pd
import scipy.interpolate as interpolate 
import scipy.signal as spsignal 
import matplotlib.pyplot as plt 
import numpy as np 
from signal_obj import Signal 
from data_loading import load_signals_abaqus, load_signals_SINTEG
from typing import Collection


class Measurement:
    def __init__(self, tx_pos, 
                 rx_pos: Collection[Collection[float]] | Collection[float] , 
                 tx_signal: Signal,
                 rx_signal: Collection[Signal]| Signal, 
                 dispersion_curves=None):
        # TODO allow measurement with / without tx signal
        # TODO and allow for comparisons to tx signal
        self._valid_input(rx_signal, rx_pos)

        self.transmitter_position = np.array(tx_pos)
        self.transmitted_signal = tx_signal
        self.received_signals = rx_signal
        
        if type(rx_signal) != Signal:
            self.receiver_positions = [np.array(pos) for pos in rx_pos]

        self.dispersion_curves = dispersion_curves

    def compensate_dispersion(self):
        signal_fft = self.received_signals.fft_output
        fft_freqs  = self.received_signals.fft_frequency
        
        first_peak_time = min(self.received_signals.peak_time)
        # Window FFT around this time? 
        estimated_velocity = np.linalg.norm(self.receiver_positions - self.transmitter_position)

        lowest_vel_error = np.inf
        lowest_vel_index = None 
        # TODO check units
        # TODO maybe simply apply a filter for all frequencies and then add
        # TODO what if you do a couple very narrow bandpass filters and only use those
                # There is probably a more efficient way of doing this mathematically.
        for mode in self.dispersion_curves:
            velocities = mode(fft_freqs)            
            
        # phase_shift = -2 * np.pi * frequencies * d / vp


    def compare_signals(self, base_index: int, comparison_indices: int | Collection[int], tlim=None, mode='signal'):
        if type(comparison_indices) == int:
            comparison_indices = [comparison_indices]

        base_signal = self.received_signals[base_index]

        for rx_index in comparison_indices:
            comparison_signal = self.received_signals[rx_index]

            scaling_factor = np.max(base_signal.amplitude_envelope) / np.max(comparison_signal.amplitude_envelope)
            scaled_signal = Signal(comparison_signal.time,comparison_signal.data * scaling_factor,comparison_signal.t_unit, comparison_signal.d_unit)

            fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
            axtime, axfrequency = Signal._plot_helper(base_signal, axtime, axfrequency, tlim, label=f"Base_sig ", colors=['c', 'blue'])
            axtime, axfrequency = Signal._plot_helper(scaled_signal, axtime, axfrequency, tlim, label=f"Scaled_comp_sig{rx_index}", colors=['yellow', 'orange'])

            fig2 = plt.figure()
            ax = fig2.add_axes(111)
            correlation_envelope    = spsignal.correlate(base_signal.amplitude_envelope, scaled_signal.amplitude_envelope, mode="full")
            correlation_data        = spsignal.correlate(base_signal.data, scaled_signal.data, mode="full")
            lags = spsignal.correlation_lags(base_signal.data.size, scaled_signal.data.size, mode="full")
            # print(correlation.size // 2 - np.argmax(correlation), lags[np.argmax(correlation)])
            ax.plot(lags, correlation_data, alpha=0.5, color='b', label='Scaled signal')
            ax.plot(lags, correlation_envelope * (np.max(correlation_data) / np.max(correlation_envelope)), alpha=0.5, color='r', label='Scaled signal envelope')
            ax.set(xlabel=f"Indices of shift", ylabel="Correlation")
            ax.legend()

            # TODO can this be improved by correlating raw analytical complex signal rather than envelope??
            # TODO also compare from Tx signal, by taking t=0 as t for ~weighted average t of excitation signal pulse

            time_shift_data     = abs(lags[np.argmax(correlation_data)]) / base_signal.sample_frequency
            time_shift_envelope = abs(lags[np.argmax(correlation_envelope)]) / base_signal.sample_frequency
            time_shift_peaks    = abs(comparison_signal.peak_time[0] - base_signal.peak_time[0])

            print(f"ToF for max correlation of scaled signal: {time_shift_data:2e} {base_signal.t_unit}")
            print(f'           "            of scaled envelope: {time_shift_envelope:.2e} {base_signal.t_unit}')
            print(f"ToF based on first envelope peaks: {time_shift_peaks:.2e} {base_signal.t_unit}")
            distance = np.linalg.norm(self.receiver_positions[base_index] - self.receiver_positions[rx_index])
            print(f"Distance travelled {distance}")
            print(f"Wave velocity: {distance / time_shift_data:.2f} (corr. scaled signal) - {distance / time_shift_envelope:.2f} (corr. scaled envelope) - {distance / time_shift_peaks:.2f} (peaks)")

            fig, (axtime2, axfrequency2) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
            axtime2.set(title="Shifted signals according to envelope correlation")
            scaled_shifted_signal = Signal(scaled_signal.time - time_shift_envelope, scaled_signal.data, scaled_signal.t_unit, scaled_signal.d_unit)
            axtime2, axfrequency2 = Signal._plot_helper(base_signal, axtime2, axfrequency2, tlim, label=f"Base_sig ", colors=['c', 'blue'])
            axtime2, axfrequency2 = Signal._plot_helper(scaled_shifted_signal, axtime2, axfrequency2, tlim, label=f"Shifted_scaled_comp_sig{rx_index}", colors=['yellow', 'orange'])


            plt.show()


    def _valid_input(self, rx_pos, rx_signal):
        # TODO not exhaustive
        valid = True 
        reason = ""
        if type(rx_signal) == Signal:
            if len(rx_pos) != 2:            
                valid=False; reason += f" The rx position should be provided as [x, y]. Given size: {len(rx_pos)}"
            if type(rx_signal[0]) != float: 
                valid=False; reason += f" The rx coordinates should be floats, not {type(rx_signal[0])}"
        elif isinstance(rx_signal, Collection):
            if len(rx_signal) != len(rx_pos):
                valid=False
                reason += f" rx_signal and rx_pos should have the same number of entries. Not: {len(rx_signal)} vs {len(rx_pos)}"
        else:
            valid=False 
            reason += f" rx_signal should either be a collection of Signals, or a single Signal. Not: {type(rx_signal)}"

        if not valid:
            raise TypeError(reason)


if __name__ == '__main__':
    data_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 's355j2_dispersion_curves'
    print(data_dir)
    modes = []
    for path in data_dir.glob('*.txt'):
        curves = pd.read_csv(path)
        
        # In the exported csvs, "Attenuation" is always the last column of a mode
        modes_in_class = sum([1 for col_name in curves.columns if "Attenuation" in col_name])
        columns_per_mode = len(curves.columns) // modes_in_class
        mode_freq_headers = curves.columns[0::columns_per_mode]
        mode_phase_vel_headers = curves.columns[1::columns_per_mode]
        # TODO make this work for anisotropic 
        for i, mode_freq in enumerate(mode_freq_headers):
            freq_vel = curves[[mode_freq, mode_phase_vel_headers[i]]].dropna()
            freq_vel = freq_vel.sort_values(mode_freq)

            spline = interpolate.CubicSpline(freq_vel[mode_freq], freq_vel[mode_phase_vel_headers[i]], extrapolate=False)
            modes.append(spline)
        
    # data_dir_ab = pathlib.Path(__file__).parent / 'data' / 'measurement_data'/ 'abaqus_test_steel'
    # sig = load_signals_abaqus(data_dir_ab)[0]

    data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "measurements_0"
    avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={31}, plot_outliers=True)
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "measurements_90"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={33}, plot_outliers=True)
    
    avg_signals = [sig.zero_average_signal().bandpass(30e3, 90e3) for sig in avg_signals]
    measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], avg_signals[-1], avg_signals[:-1], modes)
    measurement.compare_signals(1, 2)


    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "longer_measurements_1"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={}, plot_outliers=True)

    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 90e3) for sig in avg_signals]
    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (182e-3, 0.)], avg_signals[-1], avg_signals[:-1], modes)
    # measurement.compare_signals(1, 2)


    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "longer_measurements_2"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={}, plot_outliers=False)

    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 90e3) for sig in avg_signals]
    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (305e-3, 0.)], avg_signals[-1], avg_signals[:-1], modes)
    # measurement.compare_signals(1, 2)


    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "elevated"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={10, 19, 23, 36, 48, 54, 58}, plot_outliers=True)

    # avg_signals = [sig.zero_average_signal().bandpass(40e3, 80e3) for sig in avg_signals]
    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], avg_signals[-1], avg_signals[:-1], modes)
    # measurement.compare_signals(0, 2)

    # avg_signals[2].get_stfft(1e-4)
