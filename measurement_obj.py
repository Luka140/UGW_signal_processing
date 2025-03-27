import pathlib
import pandas as pd
import scipy.interpolate as interpolate 
import scipy.signal as spsignal 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.fft as spfft 
from typing import Collection
from signal_obj import Signal 
from data_loading import load_signals_abaqus, load_signals_SINTEG
from dispersiondata_obj import DispersionData


class Measurement:
    def __init__(self, tx_pos, 
                 rx_pos: Collection[Collection[float]] | Collection[float] , 
                 tx_signal: Signal,
                 rx_signal: Collection[Signal]| Signal, 
                 dispersion_curves: None|DispersionData=None):
        # TODO allow measurement with / without tx signal
        # TODO and allow for comparisons to tx signal
        self._valid_input(rx_signal, rx_pos)

        self.transmitter_position = np.array(tx_pos)
        self.transmitted_signal = tx_signal
        self.received_signals = rx_signal
        
        if type(rx_signal) != Signal:
            self.receiver_positions = [np.array(pos) for pos in rx_pos]

        self.dispersion_curves = dispersion_curves

    def _remove_dispersion_known_distance(self, signal_obj: Signal, k_omega, omega_0, distance):
        n = len(signal_obj.data)
        
       # TODO I think for this to work, t=0 needs to be set correctly

        # Add zero-padding to improve frequency resolution
        n_pad = 4 * n  # Pad to 4x original length
        padded_signal = np.pad(signal_obj.data, (0, n_pad - n), 'constant')
        
        dt = 1 / signal_obj.sample_frequency
        fft_signal = np.fft.fft(padded_signal)
        freqs = np.fft.fftfreq(n_pad, dt) * 2 * np.pi  # Use padded length
        
        # Compute k2 = ½ * d²k/dω² (corrected numerical derivatives)
        omega_uniform = np.linspace(np.min(k_omega[:,0]), np.max(k_omega[:,0]), k_omega.shape[0])
        
        kinterp = interpolate.interp1d(k_omega[:,0], k_omega[:,1], kind='cubic')
        k_uniform = kinterp(omega_uniform)
        dk_domega = np.gradient(k_uniform, omega_uniform)
        dk2_domega = np.gradient(dk_domega, omega_uniform) / 2
        
        # dk_domega = np.gradient(k_omega[:,1], k_omega[:,0])  # Use interpolated k values
        # dk2_domega = np.gradient(dk_domega, k_omega[:,0]) / 2
        
        k2_interp = interpolate.interp1d(omega_uniform, dk2_domega, kind='cubic', fill_value=0)
        
        # Apply phase correction to entire bandwidth
        modified_spectrum = fft_signal.copy()
        
        # Get signal's dominant frequency range (e.g., ±3σ around ω₀)
        bandwidth = 3 * (omega_0 / 10)  # Adjust based on your signal
        omega_mask = (freqs >= omega_0 - bandwidth) & (freqs <= omega_0 + bandwidth)
        
        # Apply correction only where dispersion data is valid
        valid_freq_mask = (freqs >= np.min(k_omega[:,0])) & (freqs <= np.max(k_omega[:,0]))
        omega_mask = omega_mask & valid_freq_mask
        
        modified_spectrum[omega_mask] *= np.exp(
            1j * k2_interp(freqs[omega_mask]) * (freqs[omega_mask] - omega_0)**2 * distance
        )

        # ------ Plot for debugging 
        # phase_term = k2_interp(freqs[omega_mask]) * (freqs[omega_mask] - omega_0)**2 * distance
        # plt.plot(freqs[omega_mask], phase_term)
        # plt.title("Applied Phase Correction")
        # plt.show()

        
        # Inverse FFT and remove padding
        compensated_signal = np.fft.ifft(modified_spectrum)[:n]  # Truncate to original length
        return np.real(compensated_signal)

    
    def compensate_dispersion(self, center_frequency, mode):

        omega0 = center_frequency * 2 * np.pi 
        compensated_signals = []
        for i in range(len(self.received_signals)):
            distance = np.linalg.norm(self.receiver_positions[i] - self.transmitter_position) 
            curves = self.dispersion_curves.get_dispersion_curves(frequency_range=(0, center_frequency * 3 /1000))
            k_omega = curves[mode][['f (kHz)','Wavenumber (rad/mm)']].to_numpy(dtype=float)
            # Convert to angular frequency and base units 
            k_omega[:,0] *= 2 * np.pi * 1000 
            k_omega[:,1] *= 1000 
            # TODO unit conversions in dispersiondata object 
            
            new_signal_data = self._remove_dispersion_known_distance(self.received_signals[i], k_omega, omega0, distance)
            compensated_signals.append(Signal(self.received_signals[i].time, new_signal_data, self.received_signals[i].t_unit, self.received_signals[i].d_unit)) 
        return compensated_signals



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


            # ----------------------Get the time of the second peak for both signals
                    # base_second_peak_time = np.sort(base_signal.peak_time)[1]   if len(base_signal.peak_time)   > 1 else base_signal.time[-1]
                    # comp_second_peak_time = np.sort(scaled_signal.peak_time)[1] if len(scaled_signal.peak_time) > 1 else scaled_signal.time[-1]
                    # second_peak_time = max(base_second_peak_time, comp_second_peak_time)

                    # # Trim the signals up to their second peaks
                    # base_envelope_trimmed = base_signal.get_trimmed_signal(0, second_peak_time)
                    # comp_envelope_trimmed = scaled_signal.get_trimmed_signal(0, second_peak_time)
                    
                    # # Calculate correlation using only the trimmed portions
                    # correlation_envelope = spsignal.correlate(base_envelope_trimmed.amplitude_envelope, comp_envelope_trimmed.amplitude_envelope, mode="full")
                    # lags = spsignal.correlation_lags(len(base_envelope_trimmed.time), len(comp_envelope_trimmed.time), mode="full")
                    
                    # fig0 = plt.figure()
                    # ax0 = fig0.add_axes(111)
                    # ax0.plot(base_envelope_trimmed.time, base_envelope_trimmed.amplitude_envelope, label='Base signal')
                    # ax0.plot(comp_envelope_trimmed.time, comp_envelope_trimmed.amplitude_envelope, label='Comp signal')
                    # ax0.set(title="Amplitude envelope sections used for correlation")
                    # plt.show()

            correlation_envelope    = spsignal.correlate(base_signal.amplitude_envelope, scaled_signal.amplitude_envelope, mode="full")
            lags = spsignal.correlation_lags(base_signal.data.size, scaled_signal.data.size, mode="full")
            # print(correlation.size // 2 - np.argmax(correlation), lags[np.argmax(correlation)])
            
            fig2 = plt.figure()
            ax = fig2.add_axes(111)
            # ax.plot(lags, correlation_data, alpha=0.5, color='b', label='Scaled signal')
            ax.plot(lags, correlation_envelope) #* (np.max(correlation_data) / np.max(correlation_envelope)), alpha=0.5, color='r', label='Scaled signal envelope')
            ax.set(xlabel=f"Indices of shift", ylabel="Correlation")

            # TODO can this be improved by correlating raw analytical complex signal rather than envelope??
            # TODO also compare from Tx signal, by taking t=0 as t for ~weighted average t of excitation signal pulse
            # TODO try to backpropagate signals to the tx?

            time_shift_envelope = abs(lags[np.argmax(correlation_envelope)]) / base_signal.sample_frequency
            time_shift_peaks    = abs(comparison_signal.peak_time[0] - base_signal.peak_time[0])
            time_shift_waveform_start = abs(comparison_signal.time[comparison_signal.fft_start_index] - base_signal.time[base_signal.fft_start_index])

            # print(f"\nToF for max correlation of scaled signal: {time_shift_data:2e} {base_signal.t_unit}")
            print(f'\nTof for max correlation of scaled envelope: {time_shift_envelope:.2e} {base_signal.t_unit}')
            print(f"ToF based on first envelope peaks: {time_shift_peaks:.2e} {base_signal.t_unit}")
            print(f"ToF based on 2% threshold crossing: {time_shift_waveform_start:.2e} {base_signal.t_unit}")
            distance = np.linalg.norm(self.receiver_positions[base_index] - self.receiver_positions[rx_index])
            print(f"Distance travelled {distance}")
            print(f"Wave velocity: {distance / (time_shift_envelope+1e-14):.2f} (corr. scaled envelope) - {distance / (time_shift_peaks+1e-14):.2f} (peaks) - {distance / time_shift_waveform_start+1e-14:.2f} (Initial threshold)")

            fig, (axtime2, axfrequency2) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
            axtime2.set(title="Shifted signals according to envelope correlation")
            scaled_shifted_signal = Signal(scaled_signal.time - time_shift_envelope, scaled_signal.data, scaled_signal.t_unit, scaled_signal.d_unit)
            axtime2, axfrequency2 = Signal._plot_helper(base_signal, axtime2, axfrequency2, tlim, label=f"Base_sig ", colors=['c', 'blue'])
            axtime2, axfrequency2 = Signal._plot_helper(scaled_shifted_signal, axtime2, axfrequency2, tlim, label=f"Shifted_scaled_comp_sig{rx_index}", colors=['yellow', 'orange'])

            # axtime2.vlines([second_peak_time], ymin=-np.max(np.abs(base_envelope_trimmed.amplitude_envelope)), ymax=np.max(np.abs(base_envelope_trimmed.amplitude_envelope)))

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
    dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    dispersion = DispersionData()
    for curves_file in dispersion_dir.glob('*.txt'):
        dispersion.merge(DispersionData(curves_file))
    print("Available modes:", dispersion.get_available_modes())

    
    # data_dir_ab = pathlib.Path(__file__).parent / 'data' / 'measurement_data'/ 'abaqus_test_steel'
    # sig = load_signals_abaqus(data_dir_ab)[0]

    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "measurements_0"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={31}, plot_outliers=True)
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "measurements_90"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={33}, plot_outliers=True)
    
    # avg_signals = [sig.zero_average_signal().bandpass(50e3, 70e3) for sig in avg_signals]
    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], avg_signals[-1], avg_signals[:-1])
    # measurement.compare_signals(0, 2)


    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "longer_measurements_1"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={}, plot_outliers=True)

    # avg_signals = [sig.zero_average_signal().bandpass(50e3, 70e3) for sig in avg_signals]
    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (182e-3, 0.)], avg_signals[-1], avg_signals[:-1])
    # measurement.compare_signals(0, 1)


    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "longer_measurements_2"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={}, plot_outliers=False)

    # avg_signals = [sig.zero_average_signal().bandpass(50e3, 70e3) for sig in avg_signals]
    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (305e-3, 0.)], avg_signals[-1], avg_signals[:-1])
    # measurement.compare_signals(1, 2)


    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "elevated"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={10, 19, 23, 36, 48, 54, 58}, plot_outliers=True)

    # avg_signals = [sig.zero_average_signal().bandpass(40e3, 80e3) for sig in avg_signals]
    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], avg_signals[-1], avg_signals[:-1])
    # measurement.compare_signals(0, 2)


    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_test_plate" / "sh_measurements_120khz"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={40,45}, plot_outliers=False)

    # avg_signals = [sig.zero_average_signal().bandpass(40e3, 80e3) for sig in avg_signals]
    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=avg_signals[-1], rx_signal=avg_signals[:-1], dispersion_curves=dispersion)
    # measurement.compare_signals(0,2)

    data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_test_plate" / "a_measurements_120khz"
    avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={40,45}, plot_outliers=False)

    avg_signals = [sig.zero_average_signal().bandpass(40e3, 80e3, order=2) for sig in avg_signals]
    measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=avg_signals[-1], rx_signal=avg_signals[:-1], dispersion_curves=dispersion)
    measurement.compare_signals(0,2)
    new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")


    
    # for i in range(len(avg_signals[:-1])):
    #     sig_og = avg_signals[i]
    #     sig_compensated = new_signals[i]
    #     plt.plot(sig_og.time, sig_og.data, label="Original", alpha=0.5)
    #     plt.plot(sig_compensated.time, sig_compensated.data, label="Compensated", alpha=0.5)
    #     plt.legend()
    #     plt.show()

    # fig2, (axtime2, axfrequency2) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(new_signals)):
    #         axtime2, axfrequency2 = Signal._plot_helper(new_signals[i], axtime2, axfrequency2,  label=f"sig{i}", plot_waveform=False)
    # plt.legend()
    # plt.show()

    avg_signals[2].get_stfft(1e-4)

    # ------------ CHECK ORDER OF ARIVAL
    fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    for i in range(len(avg_signals)):
            axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    plt.show()