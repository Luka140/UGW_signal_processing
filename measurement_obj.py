import pathlib
import pandas as pd
import scipy.interpolate as interpolate 
import scipy.signal as spsignal 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.fft as spfft 
from typing import Collection
from signal_obj import Signal, SignalPlot 
from data_loading import load_signals_abaqus, load_signals_SINTEG
from dispersiondata_obj import DispersionData


class Measurement:
    def __init__(self, tx_pos, 
                 rx_pos: Collection[Collection[float]] | Collection[float], 
                 tx_signal: Signal,
                 rx_signal: Collection[Signal] | Signal, 
                 dispersion_curves: None | DispersionData = None):

        self.transmitter_position = np.array(tx_pos)
        self.transmitted_signal = tx_signal
        self.received_signals = rx_signal
        
        # Store positions appropriately based on input type
        if isinstance(rx_signal, Signal):
            self.receiver_positions = [np.array(rx_pos)]
            self.received_signals = [rx_signal]
        else:
            self.receiver_positions = [np.array(pos) for pos in rx_pos]

        self._valid_input(self.received_signals, self.receiver_positions)
        self.dispersion_curves = dispersion_curves

    def plot_envelopes(self):
        plot = SignalPlot()
        if self.transmitted_signal is not None:
            plot.add_signal(self.transmitted_signal, label="tx", colors=['red'], plot_waveform=False)
        for i, sig in enumerate(self.received_signals):
            plot.add_signal(sig, label=f"Rx sig{i}", plot_waveform=False)

        plot.axtime.set_title("Signal envelopes")
        plot.axtime.set(xlabel="Time (s)")
        
        plot.show()

    def compare_signals(self, base_index: int | str = 0, comparison_indices: int | Collection[int] = None, tlim=None, mode='signal', plot_correlation=False):
        """
        Compare signals with options to use transmitted signal as reference.
        
        Parameters:
        - base_index: Either 'tx' to use transmitted signal as reference, or an int for receiver index
        - comparison_indices: Receiver indices to compare against base signal
        """
        # Handle default case where we want to compare all received signals to tx
        if comparison_indices is None:
            if isinstance(self.received_signals, Signal):
                comparison_indices = [0]
            else:
                comparison_indices = range(len(self.received_signals))

        if isinstance(comparison_indices, int):
            comparison_indices = [comparison_indices]

        # Get base signal
        if base_index == 'tx':
            base_signal = self.transmitted_signal
            base_pos = self.transmitter_position
        else:
            base_signal = self.received_signals[base_index]
            base_pos = self.receiver_positions[base_index]

        for rx_index in comparison_indices:
            comparison_signal = self.received_signals[rx_index]
            comp_pos = self.receiver_positions[rx_index]

            scaling_factor = np.max(base_signal.amplitude_envelope) / np.max(comparison_signal.amplitude_envelope)
            scaled_signal = Signal(comparison_signal.time,
                                 comparison_signal.data * scaling_factor,
                                 comparison_signal.t_unit, 
                                 comparison_signal.d_unit)

                    
            # Create the SignalPlot object
            signal_plot = SignalPlot()

            # Add the base signal with specified styling
            signal_plot.add_signal(
                base_signal,
                label=f"Base_sig {'tx' if base_index == 'tx' else base_index}",
                colors=['c', 'blue']
            )

            # Add the scaled comparison signal with specified styling
            signal_plot.add_signal(
                scaled_signal,
                label=f"Scaled_comp_sig{rx_index}",
                colors=['yellow', 'orange']
            )

            # Customize the plot if needed
            signal_plot.axtime.set_title("Signal Comparison")
            signal_plot.axfrequency.set_xlim(0, 2 * base_signal.fft_frequency[np.argmax(base_signal.fft_magnitude)])

            # Show the interactive plot
            signal_plot.show()


            # Correlation calculations
            correlation_envelope = spsignal.correlate(base_signal.amplitude_envelope, 
                                                    scaled_signal.amplitude_envelope, 
                                                    mode="full")
            lags = spsignal.correlation_lags(base_signal.data.size, 
                                           scaled_signal.data.size, 
                                           mode="full")
            if plot_correlation:
                fig2 = plt.figure()
                ax = fig2.add_axes(111)
                ax.plot(lags, correlation_envelope)
                ax.set(xlabel=f"Indices of shift", ylabel="Correlation")

            # Time shift calculations
            time_shift_envelope = abs(lags[np.argmax(correlation_envelope)]) / base_signal.sample_frequency
            time_shift_peaks = abs(comparison_signal.peak_time[0] - base_signal.peak_time[0])
            time_shift_waveform_start = abs(comparison_signal.time[comparison_signal.fft_start_index] - 
                                          base_signal.time[base_signal.fft_start_index])

            print(f'\nToF for max correlation of scaled envelope: {time_shift_envelope:.2e} {base_signal.t_unit}')
            print(f"ToF based on max envelope peaks: {time_shift_peaks:.2e} {base_signal.t_unit}")
            print(f"ToF based on 2% threshold crossing: {time_shift_waveform_start:.2e} {base_signal.t_unit}")
            
            distance = np.linalg.norm(base_pos - comp_pos)
            print(f"Distance travelled: {distance}")
            print(f"Wave velocity: {distance / (time_shift_envelope+1e-14):.2f} (corr. scaled envelope) - "
                 f"{distance / (time_shift_peaks+1e-14):.2f} (peaks) - "
                 f"{distance / (time_shift_waveform_start+1e-14):.2f} (Initial threshold)")

            sh0_tag = None 
            for sh_tag in ["ASH0", "SSH0", "BSH0"]:
                if sh_tag in self.dispersion_curves.get_available_modes():
                    sh0_tag = sh_tag
                    break
                    
            if sh0_tag is None:
                print("No SH0 mode found in dispersion curves")

            if "A0" in self.dispersion_curves.get_available_modes() and "S0" in self.dispersion_curves.get_available_modes():
                # Get dispersion curves for A0 and S0
                a0_tag = "A0"
                s0_tag = "S0"
            elif "B0" in self.dispersion_curves.get_available_modes() and "B1" in self.dispersion_curves.get_available_modes():
                # Get dispersion curves for B0 and B1
                a0_tag = "B0"
                s0_tag = "B1"
            else:
                print("No A0, S0, B0, or B1 mode found in dispersion curves")

            signal_names = ["Base signal", "Comparison signal"]
            for i, sig in enumerate([base_signal, comparison_signal]):

                print(f"\nCharacteristic frequency {signal_names[i]}: {sig.characteristic_frequency/10**3:.2f} kHz")
                base_vg_a0, base_vg_s0, base_vg_sh0 = self.dispersion_curves.get_value((a0_tag, s0_tag, sh0_tag), sig.characteristic_frequency/10**3, target_header="Energy velocity")
                print(f"Energy velocity dispersion curve - A0/B0: {base_vg_a0:.2f} m/ms, S0/B1: {base_vg_s0:.2f} m/ms, {sh0_tag}: {base_vg_sh0:.2f} m/ms")

            # print(f"\nCharacteristic frequency comp signal: {comparison_signal.characteristic_frequency/10**3:.2f} kHz")
            # comp_vg_a0, comp_vg_s0, comp_vg_sh0 = self.dispersion_curves.get_value(("A0", "S0", sh0_tag), comparison_signal.characteristic_frequency/10**3, target_header="Energy velocity")
            # print(f"Energy velocity dispersion curve - A0: {comp_vg_a0:.2f} m/ms, S0: {comp_vg_s0:.2f} m/ms, {sh0_tag}: {comp_vg_sh0:.2f} m/ms")

            if plot_correlation:
                # Plot shifted signals
                fig, (axtime2, axfrequency2) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
                axtime2.set(title="Shifted signals according to envelope correlation")
                scaled_shifted_signal = Signal(scaled_signal.time - time_shift_envelope, 
                                            scaled_signal.data, 
                                            scaled_signal.t_unit, 
                                            scaled_signal.d_unit)
                axtime2, axfrequency2 = Signal._plot_helper(base_signal, axtime2, axfrequency2, tlim, 
                                                        label=f"Base_sig {'tx' if base_index == 'tx' else base_index}", 
                                                        colors=['c', 'blue'])
                axtime2, axfrequency2 = Signal._plot_helper(scaled_shifted_signal, axtime2, axfrequency2, tlim, 
                                                        label=f"Shifted_scaled_comp_sig{rx_index}", 
                                                        colors=['yellow', 'orange'])

            plt.show()

    
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