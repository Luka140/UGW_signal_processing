from signal_obj import Signal 
import pathlib
import pandas as pd
import scipy.interpolate as interpolate 
import numpy as np 
from data_loading import load_signals_abaqus


class Measurement:
    def __init__(self, tx_pos, rx_pos, rx_signal, dispersion_curves):
        
        self.transmitter_position = np.array(tx_pos)
        self.receiver_position = np.array(rx_pos)
        self.received_signal = rx_signal

        self.dispersion_curves = dispersion_curves

    def compensate_dispersion(self):
        signal_fft = self.received_signal.fft_output
        fft_freqs  = self.received_signal.fft_frequency
        
        first_peak_time = min(self.received_signal.peak_time)
        # Window FFT around this time? 
        estimated_velocity = np.linalg.norm(self.receiver_position - self.transmitter_position)

        lowest_vel_error = np.inf
        lowest_vel_index = None 
        # TODO check units
        # TODO maybe simply apply a filter for all frequencies and then add
        # TODO what if you do a couple very narrow bandpass filters and only use those
                # There is probably a more efficient way of doing this mathematically.
        for mode in self.dispersion_curves:
            velocities = mode(fft_freqs)            
            
        # phase_shift = -2 * np.pi * frequencies * d / vp

        


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
        
    data_dir_ab = pathlib.Path(__file__).parent / 'data' / 'measurement_data'/ 'abaqus_test_steel'
    sig = load_signals_abaqus(data_dir_ab)[0]
    
    Measurement((0,0), (0.200,0), sig, modes)
