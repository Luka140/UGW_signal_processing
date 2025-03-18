from signal_obj import Signal 
import pathlib
import pandas as pd
import scipy.interpolate as interpolate 
from data_loading import load_signals_abaqus


class Measurement:
    def __init__(self, tx_pos, rx_pos, rx_signal, dispersion_curves):
        
        self.transmitter_position = tx_pos
        self.receiver_position = rx_pos
        self.received_signal = rx_signal

        self.dispersion_curves = dispersion_curves

    def compensate_dispersion(self):
        ...
        


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
        # TODO it seems like these are ordered by velocity, not frequency. Sort them first 
        for i, mode_freq in enumerate(mode_freq_headers):
            print(mode_freq)
            freq = curves[mode_freq].dropna()
            print(((freq.to_numpy()[1:]-freq.to_numpy()[:-1])<0).any())  
            spline = interpolate.CubicSpline(freq, curves[mode_phase_vel_headers[i]].dropna(), extrapolate=False)
            modes.append(spline)
        
    data_dir_ab = pathlib.Path(__file__).parent / 'data' / 'abaqus_test_steel'
    sig = load_signals_abaqus(data_dir_ab)[0]
    
    Measurement((0,0), (200,0), sig, modes)
