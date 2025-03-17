import warnings
import pandas as pd
import numpy as np
import pathlib
from typing import Iterable
from scipy.interpolate import interpolate

def load_signals_labview(path):
    unit_row = 0
    print(path)
    signals = []
    prev_units = None
    for file in path.glob('*.csv'):
        csv_data = pd.read_csv(file, skip_blank_lines=True)
        units = csv_data.iloc[unit_row, :]
        if prev_units is not None and (units != prev_units).any():
            warnings.warn(f"The file at {file} has different units than the previous one ({prev_units}) vs ({units}). This will cause issues with averaging.")
        prev_units = units

        csv_data.drop(unit_row, inplace=True)
        signals.append(csv_data.to_numpy(dtype=float))
    print(f"warning: Units not considered currently.\nUnits: {prev_units}")
    return signals, prev_units

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


def load_signals_abaqus(path):
    signals = []
    for file in path.glob('*.csv'):
        csv_data = pd.read_csv(file, skip_blank_lines=True, header=None)
        signals.append(csv_data.to_numpy(dtype=float))
    return signals

if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent / 'data' / 'abaqus_test_steel'
    signal_list = load_signals_abaqus(data_dir)
