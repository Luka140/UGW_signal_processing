import pathlib
import pandas as pd
import scipy.signal as signal
import scipy.interpolate as interpolate
import numpy as np
import matplotlib.pyplot as plt
import warnings

def average_signals(signals, channels):

    min_time, max_time = min([sig[0, 0] for sig in signals]), max([sig[-1, 0] for sig in signals])
    common_time = np.linspace(min_time, max_time, int(signals[0].shape[0]*2.5))

    averaged_channels = []
    for i in range(channels):
        interpolated_signals = []
        for sig in signals:
            time    = sig[:, 0]
            voltage = sig[:, 1+i]
            interp_func = interpolate.interp1d(time, voltage, kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolated_voltage = interp_func(common_time)
            interpolated_signals.append(interpolated_voltage)

        avg_signal = np.mean(interpolated_signals, axis=0)
        averaged_channels.append(avg_signal)
    return common_time, averaged_channels


def load_signals(path):
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
    print(f"warning: Units not considered currently.\nUnits: {units}")
    return signals, units

if __name__ == '__main__':
    data_dir = pathlib.Path(__file__).parent / 'data' / 'test_Abishay_example'
    signal_list, signal_units = load_signals(data_dir)
    time, avg_channels = average_signals(signal_list, 2)
    for i, channel in enumerate(avg_channels):
        plt.plot(time, channel)
        xlab = list(signal_units.index)[0]
        ylab = list(signal_units.index)[i+1]
        plt.xlabel(f"{xlab} {signal_units[xlab]}")
        plt.ylabel(f"{ylab} {signal_units[ylab]}")

        plt.show()
