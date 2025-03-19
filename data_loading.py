import warnings
import pandas as pd
import numpy as np
import pathlib
from typing import Iterable
from scipy.interpolate import interpolate
from signal_obj import Signal

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
        data = csv_data.to_numpy(dtype=float)
        channels = []
        unit_list = list(units)
        for ch in range(1, data.shape[1]):
            channels.append(Signal(data[:, 0], data[:, ch], t_unit=unit_list[0], d_unit=unit_list[1]))
        signals.append(channels)
    # print(f"warning: Units not considered currently.\nUnits: {prev_units}")

    # Format is [measurement1[ch1, ch2, ch3...], measurement2[...],...]
    # The channels are kept separated for averaging, to ensure different channels are not mixed
    # If there are multiple measurements, average. Otherwise, unpack to [ch1, ch2, ch3,...]
    # Averaging will result in the same data structure of [avg_ch1, avg_ch2, avg_ch3,...]
    if len(signals) > 1:
        signals = average_signals(signals)
    else:
        signals = signals[0]

    return signals

def average_signals(signals: list[list[Signal]]) -> list[Signal]:
    """_summary_

    Args:
        signals (list[list[Signal]]): A nested list of Signal objects. 
            The data structure should be: [measurement1[ch1, ch2, ...], measurement2[...], ...]
    Returns:
        list[Signal]: A list of averaged Signals, one for each channel
    """
    averaged_channels = []
    channels = len(signals[0])
    for ch in range(channels):
        min_time, max_time = min([sig[ch].time[0] for sig in signals]), max([sig[ch].time[-1] for sig in signals])
        common_time = np.linspace(min_time, max_time, int(signals[0][ch].time.size))
        interpolated_signal_arrays = []
        for measurement in signals:
            # Interpolate the signal to the common time points, and store the interpolated points
            interp_func = interpolate.interp1d(measurement[ch].time, measurement[ch].data, kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolated_signal = interp_func(common_time)
            interpolated_signal_arrays.append(interpolated_signal)
        avg_signal = np.mean(np.stack(interpolated_signal_arrays, axis=0), axis=0)
        averaged_channels.append(Signal(common_time, avg_signal, measurement[ch].t_unit, measurement[ch].d_unit))
    return averaged_channels


def load_signals_abaqus(path, t_unit=None, d_unit=None):
    signals = []
    for file in path.glob('*.csv'):
        csv_data = pd.read_csv(file, skip_blank_lines=True, header=None).to_numpy(dtype=float)

        # Format of the Abaqus data is [time1, data1, time2, data2, ...]
        for i in range(0, csv_data.shape[1], 2):
            signals.append(Signal(csv_data[:,i], csv_data[:,i+1], t_unit, d_unit))

    return signals

def load_data_SINTEG(directory, sample_frequency=1e6):
    """Load 4 channel signals from the SINTEG acquisition setup.

    Args:
        directory (_type_): The directory containing signals to be averaged, representing a single measurement.
        sample_frequency (_type_, optional): The sample frequency. Defaults to 1e6 (1 MHz).

    Returns:
        _type_: _description_
    """
    sample_spacing = 1 / sample_frequency
    signals = []
    for path in directory.glob("*.txt"):
        data = np.loadtxt(path)
        measurement_channels = []
        samples = data.shape[0]
        time = np.arange(0, samples * sample_spacing, sample_spacing)

        for i in range(data.shape[1]):
            measurement_channels.append(Signal(time, data[:,i], t_unit='s', d_unit='micro_v'))
        
        signals.append(measurement_channels)

    return average_signals(signals)




if __name__ == "__main__":
    # data_dir = pathlib.Path(__file__).parent / 'data' / 'abaqus_test_steel'
    # signal_list = load_signals_abaqus(data_dir)

    data_sinteg = pathlib.Path(__file__).parent / "data" / "GFRP_test_plate_SINTEG" / "measurements_0"
    avg_signals = load_data_SINTEG(data_sinteg)
    [sig.bandpass(30e3, 90e3).plot() for sig in avg_signals]
