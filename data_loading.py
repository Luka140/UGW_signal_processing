import warnings
import pandas as pd
import numpy as np
import pathlib
from typing import Iterable
import scipy.interpolate as interpolate
import scipy.signal as spsignal
from signal_obj import Signal
import matplotlib.pyplot as plt

def load_signals_labview(path, plot_outliers=True):
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
        signals = average_signals(signals, plot_outliers=plot_outliers)
    else:
        signals = signals[0]

    return signals

def average_signals(signals: list[list[Signal]], plot_outliers=True) -> list[Signal]:
    """_summary_

    Args:
        signals (list[list[Signal]]): A nested list of Signal objects. 
            The data structure should be: [measurement1[ch1, ch2, ...], measurement2[...], ...]
    Returns:
        list[Signal]: A list of averaged Signals, one for each channel
    """
    averaged_channels = []     # Contains the averaged Signal objects
    interpolated_channels = [] # Contains interpolated version of each signal [ch1[measurement1, ...], ch2[...], ...]
   
    channels = len(signals[0])
    for ch in range(channels):
        min_time, max_time = min([sig[ch].time[0] for sig in signals]), max([sig[ch].time[-1] for sig in signals])
        common_time = np.linspace(min_time, max_time, int(signals[0][ch].time.size))
        interpolated_signal_arrays = []
        for measurement in signals:
            # Interpolate the signal to the common time points, and store the interpolated points
            interp_func = interpolate.CubicSpline(measurement[ch].time, measurement[ch].data, extrapolate=False)
            interpolated_signal = interp_func(common_time)
            interpolated_signal_arrays.append(interpolated_signal)
            
            # check quality of interpolation
            # if ch > 0:
                # plt.plot(common_time, interpolated_signal)
                # plt.scatter(common_time, measurement[ch].data)
                # plt.show()
        
        interpolated_channels.append(interpolated_signal_arrays) # Store to compare with averages for outlier detection
        avg_signal = np.mean(np.stack(interpolated_signal_arrays, axis=0), axis=0)
        averaged_channels.append(Signal(common_time, avg_signal, measurement[ch].t_unit, measurement[ch].d_unit))    

    # TODO this doesnt really work if most of the signal is just noise like in the Labview signals
    # Outlier detection using Z-score
    outlier_idxs = [] # Format [(measurement, channel), (measurement, channel)]
    for channel_idx, (channel, avg_channel) in enumerate(zip(interpolated_channels, averaged_channels)):
        z_scores = []
        for interp_measurement in channel:
            z_score = (interp_measurement - avg_channel.data) / np.std(avg_channel.data)
            z_scores.append(np.max(np.abs(z_score)))
        
        # Identify outliers
        outliers = np.where(np.array(z_scores) > 3)
        if len(outliers[0]) > 0:
            print(f"Outliers detected in channel {channel_idx} at indices: {outliers}")
            
            [outlier_idxs.append((outlier, channel_idx)) for outlier in outliers[0]]  
    
    if plot_outliers and len(outlier_idxs) > 0:
        for outlier_index in set(np.array(outlier_idxs)[:,0]):   
            fig, axs = plt.subplots(channels // 2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots
            fig.suptitle(f"Outlier signals vs Averages (measurement index: {outlier_index})")


            for ch in range(channels):
                row, col = ch // 2, ch % 2  # Determine subplot position
                if row > 1:
                    ax = axs[row, col]
                else:
                    ax = axs[col]

                # Plot individual signals
                ax.plot(common_time, interpolated_channels[ch][outlier_index], color='red', label='Outlier signal' if ch == 0 else "")

                # Plot the average signal
                ax.plot(common_time, averaged_channels[ch].data, color='blue', linewidth=2, label='Average signal')

                # Add labels and legend
                ax.set_title(f"Channel {ch}")
                ax.set_xlabel(f"Time ({averaged_channels[ch].t_unit})")
                ax.set_ylabel(f"Amplitude ({averaged_channels[ch].d_unit})")
                if ch == 0:
                    ax.legend()

            plt.tight_layout()
            plt.show()    

    return averaged_channels


def load_signals_abaqus(path, t_unit=None, d_unit=None):
    signals = []
    for file in path.glob('*.csv'):
        csv_data = pd.read_csv(file, skip_blank_lines=True, header=None).to_numpy(dtype=float)

        # Format of the Abaqus data is [time1, data1, time2, data2, ...]
        for i in range(0, csv_data.shape[1], 2):
            signals.append(Signal(csv_data[:,i], csv_data[:,i+1], t_unit, d_unit))

    return signals

def load_signals_SINTEG(directory, sample_frequency=1e6, skip_idx={}, plot_outliers=False):
    """Load 4 channel signals from the SINTEG acquisition setup.

    Args:
        directory (_type_): The directory containing signals to be averaged, representing a single measurement.
        sample_frequency (_type_, optional): The sample frequency. Defaults to 1e6 (1 MHz).

    Returns:
        _type_: _description_
    """
    sample_spacing = 1 / sample_frequency
    signals = []
    for i, path in enumerate(directory.glob("*.txt")):
        if i in skip_idx:
            continue
        data = np.loadtxt(path)
        measurement_channels = []
        samples = data.shape[0]
        time = np.arange(0, samples * sample_spacing, sample_spacing)

        for i in range(data.shape[1]):
            measurement_channels.append(Signal(time, data[:,i], t_unit='s', d_unit='micro_v'))
        # [signal.plot() for signal in measurement_channels]
        signals.append(measurement_channels)
    return average_signals(signals, plot_outliers=plot_outliers)




if __name__ == "__main__":
    # data_dir = pathlib.Path(__file__).parent / 'data' / 'test_Abishay_example'
    # signal_list = load_signals_labview(data_dir, plot_outliers=False)
    

    # data_dir = pathlib.Path(__file__).parent / 'data' / 'abaqus_test_steel'
    # signal_list = load_signals_abaqus(data_dir)

    data_sinteg = pathlib.Path(__file__).parent / "data" / "GFRP_test_plate_SINTEG" / "measurements_0"
    avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={31}, plot_outliers=True)
    # [sig.plot() for sig in avg_signals]
    # [sig.bandpass(30e3, 90e3).plot() for sig in avg_signals]
    avg_signals[0].plot()
    avg_signals[0].zero_average_signal().plot()
    avg_signals[0].bandpass(30e3, 90e3).plot()
    avg_signals[0].zero_average_signal().bandpass(30e3, 90e3).plot()

