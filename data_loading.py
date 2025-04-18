import warnings
import pandas as pd
import numpy as np
import pathlib
from typing import Iterable
import scipy.interpolate as interpolate
import scipy.signal as spsignal
from signal_obj import Signal
import matplotlib.pyplot as plt
import copy 


def load_signals_labview(path, skip_idx={}, skip_ch={}, plot_outliers=True, filter_before_average=False, lowcut=10e3, highcut=200e3, order=2, t_to_sec_factor=None, overwrite_average=False):
    
    print(path)
    signals = []
    prev_units = None

    if not path.exists():
        raise FileNotFoundError(f"The path {path} does not exist.")
    paths = sorted(list(path.glob('*.csv')))

    if len(list(paths)) < 1:
        raise FileNotFoundError(f"No data files found in {path}")

    if any('[averaged]' in path.name.lower() for path in paths) and not overwrite_average:
        print('Found averaged file, loading it instead of the original files.')
        paths = [path for path in paths if '[averaged]' in path.name.lower()]
        return _open_labview_file(paths[0], unit_row=0, skip_ch=skip_ch, t_to_sec_factor=t_to_sec_factor, prev_units=None)[0]
    

    for i, datafile in enumerate(paths):
        if i in skip_idx:
            continue
        if '[faulty]' in datafile.name.lower():
            print(f"Skipping {datafile} because it is marked as faulty.")
            continue
        if '[averaged]' in datafile.name.lower():
            print(f"Skipping {datafile} because it is marked as averaged, while overwrite_average is {overwrite_average}")
            continue
        
        file_channels, prev_units = _open_labview_file(datafile, unit_row=0, skip_ch=skip_ch, t_to_sec_factor=t_to_sec_factor, prev_units=prev_units)
        if len(file_channels) == 0:
            print(f"Skipping {datafile}")
            continue 

        if filter_before_average:
            file_channels = [sig.zero_average_signal().bandpass(lowcut, highcut, order=order) for sig in file_channels]

        signals.append(file_channels)
    
    # Format is [measurement1[ch1, ch2, ch3...], measurement2[...],...]
    # The channels are kept separated for averaging, to ensure different channels are not mixed
    # If there are multiple measurements, average. Otherwise, unpack to [ch1, ch2, ch3,...]
    # Averaging will result in the same data structure of [avg_ch1, avg_ch2, avg_ch3,...]
    if len(signals) > 1:
        signals = average_signals(signals, plot_outliers=plot_outliers)
        save_channels_labview_format(signals, path)
    else:
        signals = signals[0]

    return signals

def _open_labview_file(datafile, unit_row=0, skip_ch=[], t_to_sec_factor=None, prev_units=None):
    channels = []
    csv_data = pd.read_csv(datafile, skip_blank_lines=True, low_memory=False)
    units = list(csv_data.iloc[unit_row, :])
    if prev_units is not None and any([units[i] != prev_units[i] for i in range(len(units))]):
        warnings.warn(f"The file at {datafile} has different units than the previous one:\nPrevious:\n{prev_units}\n\nCurrent:\n{units}\nThis will cause issues with averaging.")
    prev_units = units

    csv_data.drop(unit_row, inplace=True)
    try:
        data = csv_data.to_numpy(dtype=float)
    except ValueError:
        print(f"Error converting data to float in {datafile}. Skipping this file.")
        return [], []
    
    
    for ch in range(1, data.shape[1]):
        if ch - 1 in skip_ch: # -1 because first column is time
            continue
        
        if t_to_sec_factor is not None:
            print(f"Time unit factor is set to {t_to_sec_factor}. - Ignoring csv header")
        else:

            if "(ms)" in units[0]:
                t_sec_f = 1e3
            elif "(us)" in units[0]:
                t_sec_f = 1e6
            elif "(s)" in units[0]:
                t_sec_f = 1
            else:
                warnings.warn("Time unit not detected. Assuming milliseconds")
                t_sec_f = 1e3

        sig = Signal(data[:, 0]/t_sec_f, data[:, ch], t_unit= "s", d_unit=units[ch])

        channels.append(sig)
    return channels, units
    
def save_channels_labview_format(channels: list[Signal], path: pathlib.Path):
    filename = "[AVERAGED]" 

    data_array = channels[0].to_array() 
    units = [f'({channels[0].t_unit})', channels[0].d_unit]
    headers = ["Time", "Avg CH0"]

    if len(channels) > 1:
        for i in range(1, len(channels)):
            if (channels[i].time != channels[0].time).any():
                warnings.warn("The time vectors of the signals are not the same. Using the first one...")


            data_array = np.hstack((data_array,channels[i].data.reshape(-1, 1)))
            units.append(channels[i].d_unit)
            headers.append(f"Avg CH{i}")

    with open(path / f"{filename}.csv", 'w') as f:
        f.write(f"{','.join(headers)}\n")
        f.write(f"{','.join(units)}\n")
        for row in data_array:
            f.write(f"{','.join([str(x) for x in row])}\n")
    print(f"Saved averaged signals to {path / f'{filename}.csv'}")
    return path / filename



def average_signals(initial_signals: list[list[Signal]], plot_outliers=True, tx_ch: int | None=None) -> list[Signal]:
    """

    Args:
        initial_signals (list[list[Signal]]): A nested list of Signal objects. 
            The data structure should be: [measurement1[ch1, ch2, ...], measurement2[...], ...]
        plot_outliers (bool): Show plots of signals detected as outliers.
    Returns:
        list[Signal]: A list of averaged Signals, one for each channel
    """
    signals = copy.deepcopy(initial_signals) # Prevent mutating input
    averaged_channels = []     # Contains the averaged Signal objects
    interpolated_channels = [] # Contains interpolated version of each signal [ch1[measurement1, ...], ch2[...], ...]
    channels = len(signals[0])
    
    # ---------------- Find consistent start time based on tx channel
    # start_times = [0.] * len(signals)
    # if tx_ch is not None:
    #     for i, measurement in enumerate(signals):
    #         tx_signal = measurement[tx_ch]
    #         # TODO detect clipping waveform. If this is the case, detect based on another metric
    #         start_index = np.argmax(tx_signal.amplitude_envelope)
    #         time_offset = tx_signal.time[start_index]
    #         print(time_offset)
    #         tx_signal.plot()
    #         # plt.axvline(time_offset, color='red')
    #         # plt.show()
    #         start_times[i] = time_offset

    # # ---------------- Set start time to be consistent for each signal 
    # # TODO maybe instead just drop the ones that deviate a lot
    # # TODO also compare from Tx signal, by taking t=0 as t for ~weighted average t of excitation signal pulse
    # # TODO maybe base t=0 on initial threshold crossing

    # for i, _ in enumerate(signals):
    #     for ch in range(channels):
    #         signals[i][ch] = Signal(signals[i][ch].time - start_times[i], signals[i][ch].data, signals[i][ch].t_unit, signals[i][ch].d_unit)
    ########################################################


    # ---------------- Create consistent grid of times that are available for each signal 
    min_time, max_time = max([sig[ch].time[0] for sig in signals for ch in range(channels)]), min([sig[ch].time[-1] for sig in signals for ch in range(channels)])
    common_time = np.linspace(min_time, max_time, int(signals[0][0].time.size)) # Set number of datapoints based on first measurement first channel. They should all be the same.

    # ---------------- Interpolate signals to the consistent grid, then average the results 
    for ch in range(channels):
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


    # ---------------- Outlier detection (does not remove them, only presents them)
    # TODO this doesnt really work if most of the signal is just noise like in the Labview signals
    
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
            # Determine the grid layout based on number of channels
            if channels <= 2:
                rows, cols = 1, channels
            else:
                rows, cols = 2, 2
            
            fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
            fig.suptitle(f"Outlier signals vs Averages (measurement index: {outlier_index})")
            
            # Flatten axs array for easier indexing when it's 2D
            if channels > 1:
                axs = axs.ravel()
            
            for ch in range(channels):
                # For single channel case
                if channels == 1:
                    ax = axs
                else:
                    ax = axs[ch]
                
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

def load_signals_SINTEG(directory:pathlib.Path, sample_frequency=1e6, skip_idx={}, plot_outliers=False, tx_ch=-1, average=True, skip_ch=()):
    """Load 4 channel signals from the SINTEG acquisition setup.

    Args:
        directory (_type_): The directory containing signals to be averaged, representing a single measurement.
        sample_frequency (_type_, optional): The sample frequency. Defaults to 1e6 (1 MHz).

    Returns:
        _type_: _description_
    """
    sample_spacing = 1 / sample_frequency
    signals = []
    data_files = [path for path in directory.glob("*.txt") if 'read' not in path.name.lower()]
    if len(data_files) == 0:
        raise FileNotFoundError(f"No data files found in {directory}")
    
    for i, path in enumerate(data_files):
        if i in skip_idx:
            continue
        data = np.loadtxt(path)
        measurement_channels = []
        samples = data.shape[0]
        time = np.arange(0, samples * sample_spacing, sample_spacing)

        for i in range(data.shape[1]):
            if i in skip_ch:
                continue
            measurement_channels.append(Signal(time, data[:,i], t_unit='s', d_unit='micro_v').zero_average_signal())
        # [signal.plot() for signal in measurement_channels]
        signals.append(measurement_channels)
    
    if average:
        return average_signals(signals, plot_outliers=plot_outliers, tx_ch=tx_ch)
    else:
        return signals




if __name__ == "__main__":
    # data_dir = pathlib.Path(__file__).parent / 'data' / 'test_Abishay_example'
    # signal_list = load_signals_labview(data_dir, plot_outliers=False)
    

    # data_dir = pathlib.Path(__file__).parent / 'data' / 'abaqus_test_steel'
    # signal_list = load_signals_abaqus(data_dir)

    data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "GFRP_test_plate_SINTEG" / "measurements_0"
    avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={31}, plot_outliers=True, tx_ch=None)
    # [sig.plot() for sig in avg_signals]
    [sig.zero_average_signal().bandpass(30e3, 90e3).plot() for sig in avg_signals]

    # ch4_signal = avg_signals[1].zero_average_signal().bandpass(30e3, 90e3)
    # ch1_signal = avg_signals[2].zero_average_signal().bandpass(30e3, 90e3)
    # ch4_signal.compare_other_signal(ch1_signal)


    # avg_signals[0].plot()
    # avg_signals[0].zero_average_signal().plot()
    # avg_signals[0].bandpass(30e3, 90e3).plot()
    # avg_signals[0].zero_average_signal().bandpass(30e3, 90e3).plot()

