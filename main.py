import pathlib
from signal_obj import Signal
from dispersiondata_obj import DispersionData
from measurement_obj import Measurement
from data_loading import load_signals_labview, load_signals_abaqus, load_signals_SINTEG
import matplotlib.pyplot as plt 



if __name__ == "__main__":
    # --------- Load signals from files ---------    
    # received_signal = signal_list[1]
    # trimmed_signal = received_signal.get_trimmed_signal(0, 0.2)
    # trimmed_signal.plot()
    # trimmed_signal.get_stfft(0.005)
    # bandpassed_signal = Signal.bandpass(trimmed_signal, 0.8e3, 1.2e3)
    # # bandpassed_signal.plot()
    # bandpassed_signal.get_stfft(0.005)
    
    # data_dir_ab = pathlib.Path(__file__).parent / 'data' / 'measurement_data' / 'abaqus_test_steel'
    # signal_list = load_signals_abaqus(data_dir_ab)

    # received_signal = signal_list[1]
    # received_signal.plot()

    # trimmed_signal = received_signal.get_trimmed_signal(0,400e-6)
    # trimmed_signal.plot()
    # bandpassed_signal = trimmed_signal.bandpass(40e3, 60e3)
    # bandpassed_signal.plot()

    # bandpassed_signal.set_max_peak_number(1)
    # bandpassed_signal.plot()

    # --------- Load dispersion curves from files ---------
    dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    dispersion = DispersionData()
    for curves_file in dispersion_dir.glob('*.txt'):
        dispersion.merge(DispersionData(curves_file))
    print("Available modes:", dispersion.get_available_modes())

    # --------- Load signals from files ---------
    
    data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_test_plate" / "REDO_MEASUREMENTS" / "s1802tx_s1802rx_s1803_rx_perpen_cage"
    
    # --- Without averaging
    # raw_signals = load_signals_SINTEG(data_sinteg, skip_idx={9}, plot_outliers=True, skip_ch=[], average=False)
    # signal = raw_signals[0]

    # --- With averaging
    signal = load_signals_SINTEG(data_sinteg, skip_idx={39}, plot_outliers=True, skip_ch=[], average=True)

    # --------- Bandpass filter and zero average signals ---------
    signal = [sig.zero_average_signal().bandpass(50e3, 70e3, order=5) for sig in signal]

    # SET PADDING TO 1 
    [sig.set_fft_pad_times(5) for sig in signal]
    # ------------ CHECK ORDER OF ARIVAL
    fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    for i in range(len(signal)):
            axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    plt.show()

    measurement = Measurement((0,0), [(58e-3,0), (98e-3, 0.), (100e-3, 0.)], tx_signal=signal[-1], rx_signal=signal[:-1], dispersion_curves=dispersion)
    measurement.compare_signals(base_index='tx',comparison_indices=[1,2])
    measurement.compare_signals(base_index= 1, comparison_indices=[2])
    # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")


    # TODO Measurement class
    # TODO dispersion compensation? 
    # TODO trim by windowing
    # TODO WSST? -ssqueezepy
    # TODO try abaqus data