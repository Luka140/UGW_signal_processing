import pathlib
from signal import Signal
from data_loading import load_signals_labview, load_signals_abaqus




if __name__ == "__main__":
    # data_dir = pathlib.Path(__file__).parent / 'data' / 'test_Abishay_example'
    # signal_list = load_signals_labview(data_dir)
    
    # received_signal = signal_list[1]
    # trimmed_signal = received_signal.get_trimmed_signal(0, 0.2)
    # trimmed_signal.plot()
    # trimmed_signal.get_stfft(0.005)
    # bandpassed_signal = Signal.bandpass(trimmed_signal, 0.8e3, 1.2e3)
    # # bandpassed_signal.plot()
    # bandpassed_signal.get_stfft(0.005)
    
    data_dir_ab = pathlib.Path(__file__).parent / 'data' / 'abaqus_test_steel'
    signal_list = load_signals_abaqus(data_dir_ab)

    received_signal = signal_list[1]
    received_signal.plot()

    trimmed_signal = received_signal.get_trimmed_signal(0,400e-6)
    trimmed_signal.plot()
    bandpassed_signal = trimmed_signal.bandpass(40e3, 60e3)
    bandpassed_signal.plot()

    # TODO Measurement class
    # TODO dispersion compensation? 
    # TODO trim by windowing
    # TODO WSST? -ssqueezepy
    # TODO try abaqus data