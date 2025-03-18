import pathlib
from Signal import Signal
from data_loading import load_signals_labview, load_signals_abaqus




if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent / 'data' / 'test_Abishay_example'
    # data_dir_ab = pathlib.Path(__file__).parent / 'data' / 'abaqus_test_steel'
    signal_list = load_signals_labview(data_dir)
    # signal_list = load_signals_abaqus(data_dir_ab)

    received_signal = signal_list[1]
    # received_signal.plot()

    trimmed_signal = received_signal.get_trimmed_signal(0, 0.2)
    trimmed_signal.plot()

    # bandpassed_signal = Signal.bandpass(trimmed_signal, 0.8e3, 1.2e3)
    # bandpassed_signal.plot()

    # TODO find peaks -> flight time
    # TODO add windowing
    # TODO Stfft?
    # TODO Measurement class
