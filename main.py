import pathlib
from Signal import Signal
from data_loading import load_signals_labview




if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent / 'data' / 'test_Abishay_example'
    signal_list = load_signals_labview(data_dir)

    received_signal = signal_list[1]
    received_signal.plot()

    trimmed_signal = received_signal.get_trimmed_signal(0,0.15)
    trimmed_signal.plot()

    bandpassed_signal = Signal.bandpass(trimmed_signal, 0.8e3, 1.2e3)
    bandpassed_signal.plot()

