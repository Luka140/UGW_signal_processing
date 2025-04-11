import numpy as np
import pathlib
from keysight_waveform_arb_generator.hann_window_signal_generator import generate_arb_file
from data_loading import load_signals_labview
from measurement_obj import Measurement
from signal_obj import Signal


if __name__ == '__main__':
    
    # --------- Load signals from files ---------
    data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurement_full_plate2" / "third_new_cable_100khz_3cycl_s1803_long"
    avg_signals = load_signals_labview(data, skip_idx={}, skip_ch={}, plot_outliers=False, filter_before_average=True)

    # --------- Bandpass filter and zero average signals ---------
    avg_signals = [sig.zero_average_signal().bandpass(80e3, 120e3, order=2) for sig in avg_signals]
    avg_signals[2].plot()

    t0, t1 = 0.5e-4, 1.75e-4
    rev_signal = avg_signals[2].get_time_reversed_section(t0, t1)
    rev_signal.plot()


    waveform_dir = pathlib.Path(__file__).parent / "data" / "time_reversed_waveforms"
    waveform_path = waveform_dir / f"[reversed_{t0:.3e}-{t1:.3e}]_{data.name}.arb"
    generate_arb_file(waveform_path, rev_signal.data, high_level=3, low_level=-3, sample_rate=rev_signal.sample_frequency)





