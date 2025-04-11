import pathlib
import pandas as pd
import scipy.signal as spsignal
import scipy.interpolate as interpolate
import scipy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Iterable

from data_loading import load_signals_labview, load_signals_SINTEG
from dispersiondata_obj import DispersionData
from signal_obj import Signal
from measurement_obj import Measurement


if __name__ == '__main__':


    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'c1p1_curves_GFRP_steel_15mm'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurement_p1c1" / "GFRP_steel_p1c1_100khz_3_cycles_18_58_98_S1803"
    # avg_signals = load_signals_labview(data, skip_idx={}, plot_outliers=False, filter_before_average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(80e3, 120e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]


    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    # measurement.plot_envelopes()
    # measurement.compare_signals(0,2)
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")



    # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'c1p1_curves_GFRP_steel_15mm'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurement_p1c1" / "GFRP_steel_p1c1_50khz_3_cycles_full_cage_SH_S1802_RESAVE"
    # avg_signals = load_signals_labview(data, skip_idx={}, skip_ch={1}, plot_outliers=False, filter_before_average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 70e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]


    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    
    # measurement.plot_envelopes()
    # measurement.compare_signals(0,1)
    # measurement.compare_signals(0,2)
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")





    # --------- Load dispersion curves from files ---------
    dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'c1p1_curves_GFRP_steel_15mm'
    
    dispersion = DispersionData()
    for curves_file in dispersion_dir.glob('*.txt'):
        dispersion.merge(DispersionData(curves_file))
    print("Available modes:", dispersion.get_available_modes())

    # --------- Load signals from files ---------
    data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurement_full_plate2" / "third_new_cable_100khz_3cycl_s1803_long"
    avg_signals = load_signals_labview(data, skip_idx={}, skip_ch={}, plot_outliers=False, filter_before_average=True)

    # --------- Bandpass filter and zero average signals ---------
    avg_signals = [sig.zero_average_signal().bandpass(80e3, 120e3, order=2) for sig in avg_signals]
    # avg_signals = [sig.zero_average_signal() for sig in avg_signals]


    measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion, excitation_cycles=3, excitation_frequency=100e3)
    
    measurement.plot_envelopes(plot_predicted_arrival_times=False)
    measurement.compare_signals(0,1)
    measurement.compare_signals(1,2)
    
    # new_signals = measurement.compensate_dispersion(center_frequency=100e3, mode="B1")
    # compensated_measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=new_signals, dispersion_curves=dispersion)
    # measurement.plot_envelopes()
    # # measurement.compare_signals(0,1)
    # # measurement.compare_signals(1,2)
    



    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'c1p1_curves_GFRP_steel_15mm'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurement_full_plate2" / "second_set_full_cage_100khz_s1803_long"
    # avg_signals = load_signals_labview(data, skip_idx={}, skip_ch={}, plot_outliers=False, filter_before_average=True, t_to_sec_factor=1000000)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(80e3, 120e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]


    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion, excitation_cycles=3, excitation_frequency=100e3)
    
    # measurement.plot_envelopes(plot_predicted_arrival_times=False)
    # measurement.compare_signals(0,1)
    # measurement.compare_signals(1,2)
    
    # # new_signals = measurement.compensate_dispersion(center_frequency=100e3, mode="B1")
    # # compensated_measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=new_signals, dispersion_curves=dispersion)
    # # measurement.plot_envelopes()
    # # # measurement.compare_signals(0,1)
    # # # measurement.compare_signals(1,2)
    





    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'c1p1_curves_GFRP_steel_15mm'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurement_full_plate2" / "fourth_50khz_3cycl_s1803_long" /"full_cage_50khz_3cycle_s1803long"
    # avg_signals = load_signals_labview(data, skip_idx={}, skip_ch={}, plot_outliers=False, filter_before_average=True, t_to_sec_factor=1000000)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 70e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]


    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion, excitation_cycles=3, excitation_frequency=100e3)
    
    # measurement.plot_envelopes(plot_predicted_arrival_times=False)
    # measurement.compare_signals(0,1)
    # measurement.compare_signals(1,2)
    
    # # new_signals = measurement.compensate_dispersion(center_frequency=100e3, mode="B1")
    # # compensated_measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=new_signals, dispersion_curves=dispersion)
    # # measurement.plot_envelopes()
    # # # measurement.compare_signals(0,1)
    # # # measurement.compare_signals(1,2)
    