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
    ...

    # # TODO A0 between signal 0 and 1 makes sense, between 1 and 2 not so much
    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurement_100khz_5_cycles_18mm_98mm_200mm_s1803_longitudinal"
    # avg_signals = load_signals_labview(data, skip_idx={}, plot_outliers=True, filter_before_average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(80e3, 120e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (98e-3, 0.), (200e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    # measurement.compare_signals(0,2)
    # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")




    # TODO A0 arrival time seems to make sense! 
    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurement_100khz_3cycles_18_58_98_s1803"
    # avg_signals = load_signals_labview(data, skip_idx={}, plot_outliers=True, filter_before_average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(80e3, 120e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    # measurement.compare_signals(0,2)
    # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")



    # # TODO A0 arrival time seems to make sense!
    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurement_100khz_5cycles_18_58_98_s1803"
    # avg_signals = load_signals_labview(data, skip_idx={}, plot_outliers=True, filter_before_average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(80e3, 120e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    # measurement.compare_signals(0,2)
    # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")




    # TODO A0 is pretty on point (1800 0-2, 1885 1-2  -- Should be 1780 m/s) 
    # --------- Load dispersion curves from files ---------
    dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    dispersion = DispersionData()
    for curves_file in dispersion_dir.glob('*.txt'):
        dispersion.merge(DispersionData(curves_file))
    print("Available modes:", dispersion.get_available_modes())

    # --------- Load signals from files ---------
    data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurements_50khz_3_cycles_18mm_58_98mm_s1803"
    avg_signals = load_signals_labview(data, skip_idx={}, plot_outliers=True, filter_before_average=True)

    [sig.set_fft_pad_times(1) for sig in avg_signals]

    # --------- Bandpass filter and zero average signals ---------
    avg_signals = [sig.zero_average_signal().bandpass(30e3, 70e3, order=2) for sig in avg_signals]
    # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # ------------ CHECK ORDER OF ARIVAL
    fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    for i in range(len(avg_signals)):
            axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    plt.show()

    measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    measurement.compare_signals(0,2)
    measurement.compare_signals(1,2)
    new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")






    # # TODO A0 arrival time 0-1 is a bit low (1760 m/s), 1-2 is very low (1185 m/s)
    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurements_50khz_3_cycles_18_98_200_S1803"
    # avg_signals = load_signals_labview(data, skip_idx={}, plot_outliers=True, filter_before_average=True)

    # [sig.set_fft_pad_times(1) for sig in avg_signals]

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 70e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (98e-3, 0.), (200e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    # measurement.compare_signals(0,2)
    # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")


    # # TODO Presumably SH0 and A0 peaks are overlapping a bit, making it hard to read. Between sensor 18-98mm, it does seem like SH0 us arriving first with ~3000 m/s
    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurements_50khz_3_cycles_18mm_98mm_200mm_s1802"
    # avg_signals = load_signals_labview(data, skip_idx={}, plot_outliers=False, filter_before_average=True)

    # [sig.set_fft_pad_times(1) for sig in avg_signals]

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 70e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (98e-3, 0.), (200e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    # measurement.compare_signals(0,2)
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")





    # # TODO Velocities 0-1: ~3280 m/s, 1-2: ~3250 m/s || aligns with SH0 theoretically (3100)
    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurements_50khz_3_cycles_full_cage_s1802"
    # avg_signals = load_signals_labview(data, skip_idx={}, skip_ch={1}, plot_outliers=False, filter_before_average=True)

    # [sig.set_fft_pad_times(1) for sig in avg_signals]

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 70e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    
    # measurement.compare_signals(0,2)
    # measurement.compare_signals(1,2)
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")



    # # TODO Inline measurement, first peaks for S1802 align with SH0, S1803 barely reads anything as expected
    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurements_50khz_3cycles_18mm_98mmS1802__100mmS1803_inline"
    # avg_signals = load_signals_labview(data, skip_idx={}, skip_ch={}, plot_outliers=True, filter_before_average=True)

    # [sig.set_fft_pad_times(1) for sig in avg_signals]

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 70e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    
    # # measurement.compare_signals(0,2)
    # # measurement.compare_signals(1,2)
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")

    
    # # TODO First peaks S1802 still align with SH0, S1803 now does read more (presumably A0), and it comes in later than the SH0 wave does!
    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    # data = pathlib.Path(__file__).parent / "data" / "measurement_data" / "TESTS_OSCILLOSCOPE_AE_HALL" / "measurements_alu" / "measurements_50khz_3cycles_18mm_98mmS1802__100mmS1803_perpendicular_new_sensor"
    # avg_signals = load_signals_labview(data, skip_idx={}, skip_ch={}, plot_outliers=True, filter_before_average=True)

    # [sig.set_fft_pad_times(1) for sig in avg_signals]

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(30e3, 70e3, order=2) for sig in avg_signals]
    # # avg_signals = [sig.zero_average_signal() for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=None, rx_signal=avg_signals, dispersion_curves=dispersion)
    
    # # measurement.compare_signals(0,2)
    # # measurement.compare_signals(1,2)
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")



