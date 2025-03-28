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

    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_test_plate" / "REDO_MEASUREMENTS" / "sh_full_cage_alu_120kHz"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={7, 36}, plot_outliers=True)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(40e3, 80e3, order=2) for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(18e-3,0), (58e-3, 0.), (98e-3, 0.)], tx_signal=avg_signals[-1], rx_signal=avg_signals[:-1], dispersion_curves=dispersion)
    # measurement.compare_signals(0,2)
    # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")





    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_test_plate" / "REDO_MEASUREMENTS" / "sh_58_40__100mm_120khz"
    # avg_signals = load_signals_SINTEG(data_sinteg, skip_idx={18}, plot_outliers=True)

    # # --------- Bandpass filter and zero average signals ---------
    # avg_signals = [sig.zero_average_signal().bandpass(40e3, 80e3, order=2) for sig in avg_signals]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(avg_signals)):
    #         axtime, axfrequency = Signal._plot_helper(avg_signals[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()
    

    # measurement = Measurement((0,0), [(58e-3,0), (98e-3, 0.), (198e-3, 0.)], tx_signal=avg_signals[-1], rx_signal=avg_signals[:-1], dispersion_curves=dispersion)
    # measurement.compare_signals(0,[1,2])
    # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")





    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_bonded_plate" / "S1802_tx_shear" / "single_tx_0_deg_connector_100mm"
    
    # # --- Without averaging
    # # raw_signals = load_signals_SINTEG(data_sinteg, skip_idx={34, 41}, plot_outliers=True, skip_ch=[2], average=False)
    # # signal = raw_signals[1]

    # # --- With averaging
    # signal = load_signals_SINTEG(data_sinteg, skip_idx={34, 41}, plot_outliers=True, skip_ch=[2], average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # signal = [sig.zero_average_signal().bandpass(50e3, 70e3, order=3) for sig in signal]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(signal)):
    #         axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(100e-3,0), (100e-3, 0.)], tx_signal=signal[-1], rx_signal=signal[:-1], dispersion_curves=dispersion)
    # measurement.compare_signals(base_index='tx',comparison_indices=[0,1])
    # # measurement.compare_signals(base_index= 0, comparison_indices=[1])
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")



    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_bonded_plate" / "S1802_tx_shear" / "single_tx_90_deg_connector_100mm"
    
    # # --- Without averaging
    # # raw_signals = load_signals_SINTEG(data_sinteg, skip_idx={}, plot_outliers=True, skip_ch=[2], average=False)
    # # signal = raw_signals[0]

    # # --- With averaging
    # signal = load_signals_SINTEG(data_sinteg, skip_idx={16, 27, 28, 30}, plot_outliers=True, skip_ch=[2], average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # signal = [sig.zero_average_signal().bandpass(50e3, 70e3, order=3) for sig in signal]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(signal)):
    #         axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(100e-3,0), (100e-3, 0.)], tx_signal=signal[-1], rx_signal=signal[:-1], dispersion_curves=dispersion)
    # measurement.compare_signals(base_index='tx',comparison_indices=[0,1])
    # # measurement.compare_signals(base_index= 0, comparison_indices=[1])
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")




    #     # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_bonded_plate" / "S1803_tx_long" / "distance_msr_single_tx1803_40mm"
    
    # # --- Without averaging
    # # raw_signals = load_signals_SINTEG(data_sinteg, skip_idx={}, plot_outliers=True, skip_ch=[2], average=False)
    # # signal = raw_signals[0]

    # # --- With averaging
    # signal = load_signals_SINTEG(data_sinteg, skip_idx={0,  4,  6, 2}, plot_outliers=True, skip_ch=[2], average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # signal = [sig.zero_average_signal().bandpass(50e3, 70e3, order=5) for sig in signal]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(signal)):
    #         axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(100e-3,0), (100e-3, 0.)], tx_signal=signal[-1], rx_signal=signal[:-1], dispersion_curves=dispersion)
    # # measurement.compare_signals(base_index='tx',comparison_indices=[0,1])
    # measurement.compare_signals(base_index= 0, comparison_indices=[1])
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")



    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_bonded_plate" / "S1803_tx_long" / "distance_msr_single_tx1803_100mm"
    
    # # --- Without averaging
    # raw_signals = load_signals_SINTEG(data_sinteg, skip_idx={}, plot_outliers=True, skip_ch=[2], average=False)
    # signal = raw_signals[0]

    # # --- With averaging
    # # signal = load_signals_SINTEG(data_sinteg, skip_idx={9}, plot_outliers=True, skip_ch=[2], average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # signal = [sig.zero_average_signal().bandpass(50e3, 70e3, order=5) for sig in signal]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(signal)):
    #         axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(100e-3,0), (100e-3, 0.)], tx_signal=signal[-1], rx_signal=signal[:-1], dispersion_curves=dispersion)
    # measurement.compare_signals(base_index='tx',comparison_indices=[0,1])
    # measurement.compare_signals(base_index= 0, comparison_indices=[1])
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")





    #     # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_bonded_plate" / "S1803_tx_long" / "40_mm_symmetrical_measurement"
    
    # # --- Without averaging
    # # raw_signals = load_signals_SINTEG(data_sinteg, skip_idx={}, plot_outliers=True, skip_ch=[2], average=False)
    # # signal = raw_signals[0]

    # # --- With averaging
    # signal = load_signals_SINTEG(data_sinteg, skip_idx={ 3, 11, 19, 41, 44}, plot_outliers=True, skip_ch=[], average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # signal = [sig.zero_average_signal().bandpass(50e3, 70e3, order=5) for sig in signal]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(signal)):
    #         axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(40e-3,0), (40e-3, 0.), (40e-3,0)], tx_signal=signal[-1], rx_signal=signal[:-1], dispersion_curves=dispersion)
    # measurement.compare_signals(base_index='tx',comparison_indices=[0,1,2])
    # measurement.compare_signals(base_index= 0, comparison_indices=[1])
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")




    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_dir = pathlib.Path(__file__).parent / 'data' / 'measurement_data' / 'test_Abishay_example'

    # signal = load_signals_labview(data_dir, plot_outliers=False)


    # # --------- Bandpass filter and zero average signals ---------
    # signal = [sig.zero_average_signal() for sig in signal]

    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(signal)):
    #         axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(200e-3,0)], tx_signal=signal[0], rx_signal=signal[1], dispersion_curves=dispersion)
    # measurement.compare_signals(base_index='tx',comparison_indices=[0])
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")


    # # signal_list = load_signals_labview(data_dir)
    


    # # --------- Load dispersion curves from files ---------
    # dispersion_dir = pathlib.Path(__file__).parent / 'data' / 'dispersion_curves' / 'reference_alu_curves'
    
    # dispersion = DispersionData()
    # for curves_file in dispersion_dir.glob('*.txt'):
    #     dispersion.merge(DispersionData(curves_file))
    # print("Available modes:", dispersion.get_available_modes())

    # # --------- Load signals from files ---------
    
    # data_sinteg = pathlib.Path(__file__).parent / "data" / "measurement_data" / "alu_test_plate" / "REDO_MEASUREMENTS" / "s1802tx_s1802rx_s1803_rx_inline_cage"
    
    # # # --- Without averaging
    # # raw_signals = load_signals_SINTEG(data_sinteg, skip_idx={0,5}, plot_outliers=True, skip_ch=[], average=False)
    # # signal = raw_signals[1]

    # # --- With averaging
    # signal = load_signals_SINTEG(data_sinteg, skip_idx={0,5}, plot_outliers=True, skip_ch=[], average=True)

    # # --------- Bandpass filter and zero average signals ---------
    # # signal = [sig.zero_average_signal().bandpass(55e3, 65e3, order=5) for sig in signal]
    # signal = [sig.zero_average_signal().bandpass(50e3, 70e3, order=2) for sig in signal]


    # # ------------ CHECK ORDER OF ARIVAL
    # fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    # for i in range(len(signal)):
    #         axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    # plt.show()

    # measurement = Measurement((0,0), [(58e-3,0), (98e-3, 0.), (100e-3, 0.)], tx_signal=signal[-1], rx_signal=signal[:-1], dispersion_curves=dispersion)
    # measurement.compare_signals(base_index='tx',comparison_indices=[1,2])
    # measurement.compare_signals(base_index= 1, comparison_indices=[2])
    # # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")



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
    signal = [sig.zero_average_signal().bandpass(50e3, 70e3, order=2) for sig in signal]

    # ------------ CHECK ORDER OF ARIVAL
    fig, (axtime, axfrequency) = plt.subplots(nrows=2, sharex='none', tight_layout=True)
    for i in range(len(signal)):
            axtime, axfrequency = Signal._plot_helper(signal[i], axtime, axfrequency,  label=f"sig{i}", plot_waveform=False)
    plt.show()

    measurement = Measurement((0,0), [(58e-3,0), (98e-3, 0.), (100e-3, 0.)], tx_signal=signal[-1], rx_signal=signal[:-1], dispersion_curves=dispersion)
    measurement.compare_signals(base_index='tx',comparison_indices=[1,2])
    measurement.compare_signals(base_index= 1, comparison_indices=[2])
    # new_signals = measurement.compensate_dispersion(center_frequency=60e3, mode="A0")

