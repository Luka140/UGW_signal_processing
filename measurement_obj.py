from signal_obj import Signal 



class Measurement
    def __init__(self, tx_pos, rx_pos, rx_signal):
        
        self.transmitter_position = tx_pos
        self.receiver_position = rx_pos
        self.received_signal = rx_signal

        self.dispersion_curve = ...

    def compensate_dispersion(self):
        ...
        