import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Conv2D, TimeDistributed
from tensorflow.keras import Model


class GeneratorModel(Model):
    def __init__(self, n_fft, batch_size, truncate):
        super(GeneratorModel, self).__init__()
        self.output_num = n_fft//2+1
        self.batch_size = batch_size
        self.truncate = truncate

        self.e_gru = GRU(self.output_num, stateful=True, return_sequences=True)  # Encoder
        self.l_d = TimeDistributed(Dense(64, activation='tanh'))  # Latent
        self.d_gru = GRU(self.output_num, stateful=True, return_sequences=True)  # Decoder
        self.d_output = TimeDistributed(Dense(self.output_num))

    def call(self, inputs):
        x = self.e_gru(inputs)
        x = self.l_d(x)
        x = self.d_gru(x)
        x = self.d_output(x)
        return x
