import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Conv2D, TimeDistributed
from tensorflow.keras import Model


class GeneratorModel(Model):
    def __init__(self, n_fft, batch_size, truncate):
        super(GeneratorModel, self).__init__()
        self.output_num = n_fft//2+1
        self.batch_size = batch_size
        self.truncate = truncate

        self.gru = GRU(self.output_num, stateful=True, return_sequences=True)
        self.d1 = TimeDistributed(Dense(300))
        self.d_output = TimeDistributed(Dense(self.output_num))

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.d1(x)
        x = self.d_output(x)
        return x
