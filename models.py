from tensorflow.keras.layers import GRU, Dense, Flatten, Concatenate, Conv2D, RepeatVector, TimeDistributed, MaxPool2D
from tensorflow.keras import Model


class GeneratorModel(Model):
    def __init__(self, n_fft):
        super(GeneratorModel, self).__init__()
        self.output_num = n_fft//2+1
        self.e_gru = GRU(120, stateful=True, return_sequences=True)  # Encoder
        self.l_d = TimeDistributed(Dense(40))  # latent
        self.d_gru = GRU(120, stateful=True, return_sequences=True)  # Decoder
        self.d_output = TimeDistributed(Dense(self.output_num))

    def call(self, inputs):
        x = self.e_gru(inputs)
        x = self.l_d(x)
        x = self.d_gru(x)
        x = self.d_output(x)

        return x
