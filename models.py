import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv2D, Flatten
from tensorflow.keras import Model


class GeneratorModel(Model):
    def __init__(self, n_fft, win_size):
        super(GeneratorModel, self).__init__()
        self.n_fft = n_fft
        self.win_size = win_size
        self.output_num = (n_fft // 2) + 1

        self.e_gru = GRU(self.output_num, return_sequences=True)  # Encoder
        self.e_d1 = TimeDistributed(Dense(80, activation='tanh'))
        self.e_d2 = TimeDistributed(Dense(50, activation='tanh'))
        self.l_d = TimeDistributed(Dense(25, activation='tanh'))  # Latent
        self.d_d1 = TimeDistributed(Dense(80, activation='tanh'))  # Decoder
        self.d_d2 = TimeDistributed(Dense(50, activation='tanh'))
        self.d_gru = GRU(self.output_num, return_sequences=True)

    def call(self, inputs):
        x = self.e_gru(inputs)
        x = self.e_d1(x)
        x = self.e_d2(x)
        x = self.l_d(x)
        x = self.d_d1(x)
        x = self.d_d2(x)
        x = self.d_gru(x)

        return x


class WaveGenerator(Model):
    def __init__(self, n_fft, win_size):
        super(WaveGenerator, self).__init__()
        self.n_fft = n_fft
        self.win_size = win_size

        self.model_real = GeneratorModel(n_fft, win_size)
        self.model_imag = GeneratorModel(n_fft, win_size)

    def call(self, inputs_real, inputs_imag):
        x_real = self.model_real(inputs_real)
        x_imag = self.model_imag(inputs_imag)

        x_disc_input = tf.dtypes.complex(x_real, x_imag)
        x = tf.signal.inverse_stft(x_disc_input, self.win_size, self.win_size // 2,
                                   self.n_fft, window_fn=tf.signal.hann_window)

        return x, x_real, x_imag

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(4, (9, 5), activation='relu')
        self.conv2 = Conv2D(8, (9, 5), activation='relu')
        self.conv3 = Conv2D(4, (9, 5), activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(50, activation='relu')
        self.d_out = Dense(1, activation='sigmoid')

    def call(self, inputs_real, inputs_imag):
        x = tf.dtypes.complex(inputs_real, inputs_imag)
        x = tf.math.abs(x)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d_out(x)

        return x
