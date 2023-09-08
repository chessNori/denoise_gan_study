import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv1D, Flatten
from tensorflow.keras import Model


class GeneratorModel(Model):
    def __init__(self, n_fft, win_size):
        super(GeneratorModel, self).__init__()
        self.n_fft = n_fft
        self.win_size = win_size
        self.output_num = (n_fft // 2) + 1

        self.e_gru = GRU(self.output_num, return_sequences=True)  # Encoder
        self.e_d = TimeDistributed(Dense(60, activation='tanh'))
        self.l_d = TimeDistributed(Dense(40, activation='tanh'))  # Latent
        self.d_d = TimeDistributed(Dense(60, activation='tanh'))  # Decoder
        self.d_gru = GRU(self.output_num, return_sequences=True)
        self.d_output = TimeDistributed(Dense(self.output_num))

    def call(self, inputs):
        x = self.e_gru(inputs)
        x = self.e_d(x)
        x = self.l_d(x)
        x = self.d_d(x)
        x = self.d_gru(x)
        x = self.d_output(x)

        return x


class WaveGenerator(Model):
    def __init__(self, n_fft, win_size):
        super(WaveGenerator, self).__init__()
        self.n_fft = n_fft
        self.win_size = win_size

        self.model_real = GeneratorModel(n_fft, win_size)
        self.model_imag = GeneratorModel(n_fft, win_size)

    def call(self, inputs):
        inputs_real = tf.signal.stft(inputs, self.win_size, self.win_size // 2,
                                     self.n_fft, window_fn=tf.signal.hann_window)
        inputs_imag = tf.math.imag(inputs_real)
        inputs_real = tf.math.real(inputs_real)

        x_real = self.model_real(inputs_real)
        x_imag = self.model_imag(inputs_imag)

        x = tf.dtypes.complex(x_real, x_imag)
        x = tf.signal.inverse_stft(x, self.win_size, self.win_size // 2, self.n_fft, window_fn=tf.signal.hann_window)

        return x


class DiscriminatorModel(Model):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        self.conv1 = Conv1D(4, 15, activation='relu')
        self.conv2 = Conv1D(4, 7, activation='relu')
        self.conv3 = Conv1D(1, 5, activation='relu')
        self.flatten = Flatten()
        self.d = Dense(40, activation='relu')
        self.result = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d(x)
        x = self.result(x)

        return x


# class DiscriminatorModel(Model):
#     def __init__(self):
#         super(DiscriminatorModel, self).__init__()
#         self.gru = GRU(160, return_sequences=True)
#         self.d1 = TimeDistributed(Dense(80, activation='tanh'))
#         self.gru2 = GRU(40)
#         self.d = Dense(40, activation='tanh')
#         self.result = Dense(1, activation='sigmoid')
#
#     def call(self, inputs):
#         x = self.gru(inputs)
#         x = self.d1(x)
#         x = self.gru2(x)
#         x = self.d(x)
#         x = self.result(x)
#
#         return x


# class GAN(Model):
#     def __init__(self, n_fft, win_size):
#         super(GAN, self).__init__()
#         self.n_fft = n_fft
#         self.win_size = win_size
#         self.generator = WaveGenerator(n_fft, win_size)
#         self.discriminator = DiscriminatorModel()
#
#     def call(self, inputs_real, inputs_imag, generator_train, inputs_origin=None):
#         if generator_train:
#             denoise = self.generator(inputs_real, inputs_imag)
#             x = tf.reshape(denoise, (16, 320, -1))
#             x = self.discriminator(x)
#             return denoise, x
#         else:
#             if inputs_origin is None:
#                 inputs = tf.dtypes.complex(inputs_real, inputs_imag)
#                 inputs = tf.signal.inverse_stft(inputs, self.win_size, self.win_size // 2, self.n_fft,
#                                                 window_fn=tf.signal.hann_window)
#                 inputs = tf.reshape(inputs, (16, 320, -1))
#             else:
#                 inputs = inputs_origin
#                 inputs = tf.reshape(inputs, (16, 320, -1))
#             x = self.discriminator(inputs)
#             return x
