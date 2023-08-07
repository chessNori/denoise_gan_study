import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import Model


class GeneratorModel(Model):
    def __init__(self, n_fft):
        super(GeneratorModel, self).__init__()
        self.output_num = n_fft//2

        self.e_gru = GRU(self.output_num, stateful=True, return_sequences=True)  # Encoder
        self.e_d = TimeDistributed(Dense(100, activation='tanh'))
        self.l_d = TimeDistributed(Dense(25, activation='tanh'))  # Latent
        self.d_d = TimeDistributed(Dense(100, activation='tanh'))  # Decoder
        self.d_gru = GRU(self.output_num, stateful=True, return_sequences=True)
        self.d_output = TimeDistributed(Dense(self.output_num))

    def call(self, inputs):
        x = self.e_gru(inputs)
        x = self.e_d(x)
        x = self.l_d(x)
        x = self.d_d(x)
        x = self.d_gru(x)
        x = self.d_output(x)
        return x


class DiscriminatorModel(Model):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        self.conv1 = Conv2D(8, 5, activation='tanh')
        self.conv2 = Conv2D(16, 3, activation='tanh')
        self.max_pool = MaxPool2D(pool_size=(3, 3))
        self.conv3 = Conv2D(1, 3, activation='tanh')
        self.flatten = Flatten()
        self.d = Dense(20, activation='relu')
        self.result = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d(x)
        x = self.result(x)

        return x


class GAN(Model):
    def __init__(self, n_fft, regularization):
        super(GAN, self).__init__()
        self.regularization = regularization
        self.generator = GeneratorModel(n_fft)
        self.discriminator = DiscriminatorModel()

    def call(self, inputs, generator_train):
        if generator_train:
            denoise = self.generator(inputs)
            x = self.discriminator(denoise / self.regularization)
            return denoise, x
        else:
            x = self.discriminator(inputs / self.regularization)
            return x
