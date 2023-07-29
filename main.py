import numpy as np
import time
import load_data
import evaluate
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Flatten, Concatenate, Conv2D, RepeatVector, TimeDistributed, MaxPool2D
from tensorflow.keras import Model

noise_list = ['DKITCHEN', 'DLIVING', 'DWASHING', 'OOFFICE']  # Folder Names
evaluate.backup_test(len(noise_list), 4)

batch_size = 16
number_batch = 6
lr = 1e-4
EPOCHS = 150
save_path = '..\\saved_model\\2023_07_10_single_denoise_model'

start = time.time()
data = load_data.Data(batch_size * number_batch, batch_size)

y_data = data.load_data()

noise_temp = data.make_noise(noise_list[0])
x_data = data.load_data(noise_temp)

y_data_temp = y_data
phase_temp = data.phase
for i in range(1, len(noise_list)):
    noise_temp = data.make_noise(noise_list[i])
    x_data_temp = data.load_data(noise_temp)
    x_data = np.concatenate((x_data, x_data_temp), axis=0)
    y_data = np.concatenate((y_data, y_data_temp), axis=0)
    data.phase = np.concatenate((data.phase, phase_temp), axis=0)

x_data /= data.regularization
y_data /= data.regularization
x_data_temp = None
y_data_temp = None
phase_temp = None

x_data_test = x_data[:x_data.shape[0]//number_batch]
y_data_test = y_data[:y_data.shape[0]//number_batch]

# evaluate.backup_snr_test(y_data_test)

x_data = x_data[x_data.shape[0]//number_batch:]
y_data = y_data[y_data.shape[0]//number_batch:]

evaluate.save_raw(x_data_test, data.phase[:x_data_test.shape[0]], 2000000, 'x', len(noise_list))
evaluate.save_raw(y_data_test, data.phase[:y_data_test.shape[0]], 2000000, 'y', len(noise_list))

print("Data Loading is Done! (", time.time() - start, ")")
print('Shape of train data(x,y):', x_data.shape, y_data.shape)
print('Shape of test data(x,y):', x_data_test.shape, y_data_test.shape)
print('Regularization:', data.regularization)
print(data.phase.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
test_dataset = tf.data.Dataset.from_tensor_slices((x_data_test, y_data_test))


class GeneratorModel(Model):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        self.output_num = data.n_fft//2+1
        self.e_gru = GRU(120, stateful=True, return_sequences=True)  # Encoder
        self.e_d = TimeDistributed(Dense(40))  # latent
        self.d_gru = GRU(120, stateful=True, return_sequences=True)  # Decoder
        self.d_output = TimeDistributed(Dense(self.output_num))

    def call(self, inputs):
        x = self.e_gru(inputs)
        x = self.e_d(x)
        x = self.d_gru(x)
        x = self.d_output(x)

        return x


class MHSModel(Model):
    def __init__(self):
        super(MHSModel, self).__init__()
        self.e_gru = GRU(128, return_sequences=True)
        self.d1 = TimeDistributed(Dense(128))
        self.d2 = TimeDistributed(Dense(65))

    def call(self, inputs):
        x = self.e_gru(inputs)
        x = self.d1(x)
        x = self.d2(x)

        return x


_single_generator = GeneratorModel()
# _single_generator = MHSModel()

single_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

loss_object = tf.keras.losses.MeanSquaredError()
# loss_object = tf.keras.losses.MeanAbsoluteError()

single_train_loss = tf.keras.metrics.Mean(name='train_loss')
single_test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def single_train_step(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _single_generator(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _single_generator.trainable_variables)
    single_generator_optimizer.apply_gradients(zip(gradients, _single_generator.trainable_variables))

    single_train_loss(loss)

    return denoise_wave


@tf.function
def single_test_step(noisy_wave, original_wave):
    denoise_wave = _single_generator(noisy_wave, training=False)
    loss = loss_object(original_wave, denoise_wave)

    single_test_loss(loss)

    return denoise_wave


table_temp_train = []  # loss result
table_temp_test = []

for epoch in range(EPOCHS):
    start = time.time()
    single_train_loss.reset_state()
    single_test_loss.reset_state()
    res = None

    i = 0
    for x_wave, y_wave in train_dataset:
        if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
            _single_generator.e_gru.reset_states()
            _single_generator.d_gru.reset_states()

        single_train_step(x_wave, y_wave)

        i += 1

    _single_generator.e_gru.reset_states()
    _single_generator.d_gru.reset_states()
    i = 0
    for x_wave, y_wave in test_dataset:
        if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
            _single_generator.e_gru.reset_states()
            _single_generator.d_gru.reset_states()

        temp = single_test_step(x_wave, y_wave)

        if (len(table_temp_test) != 0) and (table_temp_test[-1] < 1.0e-6):
            temp = np.expand_dims(temp, axis=0)
            if i == 0:
                res = temp
            else:
                res = np.concatenate((res, temp), axis=0)

        i += 1

    print(
        f'Epoch {epoch + 1}, '
        f'Single Train Loss: {single_train_loss.result()}, '
        f'Test Loss: {single_test_loss.result()}, '
        f'Time: {time.time() - start} sec'
    )

    _single_generator.e_gru.reset_states()
    _single_generator.d_gru.reset_states()
    table_temp_train.append(single_train_loss.result())
    table_temp_test.append(single_test_loss.result())

    if res is not None:
        print('Save raw files...')
        np.save('result_backup.npy', res)
        np.save('phase_backup.npy', data.phase[:res.shape[0]])
        evaluate.save_raw(res, data.phase[:res.shape[0]], 2000000, 'test', len(noise_list))
        break

evaluate.print_loss(table_temp_train, 'Train MSE(real, denoise)', 'Loss')
evaluate.print_loss(table_temp_test, 'Test MSE(real, denoise)', 'Loss')
