import numpy as np
import time
import load_data
import evaluate
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Flatten, Concatenate, Conv2D, RepeatVector, TimeDistributed, MaxPool2D
from tensorflow.keras import Model

noise_list = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'OOFFICE']  # Folder Names
evaluate.backup_test(len(noise_list), 1, n_fft=128)

batch_size = 16
number_batch = 8
lr = 1e-4
EPOCHS = 150
save_path = '..\\saved_model\\2023_07_10_single_denoise_model'
early_stop = 5.0e-7

start = time.time()
data = load_data.Data(batch_size * number_batch, batch_size, n_fft=256, win_size=256)

y_data_real, y_data_imag = data.load_data()

noise_temp = data.make_noise(noise_list[0])
x_data_real, x_data_imag = data.load_data(noise_temp)

y_data_real_temp = y_data_real
y_data_imag_temp = y_data_imag
for i in range(1, len(noise_list)):
    noise_temp = data.make_noise(noise_list[i])
    x_data_real_temp, x_data_imag_temp = data.load_data(noise_temp)
    x_data_real = np.concatenate((x_data_real, x_data_real_temp), axis=0)
    x_data_imag = np.concatenate((x_data_imag, x_data_imag_temp), axis=0)
    y_data_real = np.concatenate((y_data_real, y_data_real_temp), axis=0)
    y_data_imag = np.concatenate((y_data_imag, y_data_imag_temp), axis=0)

x_data_real /= data.regularization
x_data_imag /= data.regularization
y_data_real /= data.regularization
y_data_imag /= data.regularization

x_data_real_test = x_data_real[:x_data_real.shape[0]//number_batch]
y_data_real_test = y_data_real[:y_data_real.shape[0]//number_batch]
x_data_imag_test = x_data_imag[:x_data_imag.shape[0]//number_batch]
y_data_imag_test = y_data_imag[:y_data_imag.shape[0]//number_batch]

# evaluate.backup_snr_test(y_data_real_test, y_data_imag_test)
# print(evaluate.snr(y_data_real_test, y_data_imag_test, x_data_real_test, x_data_imag_test))

x_data_real = x_data_real[x_data_real.shape[0]//number_batch:]
y_data_real = y_data_real[y_data_real.shape[0]//number_batch:]
x_data_imag = x_data_imag[x_data_imag.shape[0]//number_batch:]
y_data_imag = y_data_imag[y_data_imag.shape[0]//number_batch:]

evaluate.save_raw(x_data_real_test, x_data_imag_test, 2000000, 'x', len(noise_list))
# evaluate.save_raw(y_data_real_test, y_data_imag_test, 2000000, 'y', len(noise_list))

print("Data Loading is Done! (", time.time() - start, ")")
print('Shape of train data(x,y):', x_data_real.shape, y_data_real.shape)
print('Shape of test data(x,y):', x_data_real_test.shape, y_data_real_test.shape)
print('Regularization:', data.regularization)

train_real_dataset = tf.data.Dataset.from_tensor_slices((x_data_real, y_data_real))
test_real_dataset = tf.data.Dataset.from_tensor_slices((x_data_real_test, y_data_real_test))
train_imag_dataset = tf.data.Dataset.from_tensor_slices((x_data_imag, y_data_imag))
test_imag_dataset = tf.data.Dataset.from_tensor_slices((x_data_imag_test, y_data_imag_test))


class GeneratorModel(Model):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        self.output_num = data.n_fft//2+1
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


_model_real = GeneratorModel()
_model_imag = GeneratorModel()

optimizer_real = tf.keras.optimizers.Adam(learning_rate=lr)
optimizer_imag = tf.keras.optimizers.Adam(learning_rate=lr)

loss_object = tf.keras.losses.MeanSquaredError()
# loss_object = tf.keras.losses.MeanAbsoluteError()

train_real_loss = tf.keras.metrics.Mean(name='train_real_loss')
test_real_loss = tf.keras.metrics.Mean(name='test_real_loss')
train_imag_loss = tf.keras.metrics.Mean(name='train_imag_loss')
test_imag_loss = tf.keras.metrics.Mean(name='test_imag_loss')


@tf.function
def single_train_step_real(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _model_real(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model_real.trainable_variables)
    optimizer_real.apply_gradients(zip(gradients, _model_real.trainable_variables))

    train_real_loss(loss)


@tf.function
def single_train_step_imag(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _model_imag(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model_imag.trainable_variables)
    optimizer_imag.apply_gradients(zip(gradients, _model_imag.trainable_variables))

    train_imag_loss(loss)


@tf.function
def single_test_step_real(noisy_wave, original_wave):
    denoise_wave = _model_real(noisy_wave, training=False)
    loss = loss_object(original_wave, denoise_wave)
    test_real_loss(loss)

    return denoise_wave


@tf.function
def single_test_step_imag(noisy_wave, original_wave):
    denoise_wave = _model_imag(noisy_wave, training=False)
    loss = loss_object(original_wave, denoise_wave)
    test_imag_loss(loss)

    return denoise_wave


table_temp_train_real = []  # loss result
table_temp_test_real = []
table_temp_train_imag = []
table_temp_test_imag = []
res_real = None
res_imag = None

for epoch in range(EPOCHS):
    start = time.time()
    train_real_loss.reset_state()
    test_real_loss.reset_state()
    train_imag_loss.reset_state()
    test_imag_loss.reset_state()

    if res_real is None:
        i = 0
        for x_wave_real, y_wave_real in train_real_dataset:
            if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
                _model_real.e_gru.reset_states()
                _model_real.d_gru.reset_states()

            single_train_step_real(x_wave_real, y_wave_real)

            i += 1

        _model_real.e_gru.reset_states()
        _model_real.d_gru.reset_states()

        i = 0
        for x_wave_real, y_wave_real in test_real_dataset:
            if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
                _model_real.e_gru.reset_states()
                _model_real.d_gru.reset_states()

            temp_real = single_test_step_real(x_wave_real, y_wave_real)

            if (len(table_temp_test_real) != 0) and (table_temp_test_real[-1] < early_stop):
                temp_real = np.expand_dims(temp_real, axis=0)
                if i == 0:
                    res_real = temp_real
                else:
                    res_real = np.concatenate((res_real, temp_real), axis=0)
            i += 1

        _model_real.e_gru.reset_states()
        _model_real.d_gru.reset_states()
        table_temp_train_real.append(train_real_loss.result())
        table_temp_test_real.append(test_real_loss.result())

    if res_imag is None:
        i = 0
        for x_wave_imag, y_wave_imag in train_imag_dataset:
            if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
                _model_imag.e_gru.reset_states()
                _model_imag.d_gru.reset_states()

            single_train_step_imag(x_wave_imag, y_wave_imag)

            i += 1

        _model_imag.e_gru.reset_states()
        _model_imag.d_gru.reset_states()

        i = 0
        for x_wave_imag, y_wave_imag in test_imag_dataset:
            if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
                _model_imag.e_gru.reset_states()
                _model_imag.d_gru.reset_states()

            temp_imag = single_test_step_imag(x_wave_imag, y_wave_imag)

            if (len(table_temp_test_imag) != 0) and (table_temp_test_imag[-1] < early_stop):
                temp_imag = np.expand_dims(temp_imag, axis=0)
                if i == 0:
                    res_imag = temp_imag
                else:
                    res_imag = np.concatenate((res_imag, temp_imag), axis=0)
            i += 1

        _model_imag.e_gru.reset_states()
        _model_imag.d_gru.reset_states()
        table_temp_train_imag.append(train_imag_loss.result())
        table_temp_test_imag.append(test_imag_loss.result())

    print(
        f'Epoch {epoch + 1}, '
        f'Real Train Loss: {train_real_loss.result()}, '
        f'Real Test Loss: {test_real_loss.result()}, '
        f'Imag Train Loss: {train_imag_loss.result()}, '
        f'Imag Test Loss: {test_imag_loss.result()}, '
        f'Time: {time.time() - start} sec'
    )

    if res_real is not None and res_imag is not None:
        print('Save raw files...')
        np.save('result_real_backup.npy', res_real)
        np.save('result_imag_backup.npy', res_imag)
        evaluate.save_raw(res_real, res_imag, 2000000, 'test', len(noise_list),
                          n_fft=data.n_fft, win_size=data.win_size)
        break

evaluate.print_loss(table_temp_train_real, 'Train MSE(real)', 'Loss')
evaluate.print_loss(table_temp_test_real, 'Test MSE(real)', 'Loss')
evaluate.print_loss(table_temp_train_imag, 'Train MSE(imag)', 'Loss')
evaluate.print_loss(table_temp_test_imag, 'Test MSE(imag)', 'Loss')
