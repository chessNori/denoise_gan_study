import numpy as np
import time
import load_data
import evaluate
import tensorflow as tf
import models
import datetime as dt
import os

batch_size = 16
number_batch = 4
lr = 5e-5
EPOCHS = 5000
early_stop = 0.0001
N_FFT = 512
WIN_SIZE = 320  # 20ms
SNR = 5
test_pick = [False, False, True, False]  # [Make result wav file, Calculate SNR, Make x, y wav file, Load weights]
save_scale = 2
save_time = dt.datetime.now()
save_time = save_time.strftime("%Y%m%d%H%M")
# save_time = "202308131843"
npy_path = '..\\results\\npy_backup\\spec_only_g\\' + save_time
print("Folder name: ", save_time)

noise_list = ['DKITCHEN', 'DWASHING', 'NFIELD', 'OOFFICE']  # Noise folder Names

if test_pick[0]:
    evaluate.backup_test(npy_path, save_time, save_scale, len(noise_list), 0, not test_pick[1])
    # load_path, path_time, scale, number_noise, test_number, end
else:
    os.makedirs(npy_path, exist_ok=True)

_start = time.time()
data = load_data.Data(batch_size * number_batch, WIN_SIZE)
#  number_file, win_size, min_sample=250000, frame_num=2000)

y_data = data.load_data()
print(y_data.shape)


noise_temp = data.make_noise(noise_list[0])
x_data = data.load_data(noise_temp, SNR)  # Make dataset has noise
# x_data = np.copy(y_data)  # x = y test

y_data_temp = np.copy(y_data)  # For matching shape with x_data
for _ in range(1, len(noise_list)):
    noise_temp = data.make_noise(noise_list[_])
    x_data_temp = data.load_data(noise_temp, SNR)
    x_data = np.concatenate((x_data, x_data_temp), axis=0)
    y_data = np.concatenate((y_data, y_data_temp), axis=0)


def slicing_datasets(x):
    flag = x.shape[0]//len(noise_list)
    test_res = x[:flag // number_batch]
    train_res = x[flag//number_batch:flag]
    for i in range(1, len(noise_list)):
        test_res = np.concatenate((test_res, x[flag*i:flag*i + flag//number_batch]), axis=0)
        train_res = np.concatenate((train_res, x[flag*i + flag//number_batch:flag*(i+1)]), axis=0)

    return train_res, test_res


x_data, x_data_test = slicing_datasets(x_data)
y_data, y_data_test = slicing_datasets(y_data)

print("SNR of x vs y: ", evaluate.snr(y_data, x_data))  # original, denoise

if test_pick[1]:
    evaluate.backup_snr_test(npy_path, y_data_test)
    # path, original_signal

if test_pick[2]:
    evaluate.save_raw(save_time, y_data_test, save_scale, 'y', 0, 0, batch_size=batch_size)
    evaluate.save_raw(save_time, x_data_test, save_scale, 'x0', 0, 0, batch_size=batch_size)
    evaluate.save_raw(save_time, x_data_test, save_scale, 'x1', 0, 1, batch_size=batch_size)
    evaluate.save_raw(save_time, x_data_test, save_scale, 'x2', 0, 2, batch_size=batch_size)
    evaluate.save_raw(save_time, x_data_test, save_scale, 'x3', 0, 3, batch_size=batch_size)
    # path_time, wave, scale, file_name, batch_number, noise_number

x_data = tf.signal.stft(x_data, WIN_SIZE, WIN_SIZE//2, N_FFT, window_fn=tf.signal.hann_window)
x_data_real = tf.math.real(x_data)
x_data_imag = tf.math.imag(x_data)
x_data_test = tf.signal.stft(x_data_test, WIN_SIZE, WIN_SIZE//2, N_FFT, window_fn=tf.signal.hann_window)
x_data_real_test = tf.math.real(x_data_test)
x_data_imag_test = tf.math.imag(x_data_test)


# def regularization(x, reg_val):
#     reg_temp = max(tf.experimental.numpy.max(x), abs(tf.experimental.numpy.min(x)))
#     if reg_temp > reg_val:
#         return reg_temp
#     else:
#         return reg_val
#
#
# reg = 0.
# reg = regularization(x_data_real, reg)
# reg = regularization(x_data_imag, reg)
# reg = regularization(x_data_real_test, reg)
# reg = regularization(x_data_imag_test, reg)
#
# x_data_real = tf.experimental.numpy.divide(x_data_real, reg)
# x_data_imag = tf.experimental.numpy.divide(x_data_imag, reg)
# x_data_real_test = tf.experimental.numpy.divide(x_data_real_test, reg)
# x_data_imag_test = tf.experimental.numpy.divide(x_data_imag_test, reg)

print("Data Loading is Done! (", time.time() - _start, ")")
print('Shape of train data(x,y):', x_data_real.shape, y_data.shape)
print('Shape of test data(x,y):', x_data_real_test.shape, y_data_test.shape)

_train_dataset = tf.data.Dataset.from_tensor_slices((x_data_real, x_data_imag, y_data)).shuffle(100).batch(batch_size)
_test_dataset = tf.data.Dataset.from_tensor_slices((x_data_real_test, x_data_imag_test, y_data_test)).batch(batch_size)

save_path_base = '..\\results\\saved_model\\spec_only_g\\' + save_time
save_path_real = save_path_base + '\\_model_real'
save_path_imag = save_path_base + '\\_model_imag'

_model = models.GAN(N_FFT, WIN_SIZE)

if test_pick[3]:
    _model.model_real.load_weights(save_path_real)
    _model.model_imag.load_weights(save_path_imag)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_object = tf.keras.losses.MeanSquaredError()
# loss_object = tf.keras.losses.MeanAbsoluteError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def single_train_step(noisy_wave_real, noisy_wave_imag, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _model(noisy_wave_real, noisy_wave_imag, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, _model.trainable_variables))

    train_loss(loss)
    # return denoise_wave


@tf.function
def single_test_step(noisy_wave_real, noisy_wave_imag, original_wave):
    denoise_wave = _model(noisy_wave_real, noisy_wave_imag, training=False)
    loss = loss_object(original_wave, denoise_wave)
    test_loss(loss)

    return denoise_wave


def test(train_dataset, test_dataset):
    table_temp_train = []
    table_temp_test = []
    res = None
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_state()
        test_loss.reset_state()

        for x_real, x_imag, y in train_dataset:
            single_train_step(x_real, x_imag, y)

        test_snr = 0.
        i = 0
        for x_real, x_imag, y in test_dataset:
            temp = single_test_step(x_real, x_imag, y)
            test_snr += evaluate.snr(y, temp)

            if (len(table_temp_test) != 0) and (table_temp_test[-1] < early_stop):
            # if epoch == EPOCHS - 1:
                temp = np.expand_dims(temp, axis=0)
                if i == 0:
                    res = temp
                else:
                    res = np.concatenate((res, temp), axis=0)
            i += 1

        test_snr /= i
        table_temp_train.append(train_loss.result())
        table_temp_test.append(test_loss.result())

        print(
            f'Epoch {epoch + 1}, '
            f'Train Loss: {train_loss.result()}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test SNR: {test_snr:.3f}dB, '
            f'Time: {time.time() - start} sec'
        )

        if res is not None:
            print("Early stop!")
            print("Save numpy backup")
            np.save(npy_path + '\\result_backup.npy', res)
            break

    evaluate.print_loss(table_temp_train, 'Train MSE', 'Loss')
    evaluate.print_loss(table_temp_test, 'Test MSE', 'Loss')


print("Start training real part.")
test(_train_dataset, _test_dataset)

evaluate.backup_snr_test(npy_path, y_data_test)  # path, original_signal

os.makedirs(save_path_base, exist_ok=True)

evaluate.backup_test(npy_path, save_time, save_scale, len(noise_list), 0, False)
# load_path, path_time, scale, number_noise, test_number, end
_model.model_real.save_weights(save_path_real)
_model.model_imag.save_weights(save_path_imag)
