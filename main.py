import numpy as np
import time
import load_data
import evaluate
import tensorflow as tf
import models
import datetime as dt
import os

batch_size = 16
number_batch = 8
lr = 1e-4
EPOCHS = 1500
early_stop = 0.00018
N_FFT = 256
WIN_SIZE = 160  # 10ms
test_pick = [False, False, False]  # [Make result wav file, Calculate SNR, Make x, y wav file]
save_scale = 2
save_time = dt.datetime.now()
save_time = save_time.strftime("%Y%m%d%H%M")
# save_time = "202307310420"
npy_path = '..\\results\\npy_backup\\spec_only_g\\' + save_time
print("Folder name: ", save_time)

noise_list = ['DKITCHEN']  #, 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'OOFFICE']  # Noise folder Names

if test_pick[0]:
    evaluate.backup_test(npy_path, save_time, save_scale, len(noise_list), number_batch,
                         0, N_FFT, WIN_SIZE, not test_pick[1])
    # load_path, path_time, scale, number_noise, test_number, n_fft, win_size, end
else:
    os.makedirs(npy_path, exist_ok=True)

_start = time.time()
data = load_data.Data(batch_size * number_batch, batch_size, N_FFT, WIN_SIZE)
#  number_file, batch_size, n_fft, win_size, min_sample=250000, frame_num=1500, truncate=100)

y_data_real, y_data_imag = data.load_data()

# noise_temp = data.make_noise(noise_list[0])
# x_data_real, x_data_imag = data.load_data(noise_temp)  # Make dataset has noise

x_data_real, x_data_imag = np.copy(y_data_real), np.copy(y_data_imag)  # x = y test

y_data_real_temp, y_data_imag_temp = y_data_real, y_data_imag  # For matching shape with x_data
# for _ in range(1, len(noise_list)):
#     noise_temp = data.make_noise(noise_list[_])
#     x_data_real_temp, x_data_imag_temp = data.load_data(noise_temp)
#     x_data_real = np.concatenate((x_data_real, x_data_real_temp), axis=0)
#     x_data_imag = np.concatenate((x_data_imag, x_data_imag_temp), axis=0)
#     y_data_real = np.concatenate((y_data_real, y_data_real_temp), axis=0)
#     y_data_imag = np.concatenate((y_data_imag, y_data_imag_temp), axis=0)

x_data_real /= data.regularization
x_data_imag /= data.regularization  # -1.0 ~ +1.0

# x_data_real_test = x_data_real[:x_data_real.shape[0]//number_batch]
# y_data_real_test = y_data_real[:y_data_real.shape[0]//number_batch]
# x_data_imag_test = x_data_imag[:x_data_imag.shape[0]//number_batch]
# y_data_imag_test = y_data_imag[:y_data_imag.shape[0]//number_batch]  # Test datasets
#
# if test_pick[1]:
#     evaluate.backup_snr_test(npy_path, y_data_real_test, y_data_imag_test)
#
# x_data_real = x_data_real[x_data_real.shape[0]//number_batch:]
# y_data_real = y_data_real[y_data_real.shape[0]//number_batch:]
# x_data_imag = x_data_imag[x_data_imag.shape[0]//number_batch:]
# y_data_imag = y_data_imag[y_data_imag.shape[0]//number_batch:]  # Train datasets
#
if test_pick[2]:
    evaluate.save_raw(save_time, x_data_real, x_data_imag, save_scale,
                      'x', len(noise_list), number_batch, 0, 0, N_FFT, WIN_SIZE)
    evaluate.save_raw(save_time, y_data_real, y_data_imag, save_scale // data.regularization,
                      'y', len(noise_list), number_batch, 0, 0, N_FFT, WIN_SIZE)
    # path_time, wave_real, wave_imag, scale, file_name, number_noise, number_batch,
    # batch, noise_number, n_fft, win_size

print("Data Loading is Done! (", time.time() - _start, ")")
print('Shape of train data(x,y):', x_data_real.shape, y_data_real.shape)
# print('Shape of test data(x,y):', x_data_real_test.shape, y_data_real_test.shape)
print('Regularization:', data.regularization)
# print("SNR of x vs y: ", evaluate.snr(y_data_real_test, y_data_imag_test, x_data_real_test, x_data_imag_test))

train_real_dataset = tf.data.Dataset.from_tensor_slices((x_data_real, y_data_real))
# test_real_dataset = tf.data.Dataset.from_tensor_slices((x_data_real_test, y_data_real_test))
train_imag_dataset = tf.data.Dataset.from_tensor_slices((x_data_imag, y_data_imag))
# test_imag_dataset = tf.data.Dataset.from_tensor_slices((x_data_imag_test, y_data_imag_test))

_model_real = models.GeneratorModel(data.n_fft, data.batch_size, data.truncate)
_model_imag = models.GeneratorModel(data.n_fft, data.batch_size, data.truncate)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

loss_object = tf.keras.losses.MeanSquaredError()
# loss_object = tf.keras.losses.MeanAbsoluteError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
# test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def single_train_step_real(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _model_real(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model_real.trainable_variables)
    optimizer.apply_gradients(zip(gradients, _model_real.trainable_variables))

    train_loss(loss)
    return denoise_wave


# @tf.function
# def single_test_step_real(noisy_wave, original_wave):
#     denoise_wave = _model_real(noisy_wave, training=False)
#     loss = loss_object(original_wave, denoise_wave)
#     test_loss(loss)
#
#     return denoise_wave


@tf.function
def single_train_step_imag(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _model_imag(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model_imag.trainable_variables)
    optimizer.apply_gradients(zip(gradients, _model_imag.trainable_variables))

    train_loss(loss)
    return denoise_wave


# @tf.function
# def single_test_step_imag(noisy_wave, original_wave):
#     denoise_wave = _model_imag(noisy_wave, training=False)
#     loss = loss_object(original_wave, denoise_wave)
#     test_loss(loss)
#
#     return denoise_wave


def single_train_step(x_data, y_data, real):
    if real:
        temp = single_train_step_real(x_data, y_data)
    else:
        temp = single_train_step_imag(x_data, y_data)

    return temp


# def single_test_step(x_data, y_data, real):
#     if real:
#         temp = single_test_step_real(x_data, y_data)
#     else:
#         temp = single_test_step_imag(x_data, y_data)
#
#     return temp


def who_reset(real):
    if real:
        _model_real.e_gru.reset_states()
        _model_real.d_gru.reset_states()
    else:
        _model_imag.e_gru.reset_states()
        _model_imag.d_gru.reset_states()


def test(train_dataset, test_dataset, real):
    table_temp_train = []
    # table_temp_test = []
    res = None
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_state()
        # test_loss.reset_state()

        i = 0
        for x_data, y_data in train_dataset:
            if (i != 0) and (i % (data.frame_num // data.truncate) == 0):
                who_reset(real)

            temp = single_train_step(x_data, y_data, real)

            if (len(table_temp_train) != 0) and (table_temp_train[-1] < early_stop):
            # if epoch == EPOCHS - 1:
                temp = np.expand_dims(temp, axis=0)
                if i == 0:
                    res = temp
                else:
                    res = np.concatenate((res, temp), axis=0)
            i += 1

        who_reset(real)
        # i = 0
        # for x_data, y_data in test_dataset:
        #     if (i != 0) and (i % (data.frame_num // data.truncate) == 0):
        #         who_reset(real)
        #
        #     temp = single_test_step(x_data, y_data, real)
        #
        # if (len(table_temp_test) != 0) and (table_temp_test[-1] < early_stop):
        #     temp = np.expand_dims(temp, axis=0)
        #     if i == 0:
        #         res = temp
        #     else:
        #         res = np.concatenate((res, temp), axis=0)
        #
        # i += 1
        #
        # who_reset(real)
        table_temp_train.append(train_loss.result())
        # table_temp_test.append(test_loss.result())

        print(
            f'Epoch {epoch + 1}, '
            f'Train Loss: {train_loss.result()}, '
            # f'Test Loss: {test_loss.result()}, '
            f'Time: {time.time() - start} sec'
        )

        if res is not None:
            print("Early stop!")
            print('Save raw files...')
            if real:
                np.save(npy_path + '\\result_real_backup.npy', res)
            else:
                np.save(npy_path + '\\result_imag_backup.npy', res)
            break

    if real:
        evaluate.print_loss(table_temp_train, 'Train MSE(real)', 'Loss')
        # evaluate.print_loss(table_temp_test, 'Test MSE(real)', 'Loss')
    else:
        evaluate.print_loss(table_temp_train, 'Train MSE(imag)', 'Loss')
        # evaluate.print_loss(table_temp_test, 'Test MSE(imag)', 'Loss')


print("Start training real part.")
# test(train_real_dataset, test_real_dataset, True)
test(train_real_dataset, None, True)
print("Start training imaginary part")
# test(train_imag_dataset, test_imag_dataset, False)
test(train_imag_dataset, None, False)

save_path_base = '..\\results\\saved_model\\spec_only_g\\' + save_time
save_path_real = save_path_base + '\\_model_real'
save_path_imag = save_path_base + '\\_model_imag'
os.makedirs(save_path_base, exist_ok=True)

evaluate.backup_test(npy_path, save_time, save_scale, len(noise_list), number_batch, 0, N_FFT, WIN_SIZE, False)
_model_real.save_weights(save_path_real)
_model_imag.save_weights(save_path_imag)
