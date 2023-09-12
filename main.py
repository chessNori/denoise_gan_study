import numpy as np
import time
import load_data
import evaluate
import tensorflow as tf
import models
import datetime as dt
import os
import gc

batch_size = 16
number_batch = 4
gen_lr = 5e-3
disc_lr = 1e-4
single_lr = 5e-3
id_imp = 4.0
EPOCHS = 1000
pre_training_stop = 8.0
gan_stop = 40.0
N_FFT = 160
WIN_SIZE = 160  # 20ms
SNR = 5
test_pick = [False, False, False, False, True]
# [Make result wav file, Calculate SNR, Make x, y wav file, Load pre_weights, Load GAN weights]
save_scale = 2
save_time = dt.datetime.now()
save_time = save_time.strftime("%Y%m%d%H%M")
save_time = "202309111330"
npy_path = '..\\results\\npy_backup\\spec_only_g\\' + save_time
save_path_base = '..\\results\\saved_model\\spec_only_g\\' + save_time
save_path_g = save_path_base + '\\_model_g'
save_path_d = save_path_base + '\\_model_d'
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

y_data = tf.signal.stft(y_data, WIN_SIZE, WIN_SIZE//2, N_FFT, window_fn=tf.signal.hann_window)
y_data_real = tf.math.real(y_data)
y_data_imag = tf.math.imag(y_data)
y_data_test = tf.signal.stft(y_data_test, WIN_SIZE, WIN_SIZE//2, N_FFT, window_fn=tf.signal.hann_window)
y_data_real_test = tf.math.real(y_data_test)
y_data_imag_test = tf.math.imag(y_data_test)


def regularization(x, reg_val):
    reg_temp = max(tf.experimental.numpy.max(x), abs(tf.experimental.numpy.min(x)))
    if reg_temp > reg_val:
        return reg_temp
    else:
        return reg_val


reg = 0.
reg = regularization(x_data_real, reg)
reg = regularization(x_data_imag, reg)
reg = regularization(x_data_real_test, reg)
reg = regularization(x_data_imag_test, reg)

x_data_real = tf.experimental.numpy.divide(x_data_real, reg)
x_data_imag = tf.experimental.numpy.divide(x_data_imag, reg)
x_data_real_test = tf.experimental.numpy.divide(x_data_real_test, reg)
x_data_imag_test = tf.experimental.numpy.divide(x_data_imag_test, reg)

reg = 0.
reg = regularization(y_data_real, reg)
reg = regularization(y_data_imag, reg)
reg = regularization(y_data_real_test, reg)
reg = regularization(y_data_imag_test, reg)

y_data_real = tf.experimental.numpy.divide(y_data_real, reg)
y_data_imag = tf.experimental.numpy.divide(y_data_imag, reg)
y_data_real_test = tf.experimental.numpy.divide(y_data_real_test, reg)
y_data_imag_test = tf.experimental.numpy.divide(y_data_imag_test, reg)

y_data = tf.dtypes.complex(y_data_real, y_data_imag)
y_data = tf.signal.inverse_stft(y_data, WIN_SIZE, WIN_SIZE // 2, N_FFT, window_fn=tf.signal.hann_window)

y_data_test = tf.dtypes.complex(y_data_real_test, y_data_imag_test)
y_data_test = tf.signal.inverse_stft(y_data_test, WIN_SIZE, WIN_SIZE // 2, N_FFT, window_fn=tf.signal.hann_window)

save_scale *= (reg * 2)

print("Data Loading is Done! (", time.time() - _start, ")")
print('Shape of train data(x,y):', x_data_real.shape, y_data_real.shape)
print('Shape of test data(x,y):', x_data_real_test.shape, y_data_real_test.shape)

_train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_data_real, x_data_imag, y_data_real, y_data_imag, y_data)).shuffle(100).batch(batch_size)
_test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_data_real_test, x_data_imag_test, y_data_real_test, y_data_imag_test, y_data_test)).batch(batch_size)

save_path_pre_g = '..\\results\\saved_model\\spec_only_g\\pre_training\\_model_pre_g'

del SNR
del _
del _start
del data
del noise_temp
del x_data_temp
del y_data_temp

gc.collect()

_model = models.WaveGenerator(N_FFT, WIN_SIZE)
_model_d = models.Discriminator()

if test_pick[3] or test_pick[4]:
    if test_pick[4]:
        _model.load_weights(save_path_g)
        _model_d.load_weights(save_path_d)
    else:
        _model.load_weights(save_path_pre_g)

optimizer = tf.keras.optimizers.Adam(learning_rate=single_lr)
loss_object = tf.keras.losses.MeanSquaredError()
# loss_object = tf.keras.losses.MeanAbsoluteError()
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr)
cross_entropy = tf.keras.losses.BinaryCrossentropy()
real_disc_accuracy = tf.keras.metrics.BinaryAccuracy(name='real discriminator accuracy')
fake_disc_accuracy = tf.keras.metrics.BinaryAccuracy(name='fake discriminator accuracy')



@tf.function
def single_train_step(noisy_wave_real, noisy_wave_imag, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave, r_, i_ = _model(noisy_wave_real, noisy_wave_imag, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, _model.trainable_variables))


@tf.function
def single_test_step(noisy_wave_real, noisy_wave_imag, original_wave):
    denoise_wave, r_, i_ = _model(noisy_wave_real, noisy_wave_imag, training=False)
    loss = loss_object(original_wave, denoise_wave)

    return denoise_wave


def pre_training(train_dataset, test_dataset):
    for epoch in range(EPOCHS):
        start = time.time()

        for x_real, x_imag, y_real, y_imag, y_wave in train_dataset:
            single_train_step(x_real, x_imag, y_wave)

        test_snr = 0.
        i = 0
        for x_real, x_imag, y_real, y_imag, y_wave in test_dataset:
            temp = single_test_step(x_real, x_imag, y_wave)
            test_snr += evaluate.snr(y_wave, temp)
            i += 1
        test_snr /= i

        print(
            f'Epoch {epoch + 1}, '
            f'Test SNR: {test_snr:.3f}dB, '
            f'Time: {time.time() - start} sec'
        )

        if test_snr > pre_training_stop:
            print("Early stop!")
            break


if not test_pick[3] and not test_pick[4]:
    print("Start pre-training.")
    pre_training(_train_dataset, _test_dataset)
    _model.save_weights(save_path_pre_g)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # un_noisy percent
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    total_loss

    real_disc_accuracy(tf.ones_like(real_output), real_output)
    fake_disc_accuracy(tf.zeros_like(fake_output), fake_output)
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def identity_loss(real, same):
    loss = tf.reduce_mean(tf.abs(real - same))
    return loss * id_imp


@tf.function
def gan_train_step(real_x, imag_x, real_y, imag_y, wave_y):
    with tf.GradientTape(persistent=True) as tape:
        generated_wave, generated_real, generated_imag = _model(real_x, imag_x, training=True)
        same_wave, _r, _i = _model(real_y, imag_y, training=True)

        real_output = _model_d(generated_real, generated_imag, training=True)
        fake_output = _model_d(real_y, imag_y, training=True)

        gen_loss = \
            generator_loss(fake_output) + identity_loss(wave_y, same_wave) + identity_loss(wave_y, generated_wave)
        disc_loss = discriminator_loss(real_output, fake_output)
    gen_gradients = tape.gradient(gen_loss, _model.trainable_variables)
    disc_gradients = tape.gradient(disc_loss, _model_d.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, _model.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, _model_d.trainable_variables))


def gan_learning(train_dataset, test_dataset):
    real_disc_accuracy.reset_state()
    fake_disc_accuracy.reset_state()

    for epoch in range(EPOCHS):
        for x_real, x_imag, y_real, y_imag, y_wave in train_dataset:
            gan_train_step(x_real, x_imag, y_real, y_imag, y_wave)

        test_snr = 0.
        i = 0
        for x_real, x_imag, y_real, y_imag, y_wave in test_dataset:
            temp = single_test_step(x_real, x_imag, y_wave)
            test_snr += evaluate.snr(y_wave, temp)
            i += 1

        end_time = dt.datetime.now()
        end_time = end_time.strftime("%H:%M")
        print(f'Epoch {epoch + 1} ({end_time})\n'
              f'Test SNR: {test_snr:.3f}dB, '
              f'Real Accuracy: {real_disc_accuracy.result() * 100:.3f}%, '
              f'Fake Accuracy: {fake_disc_accuracy.result() * 100: 3f}%')

        if test_snr > gan_stop:
            break


print("Start training.")
gan_learning(_train_dataset, _test_dataset)

print("End!")
_model.save_weights(save_path_g)
_model_d.save_weights(save_path_d)


def test(test_dataset):
    i = 0
    res = None
    for x_real, x_imag, y_real, y_imag, y_wave in test_dataset:
        res_temp = single_test_step(x_real, x_imag, y_wave)
        res_temp = np.expand_dims(res_temp, axis=0)
        if i == 0:
            res = np.copy(res_temp)
        else:
            res = np.concatenate((res, res_temp), axis=0)
        i += 1

    np.save(npy_path + '\\result_backup.npy', res)


test(_test_dataset)
os.makedirs(save_path_base, exist_ok=True)

evaluate.backup_test(npy_path, save_time, save_scale, len(noise_list), 0, False)
# load_path, path_time, scale, number_noise, test_number, end
