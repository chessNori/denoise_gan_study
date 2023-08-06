import numpy as np
import time
import load_data
import evaluate
import tensorflow as tf
import models
import datetime as dt
import os

batch_size = 16
number_batch = 15
gen_lr = 1e-5
disc_lr = 1e-4
single_lr = 5e-5
EPOCHS = 500
pre_training_stop = 0.01
phase_shift_loss = 1e-3
early_stop = 5e-4
wait_threshold = 15
N_FFT = 256
WIN_SIZE = 160  # 20ms
SNR = 20
save_scale = 3
save_time = dt.datetime.now()
save_time = save_time.strftime("%Y%m%d%H%M")
# save_time = "202308051634"
npy_path = '..\\results\\npy_backup\\spec_only_g\\' + save_time
print("Folder name: ", save_time)
noise_list = ['DKITCHEN', 'DWASHING', 'NFIELD', 'OOFFICE']  # Noise folder Names

test_pick = [False, False, True, False]
# [Make result wav file, Calculate SNR, Make x, y wav file, Load weights]

if test_pick[0]:
    evaluate.backup_test(npy_path, save_time, save_scale, len(noise_list), 0, N_FFT, WIN_SIZE, not test_pick[1])
    # load_path, path_time, scale, number_noise, test_number, n_fft, win_size, end
else:
    os.makedirs(npy_path, exist_ok=True)

_start = time.time()
data = load_data.Data(batch_size * number_batch, batch_size, N_FFT, WIN_SIZE, frame_num=1000)
#  number_file, batch_size, n_fft, win_size, min_sample=250000, frame_num=2000, truncate=100)

y_data_real, y_data_imag = data.load_data()

noise_temp = data.make_noise(noise_list[0])
x_data_real, x_data_imag = data.load_data(noise_temp, SNR)  # Make dataset has noise
# x_data_real, x_data_imag = np.copy(y_data_real), np.copy(y_data_imag)  # x = y test

y_data_real_temp, y_data_imag_temp = y_data_real, y_data_imag  # For matching shape with x_data
for _ in range(1, len(noise_list)):
    noise_temp = data.make_noise(noise_list[_])
    x_data_real_temp, x_data_imag_temp = data.load_data(noise_temp, SNR)
    x_data_real = np.concatenate((x_data_real, x_data_real_temp), axis=0)
    x_data_imag = np.concatenate((x_data_imag, x_data_imag_temp), axis=0)
    y_data_real = np.concatenate((y_data_real, y_data_real_temp), axis=0)
    y_data_imag = np.concatenate((y_data_imag, y_data_imag_temp), axis=0)

x_data_real /= data.regularization
x_data_imag /= data.regularization  # -1.0 ~ +1.0


def slicing_datasets(x):
    flag = x.shape[0]//len(noise_list)
    test_res = x[:flag // number_batch]
    train_res = x[flag//number_batch:flag]
    for i in range(1, len(noise_list)):
        test_res = np.concatenate((test_res, x[flag*i:flag*i + flag//number_batch]), axis=0)
        train_res = np.concatenate((train_res, x[flag*i + flag//number_batch:flag*(i+1)]), axis=0)

    return train_res, test_res


x_data_real, x_data_real_test = slicing_datasets(x_data_real)
y_data_real, y_data_real_test = slicing_datasets(y_data_real)
x_data_imag, x_data_imag_test = slicing_datasets(x_data_imag)
y_data_imag, y_data_imag_test = slicing_datasets(y_data_imag)

if test_pick[1]:
    evaluate.backup_snr_test(npy_path, y_data_real_test, y_data_imag_test)

if test_pick[2]:
    evaluate.save_raw(save_time, y_data_real_test, y_data_imag_test, save_scale,
                      'y', len(noise_list), 0, 0, N_FFT, WIN_SIZE)
    evaluate.save_raw(save_time, x_data_real_test, x_data_imag_test, save_scale * data.regularization,
                      'x0', len(noise_list), 0, 0, N_FFT, WIN_SIZE)
    evaluate.save_raw(save_time, x_data_real_test, x_data_imag_test, save_scale * data.regularization,
                      'x2', len(noise_list), 0, 2, N_FFT, WIN_SIZE)
    # path_time, wave_real, wave_imag, scale, file_name, number_noise, batch, noise_number, n_fft, win_size

print("Data Loading is Done! (", time.time() - _start, ")")
print('Shape of train data(x,y):', x_data_real.shape, y_data_real.shape)
print('Shape of test data(x,y):', x_data_real_test.shape, y_data_real_test.shape)
print('Regularization:', data.regularization)
print("SNR of x vs y: ", evaluate.snr(y_data_real_test, y_data_imag_test,
                                      x_data_real_test * data.regularization, x_data_imag_test * data.regularization))

train_real_dataset = tf.data.Dataset.from_tensor_slices((x_data_real, y_data_real))
test_real_dataset = tf.data.Dataset.from_tensor_slices((x_data_real_test, y_data_real_test))
train_imag_dataset = tf.data.Dataset.from_tensor_slices((x_data_imag, y_data_imag))
test_imag_dataset = tf.data.Dataset.from_tensor_slices((x_data_imag_test, y_data_imag_test))

save_path_base = '..\\results\\saved_model\\spec_only_g\\' + save_time
save_path_real = save_path_base + '\\_model_real'
save_path_imag = save_path_base + '\\_model_imag'

_model_real = models.GAN(data.n_fft)
_model_imag = models.GAN(data.n_fft)

if test_pick[3]:
    _model_real.load_weights(save_path_real)
    _model_imag.load_weights(save_path_imag)

optimizer = tf.keras.optimizers.Adam(learning_rate=single_lr)
loss_object = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def single_train_step_real(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _model_real.generator(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model_real.generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, _model_real.generator.trainable_variables))

    train_loss(loss)


@tf.function
def single_train_step_imag(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _model_imag.generator(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model_imag.generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, _model_imag.generator.trainable_variables))

    train_loss(loss)


@tf.function
def single_test_step_real(noisy_wave, original_wave):
    denoise_wave = _model_real.generator(noisy_wave, training=False)
    loss = loss_object(original_wave, denoise_wave)
    test_loss(loss)

    return denoise_wave


@tf.function
def single_test_step_imag(noisy_wave, original_wave):
    denoise_wave = _model_imag.generator(noisy_wave, training=False)
    loss = loss_object(original_wave, denoise_wave)
    test_loss(loss)

    return denoise_wave


def single_train_step(noisy_wave, original_wave, real):
    if real:
        single_train_step_real(noisy_wave, original_wave)
    else:
        single_train_step_imag(noisy_wave, original_wave)


def single_test_step(noisy_wave, original_wave, real):
    if real:
        res = single_test_step_real(noisy_wave, original_wave)
    else:
        res = single_test_step_imag(noisy_wave, original_wave)

    return res


def reset_gru(real):
    if real:
        _model_real.generator.e_gru.reset_states()
    else:
        _model_imag.generator.d_gru.reset_states()


def single_learning(train_dataset, test_dataset, model, real):
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_state()
        test_loss.reset_state()

        i = 0
        for x_data, y_data in train_dataset:
            if (i != 0) and (i % (data.frame_num // data.truncate) == 0):
                reset_gru(model)

            single_train_step(x_data, y_data, real)
            i += 1

        i = 0
        for x_data, y_data in test_dataset:
            if i % (data.frame_num // data.truncate) == 0:
                reset_gru(model)

            single_test_step(x_data, y_data, real)
            i += 1

        reset_gru(model)

        print(
            f'Epoch {epoch + 1}, '
            f'Train Loss: {train_loss.result()}, '
            f'Test Loss: {test_loss.result()}, '
            f'Time: {time.time() - start} sec'
        )

        if test_loss.result() < pre_training_stop:
            break


print("Start pre-training real part.")
single_learning(train_real_dataset, test_real_dataset, _model_real, True)
print("Start pre-training imaginary part")
single_learning(train_imag_dataset, test_imag_dataset, _model_imag, False)

gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr)
cross_entropy = tf.keras.losses.BinaryCrossentropy()
reference_loss = tf.keras.metrics.Mean(name='reference_loss')  # loss for starting single train phase


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # un_noisy percent
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    total_loss /= 2.0
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def gan_train_step(noisy_wave, original_wave, train_generator, model):
    if train_generator:  # Generator training phase
        with tf.GradientTape() as tape:
            generated_wave, fake_output = model(noisy_wave, train_generator, training=True)

            gen_loss = generator_loss(fake_output)

        gradients = tape.gradient(gen_loss, model.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(gen_loss)
    else:  # Discriminator training phase
        with tf.GradientTape() as tape:
            real_output = model(original_wave, train_generator, training=True)
            generated_wave, fake_output = model(noisy_wave, not train_generator, training=True)

            disc_loss = discriminator_loss(real_output, fake_output)

        gradients = tape.gradient(disc_loss, model.trainable_variables)
        disc_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(disc_loss)

    loss = loss_object(original_wave, generated_wave)
    reference_loss(loss)  # loss of generator


def gan_learning(train_dataset, test_dataset, model, real):
    global wait_threshold
    temp_loss = .0
    bad_generator = False  # For changing training phase(Discriminator training first)
    wait = wait_threshold
    res = None

    for epoch in range(EPOCHS):
        wrong_discriminator = False  # Do model need to single training?
        start = time.time()

        if res is None:
            if bad_generator and (temp_loss < reference_loss.result()):
                wait -= 1  # previous loss < train_loss
                if wait == 0:
                    wrong_discriminator = True
                    print("Wrong!")
                    wait_threshold -= 1
                    wait = wait_threshold  # reset wait value
                    bad_generator = not bad_generator
                    # Changing loss only happens when training generator. It will stop generator training phase.
            if train_loss.result() < phase_shift_loss:
                bad_generator = not bad_generator  # Changing phase
                wait = wait_threshold

            temp_loss = reference_loss.result()
            train_loss.reset_state()
            test_loss.reset_state()
            reference_loss.reset_state()
            model.generator.trainable = bad_generator
            model.discriminator.trainable = not bad_generator

            i = 0
            for x_data, y_data in train_dataset:
                if (i != 0) and (i % (data.frame_num // data.truncate) == 0):
                    reset_gru(model)
                if wrong_discriminator:
                    single_train_step(x_data, y_data, real)
                else:
                    gan_train_step(x_data, y_data, bad_generator, model)
                i += 1

            i = 0
            for x_data, y_data in test_dataset:
                if i % (data.frame_num // data.truncate) == 0:
                    reset_gru(model)

                res_temp = single_test_step(x_data, y_data, real)

                if (epoch < EPOCHS - 20) and ((test_loss.result() < early_stop) or (epoch == EPOCHS - 1)):
                    res_temp = np.expand_dims(res_temp, axis=0)
                    if i == 0:
                        res = res_temp
                    else:
                        res = np.concatenate((res, res_temp), axis=0)

                i += 1

            reset_gru(model)

            print(f'Epoch {epoch + 1} (', end='')
            if wrong_discriminator:
                print("Single train phase)")
            elif bad_generator:
                print("Generator train phase)")
            else:
                print("Discriminator train phase)")
            print(
                f'Real: GAN Train Loss(Cross Entropy): {train_loss.result()}, '
                f'GAN Train Error: {reference_loss.result()}, '
                f'Test Error: {test_loss.result()}, '
                f'Time: {time.time() - start} sec'
            )

        else:
            print("Early stop!")
            print("Save numpy files...")
            if real:
                np.save(npy_path + '\\result_real_backup.npy', res)
            else:
                np.save(npy_path + '\\result_imag_backup.npy', res)
            break


print("Start training real part.")
gan_learning(train_real_dataset, test_real_dataset, _model_real, True)
wait_threshold = 15
print("Start training imaginary part")
gan_learning(train_imag_dataset, test_imag_dataset, _model_imag, False)

evaluate.backup_snr_test(npy_path, y_data_real_test, y_data_imag_test)

os.makedirs(save_path_base, exist_ok=True)

evaluate.backup_test(npy_path, save_time, save_scale, len(noise_list), 0, N_FFT, WIN_SIZE, False)
_model_real.save_weights(save_path_real)
_model_imag.save_weights(save_path_imag)
