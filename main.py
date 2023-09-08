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
lr = 2e-4
LAMBDA = 10
pre_train_stop = 0.007
EPOCHS = 30
N_FFT = 160
WIN_SIZE = 160  # 10ms
SNR = 10
test_pick = [False, False, False, True]  # [Make result wav file, Calculate SNR, Make x, y wav file, Load pre_weights]
save_scale = 2
save_time = dt.datetime.now()
save_time = save_time.strftime("%Y%m%d%H%M")
save_time = "202309051656"
npy_path = '..\\results\\npy_backup\\spec_only_g\\' + save_time
save_path_base = '..\\results\\saved_model\\spec_only_g\\' + save_time
save_path_g = save_path_base + '\\_model_g'
save_path_d = save_path_base + '\\_model_d'
print("Folder name: ", save_time)

noise_list = ['DKITCHEN','NFIELD']  # Noise folder Names

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


def regularization(x, reg_val):
    reg_temp = max(tf.experimental.numpy.max(x), abs(tf.experimental.numpy.min(x)))
    if reg_temp > reg_val:
        return reg_temp
    else:
        return reg_val


reg = 0.
reg = regularization(x_data, reg)
reg = regularization(x_data_test, reg)

x_data = tf.experimental.numpy.divide(x_data, reg)
x_data_test = tf.experimental.numpy.divide(x_data_test, reg)

print("Data Loading is Done! (", time.time() - _start, ")")
print('Shape of train data(x,y):', x_data.shape, y_data.shape)
print('Shape of test data(x,y):', x_data_test.shape, y_data_test.shape)

_train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(192).batch(batch_size)
_test_dataset = tf.data.Dataset.from_tensor_slices((x_data_test, y_data_test)).batch(batch_size)

save_path_pre_g = '..\\results\\saved_model\\spec_only_g\\pre_training\\_model_pre_g'

generator_g = models.WaveGenerator(N_FFT, WIN_SIZE)  # Denoise model
generator_f = models.WaveGenerator(N_FFT, WIN_SIZE)  # Add noise model

if test_pick[3]:
    # generator_g.load_weights(save_path_pre_g)
    generator_g.load_weights(save_path_g)

single_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# single_loss_object = tf.keras.losses.MeanSquaredError()
single_loss_object = tf.keras.losses.MeanAbsoluteError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def single_train_step(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = generator_g(noisy_wave, training=True)
        loss = single_loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, generator_g.trainable_variables)
    single_optimizer.apply_gradients(zip(gradients, generator_g.trainable_variables))

    train_loss(loss)


@tf.function
def single_test_step(noisy_wave, original_wave):
    denoise_wave = generator_g(noisy_wave, training=False)
    loss = single_loss_object(original_wave, denoise_wave)
    test_loss(loss)

    return denoise_wave


def pre_training(train_dataset, test_dataset):
    for epoch in range(100):
        start = time.time()
        train_loss.reset_state()
        test_loss.reset_state()

        for x, y in train_dataset:
            single_train_step(x, y)

        test_snr = 0.
        n = 0
        for x, y in test_dataset:
            temp = single_test_step(x, y)
            test_snr += evaluate.snr(y, temp)
            n += 1
        test_snr /= n

        print(
            f'Epoch {epoch + 1}, '
            f'Train Loss: {train_loss.result()}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test SNR: {test_snr:.3f}dB, '
            f'Time: {time.time() - start} sec'
        )

        if test_loss.result() < pre_train_stop:
            print("Early stop!")
            break


if not test_pick[3]:
    print("Start pre-training.")
    pre_training(_train_dataset, _test_dataset)
#     _model.generator.save_weights(save_path_pre_g)

discriminator_x = models.DiscriminatorModel()  # Is it denoise wave?
discriminator_y = models.DiscriminatorModel()  # Is it noisy wave?

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

gen_g_train_loss = tf.keras.metrics.Mean(name='gen_g_loss')
gen_f_train_loss = tf.keras.metrics.Mean(name='gen_f_loss')
disc_x_train_loss = tf.keras.metrics.Mean(name='disc_x_loss')
disc_y_train_loss = tf.keras.metrics.Mean(name='disc_y_loss')


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # un_noisy percent
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    total_loss *= 0.5
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def calc_cycle_loss(real_output, cycled_output):
    loss1 = tf.reduce_mean(tf.abs(real_output - cycled_output))

    return LAMBDA * loss1


def identity_loss(real_output, same_output):
    loss = tf.reduce_mean(tf.abs(real_output - same_output))
    return LAMBDA * loss * 0.5


generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


@tf.function
def gan_train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    gen_g_train_loss(total_gen_g_loss)
    gen_f_train_loss(total_gen_f_loss)
    disc_x_train_loss(disc_x_loss)
    disc_y_train_loss(disc_y_loss)


def gan_learning(train_dataset, test_dataset):
    table_temp_train = []
    table_temp_test = []
    table_temp_snr = []

    for epoch in range(EPOCHS):
        start = time.time()
        time_indicator = dt.datetime.now()
        time_indicator = start.strftime("%d%H%M")
        gen_g_train_loss.reset_state()
        gen_f_train_loss.reset_state()
        disc_x_train_loss.reset_state()
        disc_y_train_loss.reset_state()
        test_loss.reset_state()

        for x_wave, y_wave in train_dataset:
            gan_train_step(x_wave, y_wave)

        test_snr = 0.
        n = 0
        for x_wave, y_wave in test_dataset:
            temp = single_test_step(x_wave, y_wave)
            test_snr += evaluate.snr(y_wave, temp)
            n += 1
        test_snr /= n

        print(
            f'Epoch {epoch + 1}({time_indicator}):\n'
            f'Generator G Loss: {gen_g_train_loss.result()}, '
            f'Generator F Loss: {gen_f_train_loss.result()}, '
            f'Discriminator X Loss: {disc_x_train_loss.result()}, '
            f'Discriminator Y Loss: {disc_y_train_loss.result()}\n'
            f'Test Loss: {test_loss.result()}, '
            f'Test SNR: {test_snr:.3f}dB, '
            f'Time: {time.time() - start} sec'
        )
        table_temp_train.append(gen_g_train_loss.result())
        table_temp_test.append(test_loss.result())
        table_temp_snr.append(test_snr)

        if test_snr > 5.0:
            break

    evaluate.print_loss(table_temp_train, table_temp_test, 'MSE Graph')
    evaluate.print_snr_graph(table_temp_snr)


print("Start training.")
gan_learning(_train_dataset, _test_dataset)


def test(test_dataset):
    i = 0
    res = None
    for x_wave, y_wave in test_dataset:
        res_temp = single_test_step(x_wave, y_wave)
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

generator_g.save_weights(save_path_g)
