import numpy as np
import time
import load_data
import evaluate
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Flatten, Concatenate, Conv2D, RepeatVector, TimeDistributed, MaxPool2D
from tensorflow.keras import Model

noise_list = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NRIVER', 'OOFFICE']  # Folder Names
# evaluate.backup_test(len(noise_list), 2)

batch_size = 16
number_batch = 8
gen_lr = 1e-5
disc_lr = 3e-5
single_lr = 1e-4
EPOCHS = 120
loss_threshold = 1e-2
wait_threshold = 3
save_path = '..\\saved_model\\2023_07_10_GAN_denoise_model'

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

evaluate.backup_snr_test(y_data_test)

x_data = x_data[x_data.shape[0]//number_batch:]
y_data = y_data[y_data.shape[0]//number_batch:]

# evaluate.save_raw(x_data_test, data.phase[:x_data_test.shape[0]], 2000000, 'x', len(noise_list))
# evaluate.save_raw(y_data_test, data.phase[:y_data_test.shape[0]], 2000000, 'y', len(noise_list))

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


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(8, 9)
        self.conv2 = Conv2D(16, 9)
        self.max_pool = MaxPool2D(pool_size=(9, 9))
        self.conv3 = Conv2D(1, 5)
        self.flatten = Flatten()
        self.d = Dense(20)
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
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = GeneratorModel()
        self.discriminator = Discriminator()

    def call(self, inputs, fake):
        if fake:
            denoise = self.generator(inputs)
            x = self.discriminator(denoise)
            return denoise, x
        else:
            x = self.discriminator(inputs)
            return x


_model = GAN()

gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr)
single_optimizer = tf.keras.optimizers.Adam(learning_rate=single_lr)


loss_object = tf.keras.losses.MeanSquaredError()
error_name = 'MSE'
# loss_object = tf.keras.losses.MeanAbsoluteError()
# error_name = 'MAE'
cross_entropy = tf.keras.losses.BinaryCrossentropy()

single_train_loss = tf.keras.metrics.Mean(name='train_loss')
gan_train_loss = tf.keras.metrics.Mean(name='gan_train_loss')
single_test_loss = tf.keras.metrics.Mean(name='test_loss')
raw_e = tf.keras.metrics.Mean(name='raw_mae')


@tf.function
def single_train_step(noisy_wave, original_wave):
    with tf.GradientTape() as tape:
        denoise_wave = _model.generator(noisy_wave, training=True)
        loss = loss_object(original_wave, denoise_wave)
    gradients = tape.gradient(loss, _model.generator.trainable_variables)
    single_optimizer.apply_gradients(zip(gradients, _model.generator.trainable_variables))

    single_train_loss(loss)


@tf.function
def single_test_step(noisy_wave, original_wave):
    denoise_wave = _model.generator(noisy_wave, training=False)
    loss = loss_object(original_wave, denoise_wave)

    single_test_loss(loss)

    return denoise_wave


for epoch in range(EPOCHS):  # Generator pre_training
    start = time.time()
    single_train_loss.reset_state()
    single_test_loss.reset_state()

    i = 0
    for x_wave, y_wave in train_dataset:
        if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
            _model.generator.e_gru.reset_states()
            _model.generator.d_gru.reset_states()

        single_train_step(x_wave, y_wave)
        i += 1

    _model.generator.e_gru.reset_states()
    _model.generator.d_gru.reset_states()
    i = 0
    for x_wave, y_wave in test_dataset:
        if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
            _model.generator.e_gru.reset_states()
            _model.generator.d_gru.reset_states()

        single_test_step(x_wave, y_wave)
        i += 1

    print(
        f'Pre Training Epoch {epoch + 1}, '
        f'Train Loss: {single_train_loss.result()}, '
        f'Test Loss: {single_test_loss.result()}, '
        f'Time: {time.time() - start} sec'
    )

    _model.generator.e_gru.reset_states()
    _model.generator.d_gru.reset_states()

    if single_test_loss.result() < 1e-6:
        break  # End of pre_training


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # un_noisy percent
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    total_loss /= 2.0
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# @tf.function
def gan_train_step(noisy_wave, original_wave, train_generator):
    if train_generator:
        with tf.GradientTape() as tape:
            generated_wave, fake_output = _model(noisy_wave, train_generator, training=True)

            gen_loss = generator_loss(fake_output)

        gradients = tape.gradient(gen_loss, _model.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients, _model.trainable_variables))
        gan_train_loss(gen_loss)
    else:
        with tf.GradientTape() as tape:
            real_output = _model(original_wave, train_generator, training=True)
            generated_wave, fake_output = _model(noisy_wave, not train_generator, training=True)

            disc_loss = discriminator_loss(real_output, fake_output)

        gradients = tape.gradient(disc_loss, _model.trainable_variables)
        disc_optimizer.apply_gradients(zip(gradients, _model.trainable_variables))
        gan_train_loss(disc_loss)

    loss = loss_object(original_wave, generated_wave)
    single_train_loss(loss)


def raw_error(noisy_wave, original_wave):
    res = loss_object(original_wave, noisy_wave)
    raw_e(res)


table_temp1 = []  # Cross entropy results
table_temp2 = []  # MAE results
temp_loss = .0
bad_generator = False  # Discriminator training first
wait = wait_threshold

for epoch in range(EPOCHS):
    wrong_discriminator = False
    res = None
    start = time.time()

    if bad_generator and (temp_loss < single_train_loss.result()) and ((temp_loss * 2) > single_train_loss.result()):
        wait -= 1
        if wait == 0:
            wrong_discriminator = True
            print('Wrong!')
            wait = wait_threshold
            bad_generator = not bad_generator
    elif (gan_train_loss.result() != .0) and (gan_train_loss.result() < loss_threshold):
        bad_generator = not bad_generator
        wait = wait_threshold

    temp_loss = single_train_loss.result()
    gan_train_loss.reset_states()
    single_train_loss.reset_states()
    single_test_loss.reset_state()
    _model.generator.trainable = bad_generator
    _model.discriminator.trainable = not bad_generator

    i = 0
    for x_wave, y_wave in train_dataset:
        if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
            _model.generator.e_gru.reset_states()
            _model.generator.d_gru.reset_states()
        i += 1
        if wrong_discriminator:
            single_train_step(x_wave, y_wave)
            gan_train_step(x_wave, y_wave, bad_generator)
        else:
            gan_train_step(x_wave, y_wave, bad_generator)

        if epoch == (EPOCHS - 1):
            raw_error(x_wave, y_wave)

    i = 0
    for x_wave, y_wave in test_dataset:
        if (i != 0) and (i % (data.frame_num // data.truncate * number_batch) == 0):
            _model.generator.e_gru.reset_states()
            _model.generator.d_gru.reset_states()

        temp = single_test_step(x_wave, y_wave)

        if (epoch >= EPOCHS - 20) and (single_test_loss.result() < 2e-6):
            temp = np.expand_dims(temp, axis=0)
            if i == 0:
                res = temp
            else:
                res = np.concatenate((res, temp), axis=0)

        i += 1

    print(
        f'Epoch {epoch + 1}, '
        f'GAN Train Loss(Cross Entropy): {gan_train_loss.result()}, '
        f'GAN Train Error: {single_train_loss.result()}, '
        f'Test Error: {single_test_loss.result()}'
        f'Time: {time.time() - start} sec'
    )

    _model.generator.e_gru.reset_states()
    _model.generator.d_gru.reset_states()
    table_temp1.append(gan_train_loss.result())
    table_temp2.append(single_train_loss.result())

    if res is not None:
        print(f'Reference error(original wave vs noise added wave): {raw_e.result()}')
        print('Model save...')
        _model.generator.save(save_path)
        print("Saving model complete.")
        print('Save raw files...')
        np.save('result_backup.npy', res)
        np.save('phase_backup.npy', data.phase[:res.shape[0]])
        evaluate.save_raw(res, data.phase[:res.shape[0]], 2000000, 'test', len(noise_list))
        break

evaluate.print_loss(table_temp1, 'Cross Entropy(y_real, pred_denoise)', 'Loss')
evaluate.print_loss(table_temp2, error_name+'(real, denoise)', 'Loss')
