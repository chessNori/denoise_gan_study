import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os


def print_loss(table_train, table_test, table_name):
    plt.plot(table_train, 'g', label="Train")
    plt.plot(table_test, 'b', label="Test")
    plt.title(table_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def save_raw(path_time, wave, scale, file_name, batch_number, noise_number, batch_size=None):
    # (number_batch, batch_size, truncate, N // 2) -> wav file
    temp = np.copy(wave)
    if batch_size is not None:
        temp = np.reshape(temp, (wave.shape[0]//batch_size, batch_size, -1))
    temp = temp[noise_number, batch_number, :]
    path = '..\\results\\wav_file\\spec_only_g\\' + path_time
    os.makedirs(path, exist_ok=True)
    sf.write(path + '\\test_file_' + file_name + '.wav', temp * scale, 16000)  # We use 16k sampling datasets


def backup_test(load_path, path_time, scale, number_noise, test_number, end):
    wave = np.load(load_path + '\\result_backup.npy')
    print('Data shape:', wave.shape)

    for i in range(number_noise):
        save_raw(path_time, wave, scale, str(test_number) + 'with_noise' + str(i), test_number, i)
    if end:
        exit()


def snr(original, denoise):
    sum_noise = original - denoise
    sum_original = np.power(np.abs(original), 2)
    sum_noise = np.power(np.abs(sum_noise), 2)
    sum_original = np.sum(sum_original)
    sum_noise = np.sum(sum_noise)
    res = np.log10(sum_original/sum_noise) * 10

    return res


def backup_snr_test(path, original_signal):
    wave = np.load(path + '\\result_backup.npy')
    if wave.shape[0] != original_signal.shape[0]:
        wave = np.reshape(wave, (original_signal.shape[0], original_signal.shape[1]))

    snr_value = snr(original_signal, wave)
    print('Result SNR dB:', snr_value, 'dB')
