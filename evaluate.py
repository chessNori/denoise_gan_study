import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf


def print_loss(table, table_name, label_name):
    plt.plot(table)
    plt.title(table_name)
    plt.xlabel('Epoch')
    plt.ylabel(label_name)
    plt.show()


def save_raw(wave, phase, scale, file_name, number_batch, file_number=0, n_fft=128):  # for test
    # temp = np.power(10, wave)
    # temp -= 1e-5
    length = wave.shape[0]//number_batch  # file_length = length * truncate
    temp = wave * phase
    temp = temp[file_number*length:(file_number+1)*length, 0, :, :]
    temp = np.reshape(temp, (-1, n_fft//2+1))
    temp = np.transpose(temp, (1, 0))
    i_temp = np.conjugate(temp[1:-1, :][::-1, :])
    temp = np.concatenate((temp, i_temp), axis=0)
    res = librosa.istft(temp, n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft, window='hann').real

    # file = open('test_file_' + file_name + '.raw', 'wb')
    # for i in range(res.shape[0]):
    #     file.write((int(res[i] * scale)).to_bytes(2, byteorder='little', signed=True))
    #     if i % 2000 == 0:
    #         print('@', end='')
    # print('\nDone!')
    # file.close()
    sf.write('test_file_' + file_name + '.wav', res * scale, 16000)


def backup_test(number_batch, test_number, n_fft=128):
    wave = np.load('result_backup.npy')
    phase = np.load('phase_backup.npy')
    print('Data shape:', wave.shape)

    save_raw(wave, phase, 30, 'test2', number_batch, file_number=test_number)
    exit()


def snr(original_signal, denoise_signal):
    sum_original = np.power(original_signal, 2)
    sum_noise = np.power(original_signal-denoise_signal, 2)
    sum_original = np.sum(sum_original)
    sum_noise = np.sum(sum_noise)
    res = np.log10(sum_original/sum_noise) * 10

    return res


def backup_snr_test(original_signal):
    wave = np.load('result_backup.npy')

    snr_value = snr(original_signal, wave)
    print('SNR dB:', snr_value, 'dB')
    exit()
