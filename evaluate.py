import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy import signal
import soundfile as sf


def print_loss(table, table_name, label_name):
    plt.plot(table)
    plt.title(table_name)
    plt.xlabel('Epoch')
    plt.ylabel(label_name)
    plt.show()


def save_raw(wave_real, wave_imag, scale, file_name, number_batch, file_number=0, n_fft=128, win_size=128):  # for test
    # temp = np.power(10, wave)
    # temp -= 1e-5
    length = wave_real.shape[0]//number_batch  # file_length = length * truncate
    # temp = wave * phase
    temp = wave_real + (wave_imag * 1j)
    temp = temp[file_number*length:(file_number+1)*length, 0, :, :]
    temp = np.reshape(temp, (-1, n_fft//2+1))
    temp = np.transpose(temp, (1, 0))
    i_temp = temp[1:-1, :][::-1, :]
    i_temp = np.conjugate(i_temp)
    temp = np.concatenate((temp, i_temp), axis=0)
    res = librosa.istft(temp, n_fft=n_fft, hop_length=win_size//2, win_length=win_size, window='cosine').real
    sf.write('test_file_' + file_name + '.wav', res * scale, 16000)


def backup_test(number_batch, test_number, n_fft=128):
    wave_real = np.load('result_real_backup.npy')
    wave_imag = np.load('result_imag_backup.npy')
    print('Data shape:', wave_real.shape)

    save_raw(wave_real, wave_imag, 30, 'test2', number_batch, file_number=test_number, n_fft=n_fft)
    exit()


def snr(original_real, original_imag, denoise_real, denoise_imag):
    sum_original = np.power(original_real, 2) + np.power(original_imag, 2)
    sum_noise = np.power(original_real-denoise_real, 2) + np.power(original_imag-denoise_imag, 2)
    sum_original = np.sum(sum_original)
    sum_noise = np.sum(sum_noise)
    res = np.log10(sum_original/sum_noise) * 10

    return res


def backup_snr_test(original_signal_real, original_signal_imag):
    wave_real = np.load('result_real_backup.npy')
    wave_imag = np.load('result_imag_backup.npy')

    snr_value = snr(original_signal_real, original_signal_imag, wave_real, wave_imag)
    print('SNR dB:', snr_value, 'dB')
    exit()


def window_istft(wave, n_fft):
    window = signal.windows.cosine(n_fft)
    window = np.expand_dims(window, axis=-1)
    frames = wave * window
    frames = np.fft.ifft(frames, n_fft, axis=0)
    # frames = np.fft.ifft(wave, n_fft, axis=0)
    # frames *= window

    res = frames[:, 0]
    for i in range(1, wave.shape[-1]):
        res[-(n_fft//2):] += frames[:n_fft//2, i]
        res = np.concatenate((res, frames[n_fft//2:, i]), axis=-1)

    return res


