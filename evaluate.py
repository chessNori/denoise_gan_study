import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import librosa
import soundfile as sf
import os


def print_loss(table, table_name, label_name):
    plt.plot(table)
    plt.title(table_name)
    plt.xlabel('Epoch')
    plt.ylabel(label_name)
    plt.show()


def save_raw(path_time, wave_real, wave_imag, scale, file_name, number_noise, batch, noise_number, n_fft, win_size):
    # (number_batch, batch_size, truncate, N // 2) -> wav file
    length = wave_real.shape[0]//number_noise  # number_frame = length * truncate
    temp = wave_real + (wave_imag * 1j)
    temp = temp[length*noise_number:length*(noise_number+1), batch, :, :]
    temp = np.reshape(temp, (-1, n_fft//2))
    temp = np.transpose(temp, (1, 0))  # (spectrum, time)
    # print_spectrogram(temp)
    i_temp = temp[1:-1, :][::-1, :]
    i_temp = np.conjugate(i_temp)
    temp = np.concatenate((np.zeros((1, temp.shape[1])), temp, i_temp), axis=0)
    res = librosa.istft(temp, n_fft=n_fft, hop_length=win_size//2,
                        win_length=win_size, window='cosine', center=False).real
    path = '..\\results\\wav_file\\spec_only_g\\' + path_time
    os.makedirs(path, exist_ok=True)
    sf.write(path + '\\test_file_' + file_name + '.wav', res * scale, 16000)  # We use 16k sampling datasets


def backup_test(load_path, path_time, scale, number_noise, test_number, n_fft, win_size, end):
    wave_real = np.load(load_path + '\\result_real_backup.npy')
    wave_imag = np.load(load_path + '\\result_imag_backup.npy')
    print('Data shape:', wave_real.shape)

    for i in range(number_noise):
        save_raw(path_time, wave_real, wave_imag, scale, str(test_number) + 'with_noise' + str(i),
                 number_noise, test_number, i, n_fft, win_size)
    if end:
        exit()


def snr(original_real, original_imag, denoise_real, denoise_imag):
    sum_original = original_real + original_imag * 1j
    sum_noise = sum_original - (denoise_real + denoise_imag * 1j)
    sum_original = np.power(np.abs(sum_original), 2)
    sum_noise = np.power(np.abs(sum_noise), 2)
    sum_original = np.sum(sum_original)
    sum_noise = np.sum(sum_noise)
    res = np.log10(sum_original/sum_noise) * 10

    return res


def backup_snr_test(path, original_signal_real, original_signal_imag):
    wave_real = np.load(path + '\\result_real_backup.npy')
    wave_imag = np.load(path + '\\result_imag_backup.npy')

    snr_value = snr(original_signal_real, original_signal_imag, wave_real, wave_imag)
    print('Result SNR dB:', snr_value, 'dB')


def print_spectrogram(spec):
    target = np.abs(spec)
    target = np.log10(target)
    plt.imshow(target, cmap=cm.hot)
    plt.gca().invert_yaxis()
    plt.show()
