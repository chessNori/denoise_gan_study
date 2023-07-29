import numpy as np
import librosa
import os
from glob import glob


class Data:
    def __init__(self, number_file: int, batch_size: int, n_fft=128,
                 min_sample=250000, frame_num=2000, truncate=100):  # real frame_num = frame_num * 2
        self.path = '..\\dataset\\LibriSpeech\\train-clean-100\\'  # Speaker\Chapter\Segment
        self.number_file = number_file
        self.n_fft = n_fft
        self.frame_num = frame_num * 2
        self.padding = n_fft * frame_num
        self.sr = 16000
        self.batch_size = batch_size
        self.truncate = truncate
        self.phase = None

        speaker_dir = [f.path for f in os.scandir(self.path) if f.is_dir()]

        chapter_dir = []
        for one_path in speaker_dir:
            chapter_dir += [f.path for f in os.scandir(one_path) if f.is_dir()]

        segment_name = []
        for one_path in chapter_dir:
            segment_name += glob(one_path + '\\*.flac')

        delete_file = []
        for one_path in segment_name:
            if os.stat(one_path).st_size < min_sample:
                delete_file.append(one_path)

        for one_path in delete_file:
            segment_name.remove(one_path)  # Delete too small segment(ex: 250000Bytes -> about 15sec at flac)

        self.file_name = segment_name[:self.number_file]
        self.regularization = 0
        self.y_data = None

    def rnn_shape(self, wave):  # (frame_num // truncate, 1, truncate, N // 2 + 1)
        spectrum = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.n_fft // 2, win_length=self.n_fft,
                                window='hann')[:, :self.frame_num]  # (n_fft/2 + 1, frame_num * 2 + 1)
        spectrum = np.transpose(spectrum, (1, 0))
        spectrum = np.reshape(spectrum, (self.frame_num // self.truncate, 1, self.truncate, self.n_fft // 2 + 1))

        return spectrum

    def rnn_spectrogram(self, file_number):
        print("Loading file_" + str(file_number) + ": ", self.file_name[file_number])
        wave, sr = librosa.load(self.file_name[file_number], sr=self.sr)
        if wave.shape[0] >= self.padding:
            wave = wave[:self.padding]
            print("The file size is bigger than padding size")
        else:
            wave = np.concatenate((wave, np.zeros(self.padding - wave.shape[0])), axis=0)

        spectrum = self.rnn_shape(wave)

        return spectrum

    def load_data(self, noise=None):
        if self.y_data is None:
            data = self.rnn_spectrogram(0)

            for i in range(1, self.number_file):
                temp = self.rnn_spectrogram(i)
                data = np.concatenate((data, temp), axis=1)  # spectrogram

            res = data[:, :self.batch_size]
            for i in range(1, data.shape[1] // self.batch_size):
                res_temp = data[:, self.batch_size*i:self.batch_size*(i+1)]
                res = np.concatenate((res, res_temp), axis=0)

            self.y_data = res

        res = self.y_data

        if noise is not None:
            res += noise
        res = self.spectral_magnitude(res, noise)  # spectral magnitude(log10)
        res = res.astype(np.float32)

        val_max = max(np.abs(np.max(res)), np.abs(np.min(res)))
        if val_max > self.regularization:
            self.regularization = val_max

        return res

    def make_noise(self, noise_name: str):
        res_temp = None
        for i in range(self.batch_size):
            if i < 9:
                path = '..\\dataset\\demand\\' + noise_name + '\\ch0' + str(i+1) + '.wav'
            else:
                path = '..\\dataset\\demand\\' + noise_name + '\\ch' + str(i+1) + '.wav'

            print('Loading noise file: ' + path)
            noise, sr = librosa.load(path, sr=self.sr)[:self.padding]
            noise = self.rnn_shape(noise)

            if i == 0:
                res_temp = noise
            else:
                res_temp = np.concatenate((res_temp, noise), axis=1)

        res = res_temp
        for _ in range(1, self.number_file // self.batch_size):
            res = np.concatenate((res, res_temp), axis=0)

        return res

    def spectral_magnitude(self, spectrum, noise=None):
        magnitude, phase = librosa.magphase(spectrum)
        if noise is None:
            self.phase = phase
        # white_for_log = 1e-5
        # magnitude = np.log10(magnitude + white_for_log)
        return magnitude
