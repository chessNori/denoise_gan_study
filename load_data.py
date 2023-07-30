import numpy as np
import librosa
import os
from glob import glob


class Data:
    def __init__(self, number_file: int, batch_size: int, n_fft: int, win_size: int,
                 min_sample=250000, frame_num=1500, truncate=100):
        # min_sample: For LibriSpeech, 250KB equals about 15 seconds of file.
        self.path = '..\\dataset\\LibriSpeech\\train-clean-100\\'  # Speaker\Chapter\Segment
        self.number_file = number_file  # How many files are you going to use?
        self.batch_size = batch_size  # Must be less than 16 because of Demand dataset
        self.sr = 16000  # We use 16k sampling datasets
        self.n_fft = n_fft  # FFT N value
        self.win_size = win_size  # Window size / Sampling rate = frame size(sec)
        self.frame_num = frame_num  # How many frames will the dataset consist of?
        self.padding = win_size * (frame_num // 2) + self.win_size * 2  # Output results file size for hearing test
        self.truncate = truncate

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
            segment_name.remove(one_path)  # Delete too small segment

        self.file_name = segment_name[:self.number_file]  # LibriSpeech file path
        self.regularization = 0  # -1.0 ~ +1.0
        self.y_data = None  # Dynamic for noise addition

    def rnn_shape(self, wave):  # (frame_num, N // 2 + 1) -> (frame_num // truncate, 1, truncate, N // 2 + 1)
        spectrum = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.win_size // 2, win_length=self.win_size,
                                window='cosine', center=False)[:, :self.frame_num]  # (n_fft/2 + 1, frame_num-3)
        spectrum = np.transpose(spectrum, (1, 0))
        spectrum = np.reshape(spectrum, (self.frame_num // self.truncate, 1, self.truncate, self.n_fft // 2 + 1))

        return spectrum

    def rnn_spectrogram(self, file_number):  # Load LibriSpeech dataset
        print("Loading file_" + str(file_number) + ": ", self.file_name[file_number])
        wave, sr = librosa.load(self.file_name[file_number], sr=self.sr)
        if wave.shape[0] >= self.padding:
            wave = wave[:self.padding]
            print("The file size is bigger than padding size")  # The file is big enough
        else:
            wave = np.concatenate((wave, np.zeros(self.padding - wave.shape[0])), axis=0)

        spectrum = self.rnn_shape(wave)

        return spectrum

    def load_data(self, noise=None):  # Return real and imaginary spectrum array
        if self.y_data is None:  # Dynamic for noise addition
            data = self.rnn_spectrogram(0)

            for i in range(1, self.number_file):
                temp = self.rnn_spectrogram(i)
                data = np.concatenate((data, temp), axis=1)  # spectrogram

            res = data[:, :self.batch_size]  # Cut to batch size
            for i in range(1, data.shape[1] // self.batch_size):
                res_temp = data[:, self.batch_size*i:self.batch_size*(i+1)]
                res = np.concatenate((res, res_temp), axis=0)

            self.y_data = res

        res = self.y_data

        if noise is not None:
            res += noise  # We need another method at here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        data_real = res.real.astype(np.float32)
        data_imag = res.imag.astype(np.float32)

        real_max = max(np.abs(np.max(data_real)), np.abs(np.min(data_real)))
        imag_max = max(np.abs(np.max(data_imag)), np.abs(np.min(data_imag)))
        val_max = max(real_max, imag_max)  # -1.0 ~ +1.0
        if val_max > self.regularization:
            self.regularization = val_max  # Update regularization

        return data_real, data_imag

    def make_noise(self, noise_name: str):  # Load Demand datasets
        res_temp = None
        for i in range(self.batch_size):
            if i < 9:
                path = '..\\dataset\\demand\\' + noise_name + '\\ch0' + str(i+1) + '.wav'
            else:
                path = '..\\dataset\\demand\\' + noise_name + '\\ch' + str(i+1) + '.wav'
                # There are 16 files in a folder

            print('Loading noise file: ' + path)
            noise, sr = librosa.load(path, sr=self.sr)[:self.padding]  # Every Demand dataset files are big enough
            noise = self.rnn_shape(noise)

            if i == 0:
                res_temp = noise
            else:
                res_temp = np.concatenate((res_temp, noise), axis=1)

        res = res_temp
        for _ in range(1, self.number_file // self.batch_size):  # For matching shape with y_data
            res = np.concatenate((res, res_temp), axis=0)

        return res
