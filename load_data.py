import numpy as np
import librosa
import os
from glob import glob


class Data:
    def __init__(self, number_file: int, win_size: int,
                 min_sample=250000, frame_num=1000):
        # min_sample: For LibriSpeech, 250KB equals about 15 seconds of file.
        self.path = '..\\dataset\\LibriSpeech\\train-clean-100\\'  # Speaker\Chapter\Segment
        self.number_file = number_file  # How many files are you going to use?
        self.sr = 16000  # We use 16k sampling datasets
        self.win_size = win_size  # Window size / Sampling rate = frame size(sec)
        self.frame_num = frame_num  # How many frames will the dataset consist of?
        self.padding = win_size * (frame_num // 2)  # Output results file size for hearing test

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
        self.y_data = None  # Dynamic for noise addition

    def load_wave(self, file_number):  # Load LibriSpeech dataset
        print("Loading file_" + str(file_number) + ": ", self.file_name[file_number])
        wave, sr = librosa.load(self.file_name[file_number], sr=self.sr)
        if wave.shape[0] >= self.padding:
            wave = wave[:self.padding]
            print("The file size is bigger than padding size")  # The file is big enough
        else:
            wave = np.concatenate((wave, np.zeros(self.padding - wave.shape[0])), axis=0)

        wave = np.expand_dims(wave, axis=0)
        return wave

    def load_data(self, noise=None, noise_snr=5):  # return wave dataset
        if self.y_data is None:  # Dynamic for noise addition
            data = self.load_wave(0)

            for i in range(1, self.number_file):
                temp = self.load_wave(i)
                data = np.concatenate((data, temp), axis=0)

            self.y_data = np.copy(data)

        res = np.copy(self.y_data)

        if noise is not None:
            for i in range(self.number_file):
                scale = adjust_snr(res[i], noise[i], noise_snr)
                res[i] += noise[i] * scale

        return res

    def make_noise(self, noise_name: str):  # Load Demand datasets
        res = None
        for i in range(16):  # There are 16 files in a folder
            if i < 9:
                path = '..\\dataset\\demand\\' + noise_name + '\\ch0' + str(i+1) + '.wav'
            else:
                path = '..\\dataset\\demand\\' + noise_name + '\\ch' + str(i+1) + '.wav'

            print('Loading noise file: ' + path)
            noise, sr = librosa.load(path, sr=self.sr)  # Every Demand dataset files are big enough
            noise = np.expand_dims(noise[:self.padding], axis=0)

            if i == 0:
                res = noise
            else:
                res = np.concatenate((res, noise), axis=0)

        res_temp = np.copy(res)
        for _ in range(1, self.number_file // 16):  # For matching shape with y_data
            res = np.concatenate((res, res_temp), axis=0)

        return res


def adjust_snr(target, noise, db):  # Because of abs, it didn't return good scale value. We need bug fix
    sum_original = np.power(np.abs(target), 2)
    sum_noise = np.power(np.abs(noise), 2)
    sum_original = np.sum(sum_original)
    sum_noise = np.sum(sum_noise)
    sum_original = np.log10(sum_original)
    sum_noise = np.log10(sum_noise)
    scale = np.power(10, (sum_original-sum_noise)/2-(db/20))
    # SNR = 10 * log(power of signal(S)/power of noise(N))
    # SNR = 10 * (log(S) - log(N) - 2 log(noise scale))
    # log(noise scale) = (log(S) - log(N))/2 - SNR/20

    return scale
