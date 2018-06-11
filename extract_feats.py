import audioread
import librosa
import numpy as np
import os
import shutil


def extract_melspec(in_fp, sr, win_size, hop_size, n_mels):
    sig, sr = librosa.core.load(in_fp, sr=sr)
    feat = librosa.feature.melspectrogram(sig, sr=sr,
                                          n_fft=win_size,
                                          hop_length=hop_size,
                                          n_mels=n_mels).T
    feat = np.log(1+10000*feat)
    return feat


if __name__ == '__main__':

    sr = 16000
    win_sizes = [512,1024,2048,4096,8192,16384]
    hop_size = 512
    n_mels = 128
    diff_order = 0

    if os.path.exists('jy_feat'):
        shutil.rmtree('jy_feat')
    os.mkdir('jy_feat')

    for win_size in win_sizes:
        path = "HL_output/audio/"
        save_path = 'jy_feat/out'+str(win_size)+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for files in os.listdir(path):
            f = os.path.splitext(files)[0]
            in_fp = path + files
            try:
                feat = extract_melspec(in_fp, sr, win_size, hop_size, n_mels)
                np.save(save_path+f+'.npy',feat)
            except (EOFError, IOError, OSError, audioread.NoBackendError):
                continue



