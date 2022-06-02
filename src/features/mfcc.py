import librosa


def mfcc(signal, rate, n_mfcc=50):
    mfcc_feat = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=n_mfcc)
    return mfcc_feat.mean(axis=1)
