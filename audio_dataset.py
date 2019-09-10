import torch
from torch.utils.data import Dataset
# from torchvision import transforms
import librosa
from librosa.effects import pitch_shift
from pathlib import Path
import numpy as np
import random

classes = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime",
           "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano",
           "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica",
           "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors",
           "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
           "Writing"]



class AudioFileDataset(Dataset):
    def __init__(self, data_dir, signal_ratio=16000):
        self.sr = signal_ratio
        self.mono = True

        self.file_paths = list(Path(data_dir).glob("**/*.wav"))
        self.minimum_len = 20000
        # for f in self.file_paths:
        #     seq, _ = librosa.load(f, sr=self.sr, mono=self.mono)
        #     if len(seq) < self.minimum_len:
        #         self.minimum_len = len(seq)

        print("num of training files : ", len(self.file_paths))
        print("minimum length of audio : ", self.minimum_len)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        seq, _ = librosa.load(self.file_paths[idx], sr=self.sr, mono=self.mono)

        return seq[:self.minimum_len]
        # return np.float32(x), np.argmax(y)



def clip_silence(audio_dir, files_list):

    """NEED TO ADD PATH -> export PATH=$PATH:/Users/taku-ueki/sox-14.4.2"""
    import os
    fin = open(files_list, "r")
    lines = fin.readlines()
    save_dir = Path("data/processed_data_wav")

    for i, line in enumerate(lines[1:]):
        filename = line.split(",")[0]
        file = str(audio_dir / filename)
        aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"
        aug_audio_file = str(save_dir / filename)
        os.system("sox %s %s %s" % (file, aug_audio_file, aug_cmd))
        # exit()
        if i % 100 == 0:
            print("{} / {}".format(i, len(lines)))

def convert2npy(data_dir, save_dir, sr=32000):
    from pathlib import Path
    data_dir = Path(data_dir)
    files = list(data_dir.glob('**/*.wav'))

    save_dir = Path(save_dir)
    for file in files:
        sig, sr = librosa.load(file, sr=sr, mono=True)
        filename = str(file.name).replace("wav", "npy")
        np.save(save_dir / filename, sig)

if __name__ == '__main__':
    # clip silence
    # clip_silence(Path("/Users/taku-ueki/Documents/data_set/freesound-audio-tagging/audio_train"), "/Users/taku-ueki/Documents/data_set/freesound-audio-tagging/train.csv")

    # convert wav to numpy array
    convert2npy("data/clipped_audio_wav", "data/clipped_audio_npy")
    exit()

    transform = transforms.Compose([Random_slice(), Mixup()])
    dataset = AudioDataset(Path("data/processed_data"), \
        "test_data_prep.txt", transform=[Random_slice(), Mixup(), Random_Noise()])

    print(dataset[0][0].shape)
    print(dataset[1][0].shape)
    print(dataset[2][0].shape)
