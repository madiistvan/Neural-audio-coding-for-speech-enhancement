from torch.utils.data import Dataset
from os import listdir
from torch import load

class EncodedDataset(Dataset):
    def __init__(self, data_dir, noise_dir):
        self.data_dir = data_dir
        self.noise_dir = noise_dir
        self.audio_files = [f for f in listdir(data_dir) if f.endswith(".pt")]
        self.noise_files = [f for f in listdir(noise_dir) if f.endswith(".pt")]        
        print(f'Noise len: {len(self.noise_files)}')
        print(f'Audio len: {len(self.audio_files)}')

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = f'{self.data_dir}/{self.audio_files[idx]}'
        waveform = load(audio_path)
        noise_path = f'{self.noise_dir}/{self.noise_files[idx]}'
        noisy = load(noise_path)
        return noisy, waveform
