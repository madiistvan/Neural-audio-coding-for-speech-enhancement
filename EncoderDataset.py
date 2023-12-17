from torch.utils.data import Dataset, DataLoader
import os
import torch
import soundfile as sf
import torchaudio.functional as F
import fnmatch
import numpy as np
import math
import yaml
import random
import torch.fft as fft
import torchaudio



def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find speech files recursively.
        Args:
            root_dir (str): Root root_dir to find.
            query (str): Query to find.
            include_root_dir (bool): If False, root_dir name is not included.
        Returns:
            list: List of found filenames.
    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def load_config(checkpoint, config_name='config.yml'):
        dirname = os.path.dirname(checkpoint)
        config_path = os.path.join(dirname, config_name)
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        return config

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    #print(f"Using {device} device")
    return device

class EncoderDataset(Dataset):
    def __init__(
        self,
        noise_files,
        speech_files,
        noise_count=1, # number of noise files to be mixed with one speech file
        snr_low=2,
        snr_high=18,
        query="*.wav",
        load_fn=sf.read,
        return_utt_id=False,
        trimmed_length=3*48000,
        subset_num=-1,
        fourier = False,
        pitch = False
    ):
        self.return_utt_id = return_utt_id
        self.load_fn = load_fn
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.noise_count = noise_count
        self.trimmed_length = trimmed_length
        self.subset_num = subset_num
        self.fourier= fourier
        self.pitch = pitch
        #self.device = get_device()
        self.speech_filenames = self._load_list(speech_files, query)
        self.noise_filenames = self._load_list(noise_files, query)
        self.utt_ids, self.noise_indices = self._load_ids(self.speech_filenames, self.noise_filenames, self.noise_count)

    def __getitem__(self, idx):
        mixed, speech = self._data(idx)
        if self.return_utt_id:
            utt_id = self.utt_ids[idx]
            items = utt_id, mixed, speech
        else:
            items = mixed, speech
        return items

    def __len__(self):
        return len(self.utt_ids)
    
    
    def _read_list(self, listfile):
        filenames = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                if len(line):
                    filenames.append(line)
        return filenames
    

    def _load_list(self, files, query):
        if isinstance(files, list):
            filenames = files
        else:
            if os.path.isdir(files):
                filenames = sorted(find_files(files, query))
            elif os.path.isfile(files):
                filenames = sorted(self._read_list(files))
            else:
                raise ValueError(f"{files} is not a list / existing folder or file!")
            
        if self.subset_num > 0:
           filenames = filenames[:self.subset_num]
        assert len(filenames) != 0, f"File list in empty!"
        return filenames
    
    
    def _load_ids(self, speech_filenames, noise_filenames, noise_count=1):
             
        def splittext(filename):
            return os.path.splitext(os.path.basename(filename))[0]

        mixed_ids = []
        noise_indices = []
        for speech_filename in speech_filenames:
            speech_id= splittext(speech_filename)
            tmp_noise_indices = []
            for _ in range(noise_count):
                noise_index=np.random.choice(len(noise_filenames))
                mixed_id = speech_id + "___" + splittext(noise_filenames[noise_index])
                mixed_ids.append(mixed_id)
                tmp_noise_indices.append(noise_index)
            noise_indices.append(tmp_noise_indices)
        return mixed_ids, noise_indices
    

    def _data(self, idx):
        speech_index=math.floor(idx/self.noise_count)
        speech=self.speech_filenames[speech_index]
        noise_index=self.noise_indices[speech_index][idx%self.noise_count]
        noise=self.noise_filenames[noise_index]
        return self._load_data(speech, noise, self.load_fn)
    

    def _load_data(self, speech, noise, load_fn):
        speech_data = torchaudio.load(speech, backend="soundfile")[0].to(get_device())
        noise_data = torchaudio.load(noise, backend="soundfile")[0].to(get_device())

        """ if load_fn == sf.read:
            speech_data, _ = load_fn(speech, always_2d=True) # (T, C)
            noise_data, _ = load_fn(noise, always_2d=True) # (T, C)
        else:
            speech_data = load_fn(speech)
            noise_data = load_fn(noise) """
        return self._mix_data(speech_data, noise_data)
    
    def _bandpass_filter(self, noisy_waveform, waveform, sample_rate):
        noisy_spectrum = torch.fft.fft(noisy_waveform[0])
        waveform_spectrum = torch.fft.fft(waveform[0])

        low_cutoff = int(100 / (sample_rate / noisy_waveform.size(1)))
        high_cutoff = int(17000 / (sample_rate / noisy_waveform.size(1)))

        noisy_spectrum[:low_cutoff] = 0
        noisy_spectrum[high_cutoff:] = 0

        waveform_spectrum[:low_cutoff] = 0
        waveform_spectrum[high_cutoff:] = 0
        
        # Convert the filtered spectra back to waveforms


        filtered_noisy_waveform = torch.real(fft.ifft(noisy_spectrum))
        filtered_waveform = torch.real(fft.ifft(waveform_spectrum))
        
        return filtered_noisy_waveform, filtered_waveform
    
    def _change_pitch(self, waveform):
        range = np.arange(-4, +4, 1)
        range = range.astype(np.double)
        n_steps = np.random.choice(range)
        shifted = torchaudio.functional.pitch_shift(waveform, 48000, n_steps)
        return shifted
    
    def _clear_nans(self,waveform):
        waveform[waveform != waveform] = random.uniform(-1e-10, 1e-10)
        return waveform

    def _mix_data(self, speech_data, noise_data):

        #TRIMMING
        
        noise_data = noise_data[:, :self.trimmed_length]
        speech_data = speech_data[:, :self.trimmed_length]
        
        noise_length = noise_data.shape[1]
        speech_length = speech_data.shape[1]
        
        #FILLING
        """ if noise_length < self.trimmed_length:
            filling_length = self.trimmed_length - noise_length     
            zeros = np.zeros((filling_length, 1))
            noise_data = np.concatenate((noise_data, zeros), 0)

        if speech_length < self.trimmed_length:
            filling_length = self.trimmed_length - speech_length     
            zeros = np.zeros((filling_length, 1))
            speech_data = np.concatenate((speech_data, zeros), 0) """
        
        if noise_length < self.trimmed_length:
            
            filling_length = self.trimmed_length - noise_length  
            print(filling_length)
            zeros = torch.zeros((1, filling_length)).to(get_device())
            noise_data = torch.cat((noise_data, zeros), 1)
            print(noise_data)

        if speech_length < self.trimmed_length:
            filling_length = self.trimmed_length - speech_length     
            zeros = torch.zeros((1, filling_length)).to(get_device())
            speech_data = torch.cat((speech_data, zeros), 1)


        """ #TRANSPOSE
        speech_data = np.transpose(speech_data, (1,0)) # (C, T)
        noise_data = np.transpose(noise_data, (1,0)) # (C, T) """

        
        #MIXING
        #speech_data = torch.from_numpy(speech_data)

        #CHANGING PITCH
        if self.pitch:
            speech_data = self._change_pitch(speech_data)

        #noise_data = torch.from_numpy(noise_data)
        snr = torch.from_numpy(np.array([random.randint(self.snr_low, self.snr_high)])).to(get_device())
        mixed_data = F.add_noise(speech_data, noise_data, snr)

        #HANDLING NANs
        mixed_data = self._clear_nans(mixed_data)
        speech_data = self._clear_nans(speech_data)

        mixed_data = mixed_data.to(get_device())
        speech_data = speech_data.to(get_device())

    
        if self.fourier:
            mixed_data, speech_data = self._bandpass_filter(mixed_data, speech_data, 48000)
            mixed_data = self._clear_nans(mixed_data)
            speech_data = self._clear_nans(speech_data)
        
        
        return mixed_data, speech_data
            
        

