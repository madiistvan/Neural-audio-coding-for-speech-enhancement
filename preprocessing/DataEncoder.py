from models.autoencoder.AudioDec import Encoder
import soundfile as sf
from torch.utils.data import DataLoader
import os
import torch
from preprocessing.EncoderDataset import EncoderDataset
from AudioDec.models.autoencoder.AudioDec import Generator
class DataEncoder:
    def __init__(
            self,
            speech_files,
            noise_files,
            encoded_speech_files,
            encoded_mixed_files
                 ):
        self.speech_files=speech_files
        self.noise_files=noise_files
        self.encoded_speech_files=encoded_speech_files
        self.encoded_mixed_files=encoded_mixed_files

    def _get_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        #print(f"Using {device} device")
        return device

    def encode_data(self, tx_steps = 700000):
        device = self._get_device()
        encoder_checkpoint = os.path.join('./AudioDec','exp', 'autoencoder', 'symAD_vctk_48000_hop300', f"checkpoint-{tx_steps}steps.pkl")
        generator = Generator()
        generator.load_state_dict(torch.load(encoder_checkpoint, map_location='cpu')['model']['generator'])
        encoder = generator.encoder
        encoder.to(device).eval()


        training_data = EncoderDataset(
            noise_files=self.noise_files,
            speech_files=self.speech_files,
            noise_count=2,
            query="*.wav",
            load_fn=sf.read,
            return_utt_id=True,
            subset_num=-1,
            snr_low=-5,
            snr_high=10,
            pitch=False,
            fourier=True
        )
        batch_size=6
        loader=DataLoader(training_data, batch_size=batch_size, shuffle=True)
        
        for i, (utt_id, mixed, speech) in enumerate(loader):
            mixed_code=encoder(mixed.unsqueeze(1).to(device))
            speech_code=encoder(speech.unsqueeze(1).to(device))
            with torch.no_grad():
                for batch in range(batch_size):
                    mixed = mixed_code[batch].unsqueeze(0).detach().cpu()
                    speech = speech_code[batch].unsqueeze(0).detach().cpu()
                    torch.save(mixed.detach().cpu(), f"{self.encoded_mixed_files}/{utt_id[batch]}.pt")
                    torch.save(speech.detach().cpu(), f"{self.encoded_speech_files}/{utt_id[batch]}.pt")
            if i%100 == 0:
                print(f"Iteration: {i}")