# Neural audio coding for speech enhancement

![Poster](./images/DL_Poster.svg)

## Description of classes

`preprocessing`:
-   `DataEncoder`: Responsible for creating and saving the pre-encoded versions of the audio files used for training.
-   `EncoderDataset`: Responsible for loading the raw audio files, noise files, applying preprocessing steps (such as cutting to equal length, applying bandpass filter) and creating the noisy audio files.

`train`:
-   `EncodedDataset`: Responsible for loading the pre-encoded input and target files for training the model.
-   `LatentNetwork`: Contains the structure of the latent network used in the audio denoising problem.
-   `LatentTrainer`: Responsible for training the latent network with the pre-encoded dataset.
-   `utils`:
    -   `ModelSaveHandler`: Utility class for saving model checkpoints.

## Demos

### From the training data

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/63722535/51110b5f-cbbe-4e47-b78c-afde403ad84f

### Real life setting

1. Noise: Hitting table

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/68492bca-8285-4e99-9853-069fdeaf3b61

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/dbc01134-3eb8-4591-8117-5d27fd53e93b

2. Noise: Pressing plastic bag

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/67584543-29da-4455-95d1-d429be8a6011

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/981b1f4d-6439-4d3c-b594-718a9c0770e6

3. Noise: Hitting glass with a fork

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/30cc1b53-bc45-46f2-9f0f-89fd380b1922

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/5cc30862-6619-466d-9140-7c7cbd5b473e

4. Noise: Playing pop music

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/ecfdb6a1-da5e-41ef-9339-9f996c8f9af4

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/ba249ab8-ef4e-4084-ab46-5036fd9aa79b

5. Noise: Playing classical music

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/bfbe4f20-545f-4e77-bfab-26ab3957d1f9

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/3bdb70e2-b0dd-4559-975a-2856e9d5d67c

6. Noise: Playing dubstep music

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/89d66288-c702-4e08-bf33-2c671b9c9618

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/65610103/cbfe3014-3eba-4ef8-876a-96ff6d080af4
