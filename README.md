# Neural audio coding for speech enhancement

![Poster](./images/DL_Poster.svg)

### Description of classes

`preprocessing`:
-   `DataEncoder`: Responsible for creating and saving the pre-encoded versions of the audio files used for training.
-   `EncoderDataset`: Responsible for loading the raw audio files, noise files, applying preprocessing steps (such as cutting to equal length, applying bandpass filter) and creating the noisy audio files.

`train`:
-   `EncodedDataset`: Responsible for loading the pre-encoded input and target files for training the model.
-   `LatentNetwork`: Contains the structure of the latent network used in the audio denoising problem.
-   `LatentTrainer`: Responsible for training the latent network with the pre-encoded dataset.
-   `utils`:
    -   `ModelSaveHandler`: Utility class for saving model checkpoints.

### Demos

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/63722535/5d7988a1-199b-4d30-a20c-0de1495dd6e8

https://github.com/madiistvan/Neural-audio-coding-for-speech-enhancement/assets/63722535/51110b5f-cbbe-4e47-b78c-afde403ad84f
