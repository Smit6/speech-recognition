# Speech Recognition
This project aims to replicate the Deep Speech 2 paper and create a similar deep learning model as Deep Speech 2 to convert speech to text.


### Dataset
From Torchaudio, 100 hours of audio data is utilized for training the Deep Speech 2 model from LibriSpeech. The dataset contains English speech derived from audiobooks.

### Data Preprocessing
- Audio waveforms to Mel Spectrum
- Utterance of integers


### Model
SpeechRecognitionModel

### Limitations
To generate effective results, distributed training on multiple GPUs is required.
