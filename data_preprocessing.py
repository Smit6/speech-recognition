import torch
import torch.nn as nn
import torchaudio


# TODO: Questions: Log Mel Spectrogram vs Mel Spectrogram
class LogMelSpectogram(nn.Module):
  # TODO: Understand the parameters
  def __init__(self, sample_rate=16000, n_mels=128, win_length=160, hop_length=80):
    super(LogMelSpectogram).__init__()
    self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length)

  def forward(self, waveform):
    mel_spectrogram = self.mel_spectrogram(waveform)
    # Add 1e-6 to avoid taking log of zero
    log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
    return log_mel_spectrogram



class PreprocessData(nn.utils.data.Dataset):
  def __init__(self, dataset, validation_set, sample_rate=16000, n_mels=128, win_length=160, hop_length=80):
    super(PreprocessData).__init__()
    self.dataset = dataset
    if validation_set:
      self.preprocess_audio = LogMelSpectogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length)
    else:
      # TODO: Why no frequency and time masking for validation set?
      self.preprocess_audio = nn.Sequential(
        LogMelSpectogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
      )


  def __getitem__(self, index):
    # waveform, sample_rate, label, speaker_id, chapter_id, utterance_id
    waveform, _, label, _, _, _ = self.dataset[index]
    log_mel_spectrogram = self.preprocess_audio(waveform)
    return log_mel_spectrogram, label

  def __len__(self):
    return len(self.dataset)
