import torch
import torch.nn as nn
import torch.utils.data
import torchaudio


class TextTransform:
  '''Maps characters to integers and vice versa'''
  def __init__(self):
    char_map_str = '''
      ' 0
      <SPACE> 1
      a 2
      b 3
      c 4
      d 5
      e 6
      f 7
      g 8
      h 9
      i 10
      j 11
      k 12
      l 13
      m 14
      n 15
      o 16
      p 17
      q 18
      r 19
      s 20
      t 21
      u 22
      v 23
      w 24
      x 25
      y 26
      z 27
      '''
    self.char_map = {}
    self.index_map = {}
    for line in char_map_str.strip().split('\n'):
      ch, index = line.split()
      self.char_map[ch] = int(index)
      self.index_map[int(index)] = ch
    self.index_map[1] = ' '

  def text_to_int(self, text):
    ''' Use a character map and convert text to an integer sequence '''
    int_sequence = []
    for c in text:
      if c == ' ':
        ch = self.char_map['<SPACE>']
      else:
        ch = self.char_map[c]
      int_sequence.append(ch)
    return int_sequence

  def int_to_text(self, labels):
    ''' Use a character map and convert integer labels to an text sequence '''
    string = []
    for i in labels:
      string.append(self.index_map[i])
    return ''.join(string).replace('', ' ')



# TODO: Questions: Log Mel Spectrogram vs Mel Spectrogram
class LogMelSpectogram(nn.Module):
  # TODO: Understand the parameters
  def __init__(self, sample_rate=8000, n_mels=81, win_length=160, hop_length=80):
    super(LogMelSpectogram, self).__init__()
    self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length)

  def forward(self, waveform):
    mel_spectrogram = self.mel_spectrogram(waveform)
    # Add 1e-6 to avoid taking log of zero
    log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
    return log_mel_spectrogram



class PreprocessData(torch.utils.data.Dataset):
  def __init__(self, dataset, validation_set, sample_rate=8000, n_mels=81, win_length=160, hop_length=80):
    super(PreprocessData).__init__()
    self.dataset = dataset
    self.text_transform = TextTransform()
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

    # Convert waveform to log mel spectrogram
    log_mel_spectrogram = self.preprocess_audio(waveform)
    # Get the length of the log mel spectrogram
    log_mel_spectrogram_len = log_mel_spectrogram.shape[0] // 2 # TODO: Why divide by 2?

    # Convert label text to integer sequence
    label_in_int = torch.tensor(self.text_transform.text_to_int(label.lower()))
    # Get the length of the label
    label_len = torch.tensor(len(label_in_int))

    return log_mel_spectrogram, label_in_int, log_mel_spectrogram_len, label_len

  def __len__(self):
    return len(self.dataset)
