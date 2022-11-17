import os
import torchaudio

def gather_data():
  if not os.path.isdir('data'):
        os.makedirs('data')

  train_dataset = torchaudio.datasets.LIBRISPEECH('data/', url='train-clean-100', download=True)
  test_dataset = torchaudio.datasets.LIBRISPEECH('data/', url='test-clean', download=True)

  return train_dataset, test_dataset

