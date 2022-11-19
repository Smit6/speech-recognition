from gather import gather_data

from data_preprocessing import PreprocessData


from torch.utils.data import DataLoader


def main():
  train_dataset, test_dataset = gather_data()
  train_dataset = PreprocessData(train_dataset, validation_set=False)
  DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)


if __name__ == '__main__':
  main()
