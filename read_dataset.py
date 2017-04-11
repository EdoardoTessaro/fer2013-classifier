import tensorflow as tf
from pandas import read_csv

def string_to_list(s):
  return list(map(float, s.split(" ")))

def file_size(filename):
  with open(filename) as f:
    for i, l in enumerate(f):
      pass
  return i+1

def get_dataset(start=0, size=None):
  filename = './fer2013/fer2013.csv'
  filesize = file_size(filename)
  raw_data = read_csv(filename, skiprows=start, nrows=size+1)
  usage = raw_data.ix[:,2].values.tolist()
  for train_size in range(len(usage)):
    if usage[train_size] != 'Training':
      break
  data = raw_data.ix[:,1].values.tolist()
  labels = raw_data.ix[:,0].values.tolist()
  train = list(map(string_to_list, data[:train_size]))
  train_label = list(map(int, labels[:train_size]))
  test = list(map(string_to_list, data[train_size:]))
  test_label = list(map(int, labels[train_size:]))
  return train, train_label, test, test_label
