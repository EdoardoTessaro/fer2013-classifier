import tensorflow as tf
from pandas import read_csv
from multiprocessing import Pool

def string_to_list(s):
  return list(map(float, s.split(" ")))

def one_hot(array, depth):
  def one_hot_helper(value):
    l = [0] * depth
    l[value] = 1
    return l
  return list(map(one_hot_helper, array))

def extract_data(array, start, end):
  return list(map(string_to_list, array[start:end]))

def extract_labels(array, start, end):
  return one_hot(list(map(int, array[start:end])), 7)

def file_size(filename):
  with open(filename) as f:
    for i, l in enumerate(f):
      pass
  return i+1

def get_list_from_raw_data(raw_data, index):
  return raw_data.ix[:,index].values.tolist()

class FER2013Reader(object):
  def __init__(self, start=0, size=None, verbose=True):
    filename = './fer2013/fer2013.csv'
    filesize = file_size(filename)
    if size is None:
      self.size = filesize
    else:
      self.size = size
    
    if verbose:
      print("> Loading dataset...")
    if size is None:
      raw_data = read_csv(filename)
    else:
      raw_data = read_csv(filename, skiprows=start, nrows=size+1)
    
    if verbose:
      print("> Converting data...")
    usage = raw_data.ix[:,2].values.tolist()
    for train_size in range(len(usage)):
      if usage[train_size] != 'Training':
        break
    self.train_size = train_size
    data = get_list_from_raw_data(raw_data, 1)
    labels = get_list_from_raw_data(raw_data, 0)
    self.train = extract_data(data, None, train_size)
    self.train_label = extract_labels(labels, None, train_size)
    self.test = extract_data(data, train_size, None)
    self.test_label = extract_labels(labels, train_size, None)
    
    self.batch_index=0
    
    if verbose:
      print("> Dataset loaded.")
  
  def get_batch(self, batch_size):
    a = self.batch_index
    self.batch_index += batch_size
    self.batch_index = self.batch_index % self.train_size
    b = self.batch_index
    
    if a < b:
      return self.train[a:b], self.train_label[a:b]
    else:
      return (self.train[a:] + self.train[:b]), (self.train_label[a:] + self.train_label[:b])
  
  @property
  def Train(self):
    return self.train, self.train_label
  
  @property
  def Test(self):
    return self.test, self.test_label
    
