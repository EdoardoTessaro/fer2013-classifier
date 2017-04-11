import tensorflow as tf
from read_dataset import get_dataset
import os.path
import time

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
                        padding='SAME')

def reshape(x):
  return tf.reshape(x, [-1,48,48,1])

def conv_step(x, W, b):
  h_conv = tf.nn.relu(conv2d(x, W) + b)
  h_pool = max_pool_2x2(h_conv)
  return h_pool

def activate(x_image, W_conv1,  W_conv2, W_conv3, W_fc1, W_fc2, b_conv1,
              b_conv2, b_conv3, b_fc1, b_fc2, keep_prob):
  x = reshape(x_image)
  h_conv1 = conv_step(x, W_conv1, b_conv1)
  h_conv2 = conv_step(h_conv1, W_conv2, b_conv2)
  h_conv3 = conv_step(h_conv2, W_conv3, b_conv3)
  
  h_conv3_flat = tf.reshape(h_conv3, [-1, 6*6*128])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
  
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  
  return y_conv

def main():
  
  x_image = tf.placeholder(tf.float32)
  y_ = tf.placeholder(tf.float32)
  
  W_conv1 = weight_variable([5,5,1,32])
  b_conv1 = bias_variable([32])
  W_conv2 = weight_variable([5,5,32,64])
  b_conv2 = bias_variable([64])
  W_conv3 = weight_variable([5,5,64,128])
  b_conv3 = bias_variable([128])
  
  W_fc1 = weight_variable([6*6*128, 1024])
  b_fc1 = bias_variable([1024])
  
  keep_prob = tf.placeholder(tf.float32)
  
  W_fc2 = weight_variable([1024, 7])
  b_fc2 = bias_variable([7])
  
  
  print("> Loading fer2013 dataset...")
  train, train_label, test, test_label = get_dataset()
  
  print("> Setting up convolutional graph...")
  train_label = tf.one_hot(train_label, 7)
  test_label = tf.one_hot(test_label, 7)
  y_conv = activate(x_image, W_conv1,  W_conv2, W_conv3, W_fc1, W_fc2,
                    b_conv1, b_conv2, b_conv3, b_fc1, b_fc2, keep_prob)
  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  batch_size = 50
  train_size = len(train)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = train_label.eval()
    y_test = test_label.eval()
    batch_x = []
    batch_y = []
    current_time = time.time()
    print_interval = 25
    cycle_dataset = 20
    print("> Training...")
    for j in range(cycle_dataset):
      for i in range(train_size // batch_size):
        batch_x = train[i*batch_size:min((i+1)*batch_size, train_size)]
        batch_y = y_train[i*batch_size:min((i+1)*batch_size, train_size)]
        if i % print_interval == 0:
          
          train_accuracy = accuracy.eval(feed_dict={x_image:batch_x, y_:batch_y, keep_prob:1.0})
          print("> step %d/%d -> Training accuracy: %g"%(i + j*(train_size // batch_size), (cycle_dataset*train_size // batch_size)-1, train_accuracy))
        train_step.run(feed_dict={x_image:batch_x, y_:batch_y, keep_prob:1.0})
    
    print("> Training done. Testing accuracy...")
    final_accuracy = accuracy.eval(feed_dict={x_image:test, y_:y_test, keep_prob: 1.0})
    print("> Final accuracy: %g"%final_accuracy)
  

if __name__ == '__main__':
  main()
