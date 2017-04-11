import tensorflow as tf
from read_dataset import FER2013Reader
import os.path
from time import time
from datetime import datetime

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
  
  dataset = FER2013Reader()
  
  print("> Setting up convolutional graph...")
  y_conv = activate(x_image, W_conv1,  W_conv2, W_conv3, W_fc1, W_fc2,
                    b_conv1, b_conv2, b_conv3, b_fc1, b_fc2, keep_prob)
  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    train, train_label = dataset.Train
    test, test_label = dataset.Test
    
    start_training_time = datetime.today()
    
    print("> Starting training at {:%Y-%m-%d %H:%M:%S}".format(start_training_time))
    steps = 20000
    batch_size = 50
    print_interval = 25
    last_time = 0
    for step in range(steps):
      batch = dataset.get_batch(batch_size)
      if step % print_interval == 0:
        if step == 0:
          last_time = time()
        else:
          train_accuracy = accuracy.eval(feed_dict={x_image:   batch[0],
                                                    y_:        batch[1],
                                                    keep_prob: 1.0      })
          current_time = time()
          eta = int((current_time - last_time) * (steps - step)/print_interval)
          print("> step {:5}/{:5} -> Training accuracy: {:02.0f}% -> ETA: {}h {:02.0f}m {:02.0f}s ({} images/sec)".format(
                step, steps, train_accuracy*100, eta//3600, (eta//60)%60, eta%60, int(batch_size * print_interval / (current_time - last_time))))
          last_time = current_time
      train_step.run(feed_dict={x_image:   batch[0],
                                y_:        batch[1],
                                keep_prob: 1.0      })
    end_training_time = datetime.today()
    print("> Training done. Start: {:%Y-%m-%d %H:%M:%S}, end: {:%Y-%m-%d %H:%M:%S} Testing accuracy...".format(start_training_time, end_training_time))
    final_accuracy = accuracy.eval(feed_dict={x_image:test,
                                              y_:y_test,
                                              keep_prob: 1.0 })
    print("> Final accuracy: %g"%final_accuracy)
  

if __name__ == '__main__':
  main()
