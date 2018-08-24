import argparse
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def train():
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()

  with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, 784])
    true_pred_y = tf.placeholder(tf.float32, [None, 10])

  with tf.name_scope('Input_Reshape'):
    input_reshape = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('Input', input_reshape, 10)

  def weights(input):
    W = tf.truncated_normal(input, stddev=0.1)
    return tf.Variable(W)

  def biases(input):
    b = tf.constant(0.1, shape=input)
    return tf.Variable(b)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


  def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def add_layer(input_x, size_layer,layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = weights(size_layer)
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = biases([size_layer[3]])
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):
            activations= tf.nn.relu(conv2d(input_x, W)+ b)
            tf.summary.histogram('activations', activations)
        return activations


  def add_conn_layer(input, size, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            shape = int(input.get_shape()[1])
            W = weights([shape, size])
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = biases([size])
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input, W) + b
            tf.summary.histogram('pre_activations', preactivate)
        activations = tf.nn.relu(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


  layer_1 = add_layer(input_reshape, size_layer=[5, 5, 1, 32], layer_name="Layer_1")
  with tf.name_scope('Pools'):
    l1_pool = max_pool(layer_1)


  layer_2 = add_layer(l1_pool, size_layer=[5, 5, 32, 64], layer_name="Layer_2")
  with tf.name_scope('Pools'):
    l2_pool = max_pool(layer_2)

  with tf.name_scope('Reshape'):
    l2_reshape = tf.reshape(l2_pool,[-1,7*7*64], name = "Reshape")
  connected_layer_1 = tf.nn.relu(add_conn_layer(l2_reshape,1024, layer_name="Fully_Conn_L1"))

  #dropout
  keep_prob = tf.placeholder(tf.float32)
  conn_l1_dropout = tf.nn.dropout(connected_layer_1,keep_prob=keep_prob)
  y_pred = add_conn_layer(conn_l1_dropout,10,layer_name="Fully_conn_L2")

  #Loss Function
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=true_pred_y, logits=y_pred)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  #Optimizer
  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  #Accuracy
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(true_pred_y, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  def feed_dict(train):
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, true_pred_y: ys, keep_prob: k}

  for i in range(1000):
    if i % 10 == 0:
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      if i % 100 == 99:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
