## MNIST Classifier | Basic TensorFlow Tutorial

# Import the MNIST Dataset from TensorFlow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Import TensorFlow
import tensorflow as tf
# Input placeholder
x = tf.placeholder(tf.float32, [None, 784])
# Weights, initialized with full zeros
W = tf.Variable(tf.zeros([784, 10]))
# Biases, initialized with full zeros
b = tf.Variable(tf.zeros([10]))
# Implement the SoftMax Model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training
# placeholder for the correct values
y_ = tf.placeholder(tf.float32, [None, 10])
# Implement loss function - Cross Entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# apply choice of optimization algo (Minimize Cross Entropy using Gradient Descent)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Launch the model in an interactive session
sess = tf.InteractiveSession()
# Initialize variables
tf.global_variables_initializer().run()
# Train the model 1000 times
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
# Evaluating the model
# tf.argmax(y,1) -> most likely label for each input, y_,1 is the correct label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# correct_prediciton would yield a format of  [True True False True]
# cast the datatype into numeric [1 1 0 1] or 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print the accuracy
print("The accuracy is: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
  