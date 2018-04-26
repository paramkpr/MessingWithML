import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
input data --> wieght --> hidden layer 1 --> weights
--> hidden layer 2 (activation funtion) --> weight --> output

# Data is passed straight though Feed - Forward

compare output to intended --> cost function(cross entropy)
optimization function (optimizer) > min. cost (AdamOptimizer)

# Going backwards here is Backpropogation

feed forward + backpropogation =  One Epoch
'''

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # 100 images at a time

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def NeuralNetworkModel(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl2])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl3])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}

	output_layer = {'weights':tf.Variable(tf.random_normal([784, n_classes)),
	'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}