from base.base_model import BaseModel
import tensorflow as tf
import numpy as np


class CNNModel(BaseModel):
    def __init__(self, config):
        super(CNNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.

        self.is_training = tf.placeholder(tf.bool)
        
        self.x = tf.placeholder(tf.float32, shape=[self.config.batch_size]+self.config.input_shape)
        self.y = tf.placeholder(tf.float32, shape=[self.config.batch_size]+[self.config.number_class])

        strides = [1,self.config.strides,self.config.strides,1]
        padding = self.config.padding
        #CNN architecture
        #conv1
        with tf.variable_scope('conv1'):
            weight = self.get_weight_variable(self.config.filter_1)
            bias = self.get_bias_variable(self.config.filter_1[-1])

            conv1 = tf.nn.conv2d(self.x,weight,strides,padding=padding)
            conv1 = tf.nn.bias_add(conv1,bias)
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.nn.max_pool(conv1, ksize=strides, strides=strides,padding=padding)
        #conv 2
        with tf.variable_scope('conv2'):
            weight = self.get_weight_variable(self.config.filter_2)
            bias = self.get_bias_variable(self.config.filter_2[-1])

            conv2 = tf.nn.conv2d(conv1,weight,strides,padding=padding)
            conv2 = tf.nn.bias_add(conv2,bias)
            conv2 = tf.nn.relu(conv2)
            conv2 = tf.nn.max_pool(conv2, ksize=strides, strides=strides,padding=padding)
        # Fully connected layer
        with tf.variable_scope('fc'):
            flatten = tf.layers.flatten(conv2)
            fc_size = [flatten.get_shape().as_list()[1],self.config.fc_bias]
            print('fc_size => ',fc_size)

            weight = self.get_weight_variable(fc_size)
            bias = self.get_bias_variable(self.config.fc_bias)

            fc = tf.add(tf.matmul(flatten,weight), bias)
            fc = tf.nn.relu(fc)
            # Apply Dropout
            fc = tf.nn.dropout(fc, self.config.dropout)

        # Output, class prediction
        with tf.variable_scope('out'):
            weight = self.get_weight_variable([self.config.fc_bias,self.config.number_class])
            bias = self.get_bias_variable(self.config.number_class)
            out = tf.add(tf.matmul(fc, weight), bias)
        
        #loss
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=out))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
