from base.base_model import BaseModel
import tensorflow as tf
import numpy as np


class DRMLModel(BaseModel):
    def __init__(self, config):
        super(DRMLModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def region_layer(self,input):
        batch_size = input.get_shape().as_list()[0]   
        rr_arr = []
        with tf.variable_scope('region_layer'):
            for y in range(8):
                arr = []
                for x in range(8):
                    patch = tf.slice(input, [0,0+20*y,0+20*x,0], [batch_size, 20, 20, 32],name='slice_%d_%d' % (y, x))
                    with tf.variable_scope('bn_%d_%d' % (y, x)):
	                    norm = tf.layers.batch_normalization(patch,training=self.is_training)
                    with tf.variable_scope('relu_%d_%d'%(y, x)):
                        relu = tf.nn.relu(norm)
                    
                    with tf.variable_scope('conv_%d_%d' % (y, x)):
                        weights = self.get_weight_variable([3,3,32,32])
                        bias = self.get_bias_variable(32)
                        conv = tf.nn.conv2d(relu,weights,[1,1,1,1],padding='SAME')
                        conv = tf.nn.bias_add(bias)
                        arr.append(conv + patch)
                rr = tf.concat(arr, 2)
                rr_arr.append(rr)
        
        rout = tf.concat(rr_arr,1)
        return rout


    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        self.is_training = tf.placeholder(tf.bool)   
        self.x = tf.placeholder(tf.float32, shape=[None]+self.config.input_shape)
        self.y = tf.placeholder(tf.float32, shape=[None]+[self.config.number_class])

        #CNN architecture
        #conv1
        with tf.variable_scope('conv1'):
            weight = self.get_weight_variable([11,11,3,32])
            conv1 = tf.nn.conv2d(self.x,weight,[1,1,1,1],padding='VALID')
        #Region Layer
        region = self.region_layer(conv1)
        region = tf.nn.relu(region)
        #max pool
        mpool = tf.nn.max_pool(region, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        net = tf.nn.lrn(mpool, alpha=0.0001, beta=0.75)

        con_weights = [[8,8,32,16],[8,8,16,16],[6,6,16,16],[5,5,16,16]]
        stride_list = [1,1,2,1]
        #con 2-5
        for i in range(2,6):
            with tf.variable_scope('conv%d'%i):
                weight = self.get_weight_variable(con_weights[i-2])
                bias = self.get_bias_variable(con_weights[i-2][-1])
                stride = stride_list[i-2]
                net = tf.nn.conv2d(net,weight,[1,stride,stride,1],padding='VALID')
                net = tf.nn.bias_add(net,bias)
                net = tf.nn.relu(net)

        flatten = tf.layers.flatten(net)        
        fc_size_list = [[flatten.get_shape().as_list()[1],4096],[4096,2048]]
        # Fully connected layer
        for i in range(6,8):

            with tf.variable_scope('fc%d'%i):
                
                fc_size = fc_size_list[i-6]
                weight = self.get_weight_variable(fc_size)
                bias = self.get_bias_variable(fc_size[-1])

                fc = tf.add(tf.matmul(flatten,weight), bias)
                fc = tf.nn.relu(fc)
                # Apply Dropout
                fc = tf.nn.dropout(fc, 0.5)
        

        # Output, class prediction
        with tf.variable_scope('out'):
            weight = self.get_weight_variable([2048,self.config.number_class])
            out = tf.matmul(fc, weight)
        
        #loss
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=out))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
