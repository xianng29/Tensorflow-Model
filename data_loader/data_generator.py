import tensorflow as tf
import numpy as np


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
        train_data = train_data/np.float32(255)

        
        self.train_data = train_data
        self.train_labels = train_labels

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.train_labels), batch_size,replace=False)
        num = len(idx)

        select_x = np.reshape(self.train_data[idx],[num]+self.config.input_shape)

        select_y = self.train_labels[idx]       
        
        select_y = np.eye(self.config.number_class)[select_y]
        yield select_x, select_y.t
        #tf.contrib.data.batch_and_drop_remainder(128)
