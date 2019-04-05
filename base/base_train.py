import tensorflow as tf
import numpy as np
import data_loader.data_generator as data_generator
from tqdm import tqdm


class Trainer:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
        self.sess.run(self.init)
        if isinstance(data, data_generator.ReadTFRecords):
            self.handle = sess.run(data.dataset_iterator.string_handle())



    def train(self):
        try:
            for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
                self.train_epoch()
                self.sess.run(self.model.increment_cur_epoch_tensor)
        except tf.errors.OutOfRangeError:
            tf.logging.info('Finished experiment.')

    def test(self):
        try:
            losses = []
            accs = []
            while True:
                loss, acc = self.test_step()
                losses.append(loss)
                accs.append(acc)
        except tf.errors.OutOfRangeError:
            tf.logging.info('Finished experiment.')
            loss = np.mean(losses)
            acc = np.mean(accs)
            summaries_dict = {
                'loss': loss,
                'acc': acc,
                'batchs': len(accs)
            }
            print(summaries_dict)
            #summarize(self, step, summarizer="train", scope="", summaries_dict=None)
            # self.logger.summarize(1,summarizer='test',summaries_dict=summaries_dict)
            # self.model.save(self.sess)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        print('epoch: ',cur_it,' ',summaries_dict)
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):

        feed_dict = {self.data.handle:self.handle}                
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
        
    def test_step(self):
        feed_dict = {self.data.handle:self.handle}
        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

