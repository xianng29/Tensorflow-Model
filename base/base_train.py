import tensorflow as tf
import numpy as np


class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
        self.sess.run(self.init)

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
                loss, acc = self.train_step()
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

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
