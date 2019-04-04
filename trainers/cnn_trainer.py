from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class CNNTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(CNNTrainer, self).__init__(sess, model, data, config,logger)
           


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

        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        # self.model.build_model(batch_x,batch_y)
        #h = tf.get_session_handle(batch_x,batch_y)
        #self.model.x:batch_x,self.model.y:batch_y,
        # feed_dict = {self.data.handle:self.training_handle}
        # batch_x, batch_y = self.sess.run(self.data.next_batch,feed_dict=feed_dict)

        # img = batch_x[0]
        # plt.figure(figsize=[10,10])
        # plt.subplot(121)
        # plt.imshow(img.reshape(28,28),cmap='gray')
        # plt.show()
                


        batch_x = np.reshape(batch_x,[16,28,28,1])

        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict={self.model.x:batch_x,self.model.y:batch_y,self.model.is_training: True})
        return loss, acc
    def test_step(self):
        # feed_dict = {self.data.handle:self.training_handle}
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))

        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                     feed_dict={self.model.x:batch_x,self.model.y:batch_y,self.model.is_training: False})
        return loss, acc



# class CNNTrainer(BaseTrain):
#     def __init__(self, sess, model, data, config,logger):
#         super(CNNTrainer, self).__init__(sess, model, data, config,logger)
        

#     def train_epoch(self):
#         loop = tqdm(range(self.config.num_iter_per_epoch))
#         losses = []
#         accs = []
#         for _ in loop:
#             loss, acc = self.train_step()
#             losses.append(loss)
#             accs.append(acc)
#         loss = np.mean(losses)
#         acc = np.mean(accs)

#         cur_it = self.model.global_step_tensor.eval(self.sess)
#         summaries_dict = {
#             'loss': loss,
#             'acc': acc,
#         }
#         print('epoch: ',cur_it,' ',summaries_dict)
#         self.logger.summarize(cur_it, summaries_dict=summaries_dict)
#         self.model.save(self.sess)

#     def train_step(self):
#         batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
#         feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
#         _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
#                                      feed_dict=feed_dict)
#         return loss, acc