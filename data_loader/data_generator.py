import tensorflow as tf
import numpy as np
import os


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
        train_data = train_data/np.float32(255)


        if config.split == 'train':
            self.train_data = train_data
            self.train_labels = train_labels
        else:
            self.train_data = eval_data
            self.train_labels = eval_labels
            pass
        
        

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.train_labels), batch_size,replace=False)
        num = len(idx)

        select_x = np.reshape(self.train_data[idx],[num]+self.config.input_shape)

        select_y = self.train_labels[idx]       
        
        select_y = np.eye(self.config.number_class)[select_y]
        yield select_x, select_y
        #tf.contrib.data.batch_and_drop_remainder(128)

class ReadTFRecords:
    def __init__(self, config):
        self.config = config
        self.isTrain = config.split == 'train'
        root_path = self.config.data_dir

        file_name = os.path.join(root_path, self.config.split+'.tfrecords')
        valid_path = os.path.join(root_path,'validate.tfrecords')
        with tf.name_scope('input'):
            dataset = tf.data.TFRecordDataset(file_name)
            if self.isTrain:
                if os.path.exists(valid_path):
                    dataset2 = tf.data.TFRecordDataset(valid_path)
                    dataset = dataset.concatenate(dataset2)
                dataset = dataset.shuffle(10000 + 3 * self.config.batch_size)
            dataset = dataset.map(self._extract_features)
            dataset = dataset.map(self.normalize)
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.config.batch_size))
            self.dataset = dataset
        self.handle = tf.placeholder(tf.string, shape=[],name='handle')
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, dataset.output_types, dataset.output_shapes)   
        self.dataset_iterator = dataset.make_one_shot_iterator()
        self.next_batch = self.iterator.get_next()
        self.training_handle = sess.run(self.dataset_iterator.string_handle())
        
    def _extract_features(self, example):
        if self.config.exp_name == 'CNN':
            return self.decode_CNN(example)
        elif self.config.exp_name == 'DRML':
            return self.decode_DRML(example)
        else:
            return None
    def decode(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image,self.config.input_shape)

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.decode_raw(features['label'], tf.uint8)
        label = tf.reshape(label,tf.stack([self.config.number_class]))

        image = tf.reshape(image,[28,28])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)

        return image, label


    def decode_CNN(self,example):
        

        return self.decode(example)


        # features = tf.parse_single_example(
        #     example,# Defaults are not specified since both keys are required.
        #     features={
        #         "de_image": tf.FixedLenFeature((), tf.string),
        #         "psd_image": tf.FixedLenFeature((), tf.string),
        #         "sub_id": tf.FixedLenFeature((), tf.string),
        #         "label": tf.FixedLenFeature([], tf.string)
        # })

        # de_image = tf.decode_raw(features['de_image'], tf.uint8)
        # psd_image = tf.decode_raw(features['psd_image'], tf.uint8)
        # label = tf.decode_raw(features['label'], tf.uint8)
        # label = tf.reshape(label,tf.stack([4]))

        # tensor_shape = tf.stack(self.config.input_shape)

        # de_image = tf.reshape(de_image, tensor_shape)
        # psd_image = tf.reshape(psd_image, tensor_shape)

        # de_image = tf.cast(de_image,dtype=tf.float32)
        # psd_image = tf.cast(psd_image,dtype=tf.float32)

        # return de_image, label

    def decode_DRML(self, example):
        features = {
            "image": tf.FixedLenFeature((), tf.string),
            "mask": tf.FixedLenFeature((), tf.string)
        }
        parsed_example = tf.parse_single_example(example, features)
        images = tf.cast(tf.image.decode_jpeg(parsed_example["image"]), dtype=tf.float32)
        images.set_shape([800, 600, 3])
        masks = tf.cast(tf.image.decode_jpeg(parsed_example["mask"]), dtype=tf.float32) / 255.
        masks.set_shape([800, 600, 1])
        return images, masks

    def normalize(self, texture, label):
        #tf.image.per_image_standardization(texture)
        #tf.divide(texture,(1,1,255.0))
        return texture/255.0, label


        
