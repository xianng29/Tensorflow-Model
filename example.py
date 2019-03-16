import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from models.cnn_model import CNNModel
from trainers.example_trainer import ExampleTrainer
from trainers.cnn_trainer import CNNTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as ex:
        print("missing or invalid arguments")
        print(ex)
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)


    # create an instance of the model you want
    # create trainer and pass all the previous components to it

    create_model = None
    create_trainer = None
    if config.exp_name == 'CNN':
        create_model = CNNModel
        create_trainer = CNNTrainer
    elif config.exp_name == 'Example':
        create_model = ExampleModel
        create_trainer = ExampleTrainer

    model = create_model(config)
    trainer = create_trainer(sess, model, data, config, logger)

    

    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
