#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py

import tensorflow as tf

import matplotlib.pyplot as plt
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import summary
import numpy as np
from tqdm import tqdm
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper

"""
MNIST ConvNet example.
about 0.6% validation error after 30 epochs.
"""

IMAGE_SIZE = 28
colors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])


class Model(ModelDesc):
    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.placeholder(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def f(self, sample):
        with tf.variable_scope('f', reuse=tf.AUTO_REUSE):
            x = FullyConnected('fc1', sample, 1000, activation=tf.nn.relu)
            x = FullyConnected('fc2', x, 1000, activation=tf.nn.relu)
            val = FullyConnected('fc3', x, 1, activation=None)
        return val

    def build_graph(self, image, label):
        """This function should build the model which takes the input variables
        and return cost at the end"""

        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        image = tf.layers.flatten(image)
        # image = image * 2 - 1   # center the pixels values at zero
        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use 32 channel convolution with shape 3x3
        with tf.variable_scope('encoder'):
            x = FullyConnected('fc1', image, 1000, activation=tf.nn.relu)
            x = FullyConnected('fc2', x, 1000, activation=tf.nn.relu)
            mu = tf.identity(FullyConnected('fc_mu', x, 2, activation=None), 'mu')
            logvar = FullyConnected('fc_var', x, 2, activation=None)

        eps = tf.random_normal((tf.shape(x)[0], 2))
        z = tf.identity(eps * tf.exp(0.5 * logvar) + mu, name='z')

        z_real = tf.random_normal((tf.shape(x)[0], 2))
        self.f_loss = tf.reduce_mean(-tf.reduce_sum(self.f(z_real) - self.f(z), -1), name='f_loss')
        self.g_loss = tf.reduce_mean(tf.reduce_sum(-self.f(z), -1), name='g_loss')

        z_interp = tf.stop_gradient(z) + tf.random_uniform((tf.shape(x)[0], 1)) * (z_real - tf.stop_gradient(z))
        gradient_f = tf.gradients(self.f(z_interp), [z_interp])[0]
        gp_loss = tf.reduce_mean(tf.maximum(tf.norm(gradient_f, axis=-1) - 1, 0) ** 2, name='gp_loss')

        self.f_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'f')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')

        with tf.variable_scope('decoder'):
            x = FullyConnected('fc1', z, 1000, activation=tf.nn.relu)
            x = FullyConnected('fc2', x, 1000, activation=tf.nn.relu)
            rec = tf.identity(FullyConnected('fc_rec', x, IMAGE_SIZE * IMAGE_SIZE, activation=tf.nn.sigmoid), 'rec')

        kl_loss = -tf.reduce_sum(1 + logvar - mu * mu - tf.exp(logvar), -1)
        kl_loss = tf.reduce_mean(kl_loss, name='kl_loss')

        rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(rec - image), -1), name='rec_loss')
        # total_cost = rec_loss + kl_loss
        total_cost = rec_loss + gp_loss

        summary.add_moving_summary(rec_loss, kl_loss, self.f_loss, self.g_loss, gp_loss)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        # lr = tf.train.MomentumOptimizer(lr, 0.9)
        # This will also put the summary in tensorboard, stat.json and print in terminal,
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


class WGANTrainer(TowerTrainer):
    """ A GAN trainer which runs two optimization ops with a certain ratio."""
    def __init__(self, input, model, f_period=1, g_period=1):
        """
        Args:
            d_period(int): period of each d_opt run
            g_period(int): period of each g_opt run
        """
        super(WGANTrainer, self).__init__()
        self._f_period = int(f_period)
        self._g_period = int(g_period)
        assert min(f_period, g_period) == 1

        # Setup input
        cbs = input.setup(model.get_inputs_desc())
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.get_inputs_desc())
        with TowerContext('', is_training=True):
            self.normal_loss = self.tower_func(*input.get_input_tensors())

        opt = model.get_optimizer()
        with tf.name_scope('optimize'):
            self.f_min = opt.minimize(
                model.f_loss, var_list=model.f_vars, name='f_min')
            self.g_min = opt.minimize(
                model.g_loss, var_list=model.g_vars, name='g_min')
            self.min = opt.minimize(self.normal_loss, name='min')

    def run_step(self):
        # Define the training iteration
        if self.global_step % (self._f_period) == 0:
            self.hooked_sess.run(self.f_min)
        if self.global_step % (self._g_period) == 0:
            self.hooked_sess.run(self.g_min)
        self.hooked_sess.run(self.min)


def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)

    train = PrintData(train)

    return train, test


class Evaluator(Callback):
    def __init__(self, dataset):
        dataset.reset_state()
        self.dataset = dataset
        # self.dataset = iter(dataset)

    def _setup_graph(self):
        self.encoder = self.trainer.get_predictor(['input'], ['encoder/mu'])
        self.decoder = self.trainer.get_predictor(['z'], ['decoder/rec'])

    def _trigger_epoch(self):
        zs = []
        labels = []
        for data in tqdm(self.dataset):
            img, label = data
            z = self.encoder(img[None, ...])[0][0]
            zs.append(z)
            labels.append(label)
        zs = np.asarray(zs)
        labels = np.asarray(labels)
        plt.scatter(zs[:, 0], zs[:, 1], c=colors[labels])
        plt.show()
        # idx = np.random.randint(len(self.dataset))
        # fig = plt.figure()
        # img = next(self.dataset)[0]
        # fig.add_subplot(2, 1, 1)
        # plt.imshow(img)
        # rec = self.decoder(self.encoder(img[None, ...])[0])[0][0].reshape((IMAGE_SIZE, IMAGE_SIZE))
        # fig.add_subplot(2, 1, 2)
        # plt.imshow(rec)
        # plt.show()


if __name__ == '__main__':
    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()

    evaluator = Evaluator(dataset.Mnist('test'))

    # How many iterations you want in each epoch.
    # This len(data) is the default value.
    steps_per_epoch = len(dataset_train)
    model = Model()
    # get the config which contains everything necessary in a training
    # config = TrainConfig(
    #     model=model,
    #     # The input source for training. FeedInput is slow, this is just for demo purpose.
    #     # In practice it's best to use QueueInput or others. See tutorials for details.
    #     data=FeedInput(dataset_train),
    #     callbacks=[
    #         ScheduledHyperParamSetter('learning_rate', [(50, 1e-3), (500, 1e-4)]),
    #         ModelSaver(),   # save the model after every epoch
    #         evaluator,
    #         InferenceRunner(    # run inference(for validation) after every epoch
    #             dataset_test,   # the DataFlow instance used for validation
    #             ScalarStats(    # produce `val_accuracy` and `val_cross_entropy_loss`
    #                 ['kl_loss', 'rec_loss'], prefix='val')),
    #         # MaxSaver has to come after InferenceRunner
    #         MaxSaver('val_accuracy'),  # save the model with highest accuracy
    #     ],
    #     steps_per_epoch=steps_per_epoch,
    #     max_epoch=100,
    # )
    trainer = WGANTrainer(FeedInput(dataset_train), model)
    trainer.train_with_defaults(
        callbacks=[
            ScheduledHyperParamSetter('learning_rate', [(50, 1e-3), (500, 1e-4)]),
            ModelSaver(),  # save the model after every epoch
            evaluator,
            InferenceRunner(  # run inference(for validation) after every epoch
                dataset_test,  # the DataFlow instance used for validation
                ScalarStats(  # produce `val_accuracy` and `val_cross_entropy_loss`
                    ['kl_loss', 'rec_loss'], prefix='val')),
            # MaxSaver has to come after InferenceRunner
            MaxSaver('val_accuracy'),  # save the model with highest accuracy
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )
    # launch_train_with_config(config, WGANTrainer(FeedInput(dataset_train), model))
