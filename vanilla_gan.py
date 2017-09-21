#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:08:32 2017

@author: xueyunzhe
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

with tf.name_scope('train_input'):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    z = tf.placeholder(tf.float32, shape=[None, 100])


def generator(z):
    with tf.name_scope('Generator'):
        layer1 = tf.contrib.layers.fully_connected(
                        inputs=z,
                        num_outputs=128,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=None,
                        variables_collections='generator',
                        trainable=True,
                        scope='G_hidden1'
                    )
        output = tf.contrib.layers.fully_connected(
                        inputs=layer1,
                        num_outputs=784,
                        activation_fn=tf.sigmoid,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=None,
                        variables_collections='generator',
                        trainable=True,
                        scope='G_prob'
                    )
        return output
    
def discriminator(x, reuse=False):
    with tf.name_scope('Discriminator'):
        layer1 = tf.contrib.layers.fully_connected(
                        inputs=x,
                        num_outputs=128,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=None,
                        variables_collections='discriminator',
                        trainable=True,
                        reuse=reuse,
                        scope='D_hidden1'
                    )
        d_prob = tf.contrib.layers.fully_connected(
                        inputs=layer1,
                        num_outputs=1,
                        activation_fn=tf.identity,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=None,
                        variables_collections='discriminator',
                        trainable=True,
                        reuse=reuse,
                        scope='D_logit'
                    )
        
        
        return d_prob
    
    
def plot(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28), cmap='Greys_r')
        
    return fig


G_sample = generator(z)
d_real = discriminator(x, reuse=False)
d_fake = discriminator(G_sample, reuse=True)

d_loss_real = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=tf.ones_like(d_real, dtype=tf.int64),
    logits=d_real,
)
d_loss_fake = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=tf.zeros_like(d_fake, dtype=tf.int64),
    logits=d_fake,
)
d_loss = d_loss_real + d_loss_fake
g_loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=tf.ones_like(d_fake, dtype=tf.int64),
    logits=d_fake,
)

d_solver = tf.train.AdamOptimizer().minimize(d_loss, 
            var_list=[i for i in tf.trainable_variables() if 'D' in i.name])
g_solver = tf.train.AdamOptimizer().minimize(g_loss, 
            var_list=[i for i in tf.trainable_variables() if 'G' in i.name])

mb_size = 128
z_dim = 100

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


if not os.path.exists('out/'):
    os.makedirs('out/')
    
i=0

for it in range(1000000):
    if it%100000 == 0:
        samples = sess.run(G_sample, feed_dict={z: sample_Z(16, z_dim)})
        
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i+=1
        plt.close(fig)
        
    x_mb, _ = mnist.train.next_batch(mb_size)
    
    _, d_loss_curr = sess.run([d_solver, d_loss], feed_dict={x:x_mb, z:sample_Z(mb_size, z_dim)})
    _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z: sample_Z(mb_size, z_dim)})
    
    if it%100000 == 0:
        print('Iter: {}'.format(it))
        print('D_loss: %.4f'%d_loss_curr)
        print('G_loss: %.4f'%g_loss_curr)
        print()