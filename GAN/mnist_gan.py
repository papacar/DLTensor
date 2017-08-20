import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

def discriminator(images, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        w1 = tf.get_variable('w1', [5, 5, 1, 32],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', [32],
            initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=w1, strides=[1, 1, 1, 1],
            padding='SAME')
        d1 = d1 + b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        w2 = tf.get_variable('w2', [5, 5, 32, 64],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', [64],
            initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=w2, strides=[1, 1, 1, 1],
            padding='SAME')
        d2 = d2 + b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME')

        # First fully connected layer
        w3 = tf.get_variable('w3', [7 * 7 * 64, 1024],
            initializer= tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('b3', [1024],
            initializer=tf.truncated_normal_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, w3)
        d3 = d3 + b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        w4 = tf.get_variable('w4', [1024, 1],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('b4', [1],
            initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, w4) + b4

    return d4


def generator(z, batch_size, z_dim):
    with tf.variable_scope("generator") as scope:
        w1 = tf.get_variable('w1', [z_dim, 3136], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', [3136],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(z, w1) + b1
        g1 = tf.reshape(g1, [-1, 56, 56, 1])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
        g1 = tf.nn.relu(g1)

        # Generate 50 features
        w2 = tf.get_variable('w2', [3, 3, 1, z_dim/2],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', [z_dim/2],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        g2 = tf.nn.conv2d(g1, w2, strides=[1, 2, 2, 1], padding='SAME')
        g2 = g2 + b2
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
        g2 = tf.nn.relu(g2)
        g2 = tf.image.resize_images(g2, [56, 56])

        # Generate 25 features
        w3 = tf.get_variable('w3', [3, 3, z_dim/2, z_dim/4],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('b3', [z_dim/4],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = tf.nn.conv2d(g2, w3, strides=[1, 2, 2, 1], padding='SAME')
        g3 = g3 + b3
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
        g3 = tf.nn.relu(g3)
        g3 = tf.image.resize_images(g3, [56, 56])

        # Final convolution with one output channel
        w4 = tf.get_variable('w4', [1, 1, z_dim/4, 1],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('b4', [1],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        g4 = tf.nn.conv2d(g3, w4, strides=[1, 2, 2, 1], padding='SAME')
        g4 = g4 + b4
        g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28  x 1
    return g4

batch_size = 50
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, shape=[None, z_dimensions],
    name='z_placeholder')
x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1],
    name='x_placeholder')

Gz = generator(z_placeholder, batch_size, z_dimensions)
# Gz holds the generated images

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilites
# for the real MNIST images

Dg = discriminator(Gz, reuse=True)
# Dg will hold discriminator prediction probabilties for generated images

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'discriminator' in var.name]
g_vars = [var for var in tvars if 'generator' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# Begin Training
sess = tf.Session()
sess.run([tf.global_variables_initializer()])
tf.get_variable_scope().reuse_variables()
tf.summary.scalar("Generator_loss", g_loss)
tf.summary.scalar("Discriminator_loss_fake", d_loss_fake)
tf.summary.scalar("Discriminator_loss_real", d_loss_real)
images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image("Generated_images", images_for_tensorboard, 10)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# Pretraining
# For avoiding some impredicted condition
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_fake, d_trainer_real, d_loss_fake, d_loss_real], 
        {x_placeholder:real_image_batch, z_placeholder:z_batch})

    if (i % 100 == 0):
        print("dLossReal: ", dLossReal, "dLossFake: ", dLossFake)

# Real training start
for i in range(100000):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

    # Discriminator
    _, __, dLossReal, dLossFake = sess.run([d_trainer_fake, d_trainer_real, d_loss_fake, d_loss_real], 
        {x_placeholder:real_image_batch, z_placeholder:z_batch})

    # Generator
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 10 == 0:
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder:real_image_batch})
        writer.add_summary(summary, i)
