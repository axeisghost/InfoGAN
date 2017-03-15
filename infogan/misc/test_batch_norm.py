import prettytensor as pt
import tensorflow as tf
import numpy as np


import custom_ops

BATCH_SIZE = 100

# make the templates
with pt.defaults_scope(phase=pt.UnboundVariable('phase', default=pt.Phase.train)):
    template_batch_norm = (pt.template("input").
                           conv_batch_norm(scale_after_normalization=False))

# input
x_in = tf.placeholder_with_default(tf.random_normal([BATCH_SIZE, 20, 20, 10]),
                                   [None, 20, 20, 10],
                                   name='input')

# construct the graphs
train_output = template_batch_norm.construct(input=x_in)
test_output = template_batch_norm.construct(input=x_in, phase=pt.Phase.test)


# calculates deviation from zero mean and unit variance
def deviation_from_normed(output, dims=[0, 1, 2]):
    mean, variance = tf.nn.moments(output, dims)
    deviation_mean = tf.reduce_max(tf.abs(mean))
    deviation_variance = tf.reduce_max(tf.abs(variance - tf.ones_like(variance)))
    return deviation_mean, deviation_variance

train_deviation_normed = deviation_from_normed(train_output)
test_deviation_normed = deviation_from_normed(test_output)

# init
init_op = tf.initialize_all_variables()


with tf.Session() as sess:

    tf.set_random_seed(1)

    sess.run(init_op)

    # two different constant inputs
    const1 = x_in.eval(session=sess)
    const2 = x_in.eval(session=sess)

    # run for 100 batches of size 100
    for i in range(100):
        sess.run(train_output)

    # the same input should produce the same output in the training phase
    out1 = sess.run(train_output, feed_dict={x_in: const1})
    out2 = sess.run(train_output, feed_dict={x_in: const1})
    is_same = np.all(out1 - out2 == 0)
    print "Training: Same input produces the same output:\n", is_same

    # different inputs should produce differents outputs in the training phase
    out1 = sess.run(train_output, feed_dict={x_in: const1})
    out2 = sess.run(train_output, feed_dict={x_in: const2})
    is_different = np.any(out1 - out2 != 0)
    print "Training: Different input produces different output:\n", is_different

    # same inputs should produce same outputs in the testing phase
    out1 = sess.run(test_output, feed_dict={x_in: const1})
    out2 = sess.run(test_output, feed_dict={x_in: const1})
    is_same = np.all(out1 - out2 == 0)
    print "Testing: Same input produces the same output:\n", is_same

    # the same input should produce a different output in the testing phase
    # after different training time
    out1 = sess.run(test_output, feed_dict={x_in: const1})
    # run for another 100 batches of size 100
    for i in range(100):
        sess.run(train_output)
    out2 = sess.run(test_output, feed_dict={x_in: const1})
    is_different = np.any(out1 - out2 != 0)
    print "Testing: Same input produces different outputs at different training times:\n", is_different

    deviation_test = sess.run(test_deviation_normed, feed_dict={x_in: const1})
    deviation_train = sess.run(train_deviation_normed, feed_dict={x_in: const1})
    print "Training: Maximal difference from zero mean and unit variance:\n", deviation_train
    print "Testing: Maximal difference from zero mean and unit variance:\n", deviation_test



