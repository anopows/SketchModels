import tensorflow as tf
import numpy as np
from time import time
import os
from collections import namedtuple

from sketch_io import train_data, valid_data, test_data
from sketch_util import batch_gather
from model_fns import classification
from loss_fns import classification_loss, classification_summary

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim      import conv2d


flags = tf.app.flags
FLAGS = flags.FLAGS

# Runtime options
flags.DEFINE_integer('train_steps', 100000, "Number of training steps to train model")
flags.DEFINE_integer('batch_size',     128, "Batch size")
flags.DEFINE_integer('cnn_size',       256, "Number of logits as output from CNN")
flags.DEFINE_float  ('learning_rate', 1e-3, "Learning rate optimizer")
flags.DEFINE_string ('name',        "CNNsimple", "Model name")

model_name = FLAGS.name + '_bs' + str(FLAGS.batch_size) + '_cnnSize' + str(FLAGS.cnn_size)
checkpoint_dir  = '../checkpts/'  + model_name + '/'
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

# Input
*foo, num_imgs, imgs, _, labels = train_data(batch_size=FLAGS.batch_size, with_imgs=True, num_snapshots=1)
# Reshaping
last_imgs = batch_gather(imgs, num_imgs)
last_imgs = tf.reshape(last_imgs, [-1, 224,224,1])
last_imgs = tf.cast(last_imgs, tf.float32)

# CNN
with tf.variable_scope('cnn/resnet', reuse=tf.AUTO_REUSE):
    logits, _ = resnet_v2.resnet_v2_50(last_imgs, is_training=True) # ?x1x1x2048
    logits = conv2d(inputs=logits, num_outputs=FLAGS.cnn_size, kernel_size=[1, 1], activation_fn=None,
                    normalizer_fn=None, scope='logits')
logits = tf.squeeze(logits, [1,2])

# Classification
cl_logits = classification(logits, layer_sizes=[FLAGS.cnn_size, 345], training=False)

# Training
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cl_logits, labels=labels)
avg_loss = tf.reduce_mean(loss)
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
learning_rate = tf.Variable(initial_value=FLAGS.learning_rate, trainable=False, name='learning_rate')
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)

# Summaries
cl_acc      = classification_summary(cl_logits, labels)[1]['classification_accuracy']
cl_acc_top3 = classification_summary(cl_logits, labels, top_k=3)[1]['classification_accuracy_avgtop3']

def anneal_update(state: dict) -> bool:
    improvement_ratio = 0.005
    # Improvement, store checkpoint
    if (state['best'] == float('inf')) or (state['best']-state['cur']) > np.abs(state['best']*improvement_ratio):
        state['best'] = state['cur']
        state['num_same'] = 0
        return True
    else:
        state['num_same'] += 1
        if state['num_same'] >= state['max_same']: 
            state['num_same'] = 0
            state['num_anneal'] += 1
        return False

evaluation_every, checkpt_every = 50, 400
max_same, max_anneal = 4, 8

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    half_lr = tf.assign(ref=learning_rate, value=0.5*learning_rate)
    anneal_state = dict(best=float('inf'), cur=float('inf'),
                                num_same=0, max_same=max_same, num_anneal=0, max_anneal=max_anneal)

    num_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("Number of trainable variables: {}".format(num_vars))
    # Saver
    saver = tf.train.Saver(max_to_keep=5)
    sess.graph.finalize()

    start, lasti = time(), 0
    for i in range(FLAGS.train_steps):
        sess.run(train_op)
        if i%evaluation_every == 0:
            avg_ms_passed = (time() - start) * 1000
            avg_ms_passed = avg_ms_passed / (i+1-lasti)

            # collect statistics
            losses_out, cl_out, map3_out = [], [], []
            for _ in range(20):
                sum_out = sess.run([avg_loss, cl_acc, cl_acc_top3])
                losses_out.append(sum_out[0])
                cl_out.append(sum_out[1])
                map3_out.append(sum_out[2])

            cur_avg_loss = np.mean(losses_out)
            cur_avg_cl   = np.mean(cl_out)
            cur_avg_map3 = np.mean(map3_out)
            print("{:5} steps. {:6.2f} ms/step".format(i+1, avg_ms_passed), end="\t")
            print("|| Avg loss: {:5.3f} | Cl. accuracy: {:5.3f} | MAP3 accuracy: {:5.3f} ||".format(
                cur_avg_loss, cur_avg_cl, cur_avg_map3
            ), end=" ") 

            # Anneal and save checkpts
            if i%checkpt_every == 0:
                anneal_state['cur'] = cur_avg_loss
                if anneal_update(anneal_state):
                    saver.save(sess, checkpoint_dir + 'checkpoint', i)
                    print("\t = New checkpoint stored at step {} =".format(i+1))
                elif anneal_state['num_anneal'] > anneal_state['max_anneal']:
                    print("\n===\nFinished learning after {} steps\n===".format(i+1))
                    break
                elif anneal_state['num_same'] == 0:
                    sess.run(half_lr)
                    print("\n===\nStep {:5d}. Halfing learning rate. lr now: {:4.2e}\n===".format(i+1, sess.run(learning_rate)))
                else: 
                    print("\t = No new checkpoint =")
            else: print()
            start, lasti = time(), i


        
