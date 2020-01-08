import argparse

import tensorflow as tf
import numpy as np

from utils.dataopenset import *
from utils.flip_gradient import flip_gradient
from tqdm import tqdm

parser = argparse.ArgumentParser(description = "Process training parameters.")
parser.add_argument("-batch_size", help="Batch Size for training and testing", type=int, default=2000)
parser.add_argument("-num_epoch", help="Num of Epoch for training", type=int, default=2000)
parser.add_argument("-source", help="Source load",  type=int, default=0)
parser.add_argument("-target", help="Target load",  type=int, default=1)
parser.add_argument("-lr", help="Learning Rate",  type=float, default=0.0001)
parser.add_argument("-size", help="Feature Dimension",  type=float, default=256)
args = parser.parse_args()


def feature_extractor_relu(x, training=True, reuse=False, scope=""):
    """ Feature extracture from Xiang Li.
    
    Args:
      x: a Tensor of size (batch, N) input data
      training: Boolean, if we are at training stage
      reuse: Boolean, if we are reusing the variables
    
    Returns:
      a Tensor of size N_rep
    """
    with tf.variable_scope("feature_ext"+scope, reuse= reuse):
        h = x/8. 
        h = tf.layers.conv1d(h, 10, 3, padding="same", activation=tf.nn.sigmoid, name="f_conv1")
        h = tf.layers.dropout(h, training=training)
        h = tf.layers.conv1d(h, 10, 3, padding="same", activation=tf.nn.sigmoid, name="f_conv2")
        h = tf.layers.dropout(h, training=training)
        h = tf.layers.conv1d(h, 10, 3, padding="same", activation=tf.nn.sigmoid, name="f_conv3")
        h = tf.layers.dropout(h, training=training)
        h = tf.layers.flatten(h, name="f_flat")
        h = tf.layers.dense(h, args.size, activation=tf.nn.sigmoid, name="f_fc")
    return h


def clf(x, training=True, reuse=False, scope=""):
    """ Simple classifier

    Args:
      x: a Tensor of size (batch, N) input data
      training: Boolean, if we are at training stage
      reuse: Boolean, if we are reusing the variables
    """
    with tf.variable_scope("clf"+scope, reuse= reuse):
        h = tf.layers.dense(x, 256, activation=tf.nn.relu, name="clf_fc1")
        h = tf.layers.dropout(h, training=training)
        h = tf.layers.dense(h, 10, name="clf_fc") 
    return h


def discriminator(x, training=True, reuse=False, scope=0):
    """ Simple Discriminator

    Args:
      x: a Tensor of size (batch, N) input data
      training: Boolean, if we are at training stage
      reuse: Boolean, if we are reusing the variables
    """

    with tf.variable_scope("discrimiator"+str(scope), reuse= not training or reuse):
        h = tf.layers.dense(x, args.size*4, activation=tf.nn.relu, name="dis_fc1")
        h = tf.layers.dense(h, args.size*4, activation=tf.nn.relu, name="dis_fc2")
        h = tf.layers.dense(h, 2, name="clf_fc")
    return h


def grl(src, tgt, y, training):
    """ GRL strategy

    Args:
      src: a Tensor of size (batch, N) source input data signal
      tgt: a Tensor of size (batch, N) source input data signal
      y:  label
    """
    if training:
        feat_src = feature_extractor_relu(src, training=training, scope="nnormal")
        logits = clf(feat_src, training=training)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        correct_prediction = tf.equal(y, tf.argmax(logits, axis=-1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        global_step = tf.Variable(0., trainable=False, name='global_step')
        global_step2 = tf.Variable(0., trainable=False, name='global_step2')
        lp = 2. / (1. + tf.exp(-10. * global_step/(args.num_epoch * 2000 / args.batch_size))) - 1

        pretrain = tf.less(global_step, 1.)

        feat_src_pre = feature_extractor_relu(src, training=pretrain, scope="pppre")
        feat_src_ori = feature_extractor_relu(src, training=pretrain, scope="nnormal", reuse=True)
        logits_pre = clf(feat_src_pre, training=True, scope="pppre")
        
        loss2 = 0.2 * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_pre))  
        
        
        feat_tgt = feature_extractor_relu(tgt, training=training, reuse=True, scope="nnormal")
        feat = tf.concat([feat_src, feat_tgt], axis=0)
        feat = flip_gradient(feat, lp*0.01)
        
        logits_dm = discriminator(feat)
        labels_dm = tf.concat([tf.ones([args.batch_size], tf.int32), tf.zeros([args.batch_size], tf.int32)], axis=0)
        loss_dm = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_dm, logits=logits_dm))
        
        
        
        loss_cons = tf.reduce_mean(tf.abs(tf.stop_gradient(feat_src_pre) - feat_src))
        
        loss_total = 0.2 * loss  + loss_dm + loss_cons

        pretrain_op = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss2, global_step=global_step2)
        train_op = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss_total, global_step=global_step, var_list=[v for v in tf.trainable_variables() if "pppre" not in v.name])
        return train_op, pretrain_op
    else:
        feat_src = feature_extractor_relu(src, training=training, reuse=True, scope="nnormal")
        logits = clf(feat_src, training=training, reuse=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        correct_prediction = tf.equal(y, tf.argmax(logits, axis=-1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return loss, accuracy
        

if __name__ == '__main__':
    data_src, label_src = load_cwru(args.source, truncate=120000)
    dataset_src = tf.data.Dataset.from_tensor_slices((data_src, label_src)).shuffle(len(data_src)).repeat(args.num_epoch*4).batch(args.batch_size).prefetch(10)
    iterator_src = dataset_src.make_initializable_iterator()
    src_x, src_y = iterator_src.get_next()

    data_tgt, label_tgt = load_cwru(args.target, mode="20%", truncate=120000)
    data_val, label_val = load_cwru(args.target, truncate=120000)
    
    dataset_tgt = tf.data.Dataset.from_tensor_slices((data_tgt, label_tgt)).shuffle(len(data_tgt)).repeat(4 * args.num_epoch * (len(data_src) // len(data_tgt) + 1)).batch(args.batch_size).prefetch(10)
    dataset_val = tf.data.Dataset.from_tensors((data_val, label_val))
    print(args.source, "--->", args.target)
    
    iterator_tgt = dataset_tgt.make_initializable_iterator()
    iterator_val = dataset_val.make_initializable_iterator()

    train_init_op = [iterator_src.initializer, iterator_tgt.initializer]
    test_init_op = iterator_val.initializer
    tgt_x, tgt_y = iterator_tgt.get_next()
    test_x, test_y = iterator_val.get_next()

    train_op, pretrain_op = grl(src_x, tgt_x, src_y, training=True)
    test_loss, acc_tgt = grl(test_x, test_x, test_y, training=False)

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([train_init_op])
        print("pre training...")
        for ep in tqdm(range(args.num_epoch)):
            for j in range(len(data_src)//args.batch_size): 
                sess.run([pretrain_op])
            
        print("adv training...")
        for ep in tqdm(range(args.num_epoch)):
            for j in range(len(data_src)//args.batch_size):
                sess.run([train_op])
                     
            if ep%20==0 or ep==args.num_epoch-1:
                sess.run([test_init_op])
                test_loss_, acc_ = sess.run([test_loss, acc_tgt])
        print(ep, "Test loss:", test_loss_, "Test Accuracy" ,acc_)

    with open("./log", "a") as f:
        f.write(str(args.source)+str(args.target)+": " + str(acc_)+"\n")



