import os
import sys
import numpy as np
import tensorflow as tf
import math
from model import *
from loadData import *
from sklearn.model_selection import KFold

EVAL_CONST = 20
LAMBDA = 0.1

def eval_prediction(logits, labels):
    """Evaluates the prediction.

    Args:
        logits: A [batch_size, num_class] sized tensor
        labels: A [batch_size] sized tensor
    
    """
    correct = tf.nn.in_top_k(logits,labels,1) # determine if correct class was predicted
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def create_train_op(loss, learning_rate):
    # gradient descent optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def train(config, orig_train_data_set, valid_data_set=None):

    k = config.k
    k_fold = KFold(n_splits=k)

    with tf.Graph().as_default():
        data_placeholder, labels_placeholder = placeholder_inputs_feedforward(config.batch_size, config.data_size)
        logits = feed_forward_net(data_placeholder, config)

        loss = tf.reduce_mean(compute_loss(logits, labels_placeholder))

        train_op = create_train_op(loss, config.learning_rate)

        # summary for cost
        #tf.summary.scalar("loss", loss)


        # op for counting number of correct predictions
        current_cnt_op = eval_prediction(logits, labels_placeholder)
        init_op = tf.global_variables_initializer()

        #summary_op = tf.summary.merge_all

        saver = tf.train.Saver(max_to_keep=1)

        print('With Session...')
        # run training 
        with tf.Session() as sess:
            print('Run Init op')
            sess.run(init_op)

            # Summary Writer for visualizing results on Tensorboard
            #writer = tf.summary.FileWriter(config.logs_path,graph=tf.get_default_graph())
            
            
            totalTrainAcc = 0
            totalValidAcc = 0
            xValStep = 0
            for train_indices, validation_indices in k_fold.split(orig_train_data_set._data):
                print('Step',xValStep,'...')
                # split into valid and train sets for cross-validation
                valid_data_set = DataSet(orig_train_data_set._data[validation_indices],orig_train_data_set._labels[validation_indices])
                train_data_set = DataSet(orig_train_data_set._data[train_indices],orig_train_data_set._labels[train_indices])

                for step in range(config.max_iters):
                    train_feed_dict = fill_feed_dict(train_data_set, config.batch_size, data_placeholder, labels_placeholder)
                    _, loss_val = sess.run([train_op, loss], feed_dict=train_feed_dict)
	    
                    # display current training results
                    if (step + 1) % EVAL_CONST == 0:
                        print('=======Step {0}======'.format(step))
                        print('Train data evaluation:')
                        acc = evaluation(sess, data_placeholder, labels_placeholder, train_data_set, current_cnt_op)
                        print('Train Accuracy: {:.3f}'.format(acc))
		        
                        if valid_data_set:
                            print('Validation data evaluation:')
                            acc = evaluation(sess, data_placeholder, labels_placeholder, valid_data_set, current_cnt_op)
                            print('Validation Accuracy: {:.3f}'.format(acc))
                            print('======')
		
                        checkpoint_file = os.path.join(config.model_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step = step)
		
                # Evaluation at end of cross validation step
                print('===Cross Validation Step {}==='.format(xValStep))
                print('Train data evaluation:')
                acc = evaluation(sess, data_placeholder, labels_placeholder, train_data_set, current_cnt_op)
                totalTrainAcc += acc
                print('Train Accuracy: {:.3f}'.format(acc))
                if valid_data_set:
                    print('Validation data evaluation:')
                    acc = evaluation(sess, data_placeholder, labels_placeholder, valid_data_set, current_cnt_op)
                    totalValidAcc += acc
                    print('Validation Accuracy: {:.3f}'.format(acc))

                # calculate confusion matrix
                confusion_matrix_tf = tf.confusion_matrix(tf.argmax(logits,1),labels_placeholder)
                cm = confusion_matrix_tf.eval(feed_dict = {data_placeholder:valid_data_set._data,labels_placeholder:valid_data_set._labels})
                print('CM:')
                print(cm)
                xValStep += 1
                      
            print('===Final Evaluation===')
            print('Train data evaluation:')
            totalTrainAcc /= xValStep
            print('Train Accuracy: {:.3f}'.format(totalTrainAcc))

            if valid_data_set:
                print('Validation data evaluation:')
                totalValidAcc /= xValStep
                print('Validation Accuracy: {:.3f}'.format(totalValidAcc))
                      
		
def load_data():
    data, labels = read_data('./data/training2017')
    num_points = labels.shape[0]

    train_end = math.floor(0.75*num_points)
    train_data = data[0:train_end,:]
    train_labels = labels[0:train_end]

    test_data = data[train_end+1:num_points,:]
    test_labels = labels[train_end+1:num_points]
    
    train_dataset = DataSet(train_data, train_labels)
    test_dataset = DataSet(test_data, test_labels)
    
    return train_dataset, test_dataset

def main():
    print('Loading data...')
    train_data_set, test_data_set = load_data()
    print('Complete.')
    
    config = Config()
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    print('Begin Training...')
    train(config, train_data_set)

if __name__ == '__main__':
    main()

