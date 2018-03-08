import os
import sys
import numpy as np
import tensorflow as tf
import math
from model import *
from loadData import *
from sklearn.model_selection import KFold

EVAL_CONST = 100
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


def train(config, train_data_set, valid_data_set):

    with tf.Graph().as_default():

        data_placeholder, labels_placeholder = placeholder_inputs_feedforward(config.batch_size, config.data_size)
        # create network
        logits = create_CNN(data_placeholder, config)

        loss = tf.reduce_mean(compute_loss(logits, labels_placeholder))
        tf.summary.scalar('loss', loss)

        # define optimizer
        train_op = create_train_op(loss, config.learning_rate)

        # op for counting number of correct predictions
        current_cnt_op = eval_prediction(logits, labels_placeholder)
        
        init_op = tf.global_variables_initializer()
        
        saver = tf.train.Saver(max_to_keep=1)

        print('With Session...')
        # run training 
        with tf.Session() as sess:
          
            summary_op = tf.summary.merge_all()

            # Summary Writer for visualizing results on Tensorboard
            writer = tf.summary.FileWriter(config.logs_path,graph=sess.graph)

            sess.run(init_op)
               
            print('Training Set Size:',train_data_set.num_samples)
            trainDist = np.zeros(4)
            for label in train_data_set._labels:
                trainDist[label] += 1
            print('Training Set Distribution:')
            print('Normal:{}  AF:{}  Other:{}   Noisy:{}'.format(trainDist[0],trainDist[1],trainDist[2],trainDist[3]))

            print('Validation Set Size:',valid_data_set.num_samples)
            validDist = np.zeros(4)
            for label in valid_data_set._labels:
                validDist[label] += 1
            print('Validation Set Distribution:')
            print('Normal:{}  AF:{}  Other:{}   Noisy:{}'.format(validDist[0],validDist[1],validDist[2],validDist[3]))


                

            for step in range(config.max_iters):
                # get next batch and load into feed dictionary
                train_feed_dict = fill_feed_dict(train_data_set, config.batch_size, data_placeholder, labels_placeholder)
                _, loss_val,summary = sess.run([train_op, loss,summary_op], feed_dict=train_feed_dict)
                writer.add_summary(summary,step)
            
                # display current training results
                if (step + 1) % EVAL_CONST == 0:
                    print('=======Step {0}======'.format(step))
                    print('Epochs Completed:',train_data_set._epochs_completed)
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


                        # calculate confusion matrix
                        predictions = tf.argmax(logits,1)
                        confusion_matrix_tf = tf.confusion_matrix(labels_placeholder,tf.argmax(logits,1))
                        cm = confusion_matrix_tf.eval(feed_dict = {data_placeholder:valid_data_set._data,labels_placeholder:valid_data_set._labels})
                        totals = np.sum(cm,axis=1)
                        cm = cm/ totals[:,None]
                        print('CM:')
                        print(cm)
        

                # calculate confusion matrix
                predictions = tf.argmax(logits,1)
                confusion_matrix_tf = tf.confusion_matrix(labels_placeholder,tf.argmax(logits,1))
                cm = confusion_matrix_tf.eval(feed_dict = {data_placeholder:valid_data_set._data,labels_placeholder:valid_data_set._labels})
                totals = np.sum(cm,axis=1)
                cm = cm/ totals[:,None]
                print('CM:')
                print(cm)

                
            return sess


    
def load_data(config):
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = read_data('./data/training2017',config)

    
    # shuffle training data
    indices = np.arange(train_labels.shape[0])
    np.random.shuffle(indices)

    train_data = train_data[indices]
    train_labels = train_labels[indices]
    

    train_dataset = DataSet(train_data, train_labels)
    valid_dataset = DataSet(valid_data,valid_labels)
    test_dataset = DataSet(test_data, test_labels)
    
    return train_dataset, valid_dataset, test_dataset




def main():

    config = Config()

    print('Loading data...')
    train_data_set, valid_data_set, test_data_set = load_data(config)
    print('Complete.')
    
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    print('Begin Training...')
    sess = train(config, train_data_set, valid_data_set)
    
    

if __name__ == '__main__':
    main()

