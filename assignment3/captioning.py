# This library is used for Assignment3_Part2_ImageCaptioning

# Write your own image captiong code
# You can modify the class structure
# and add additional function needed for image captionging

import copy
import json
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import pickle
import heapq


class Config():
    def __init__(self):
        self.num_epochs = 200
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.lr_decay_factor = 0.97
        self.num_steps_per_decay = 10000
        self.clip_gradients = 5.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_decode_layers = 1 
        self.dim_decode_layer = 1024
                
        self.kernel_init_scale = 0.08
        self.kernel_reg_scale = 1e-4
        self.activity_reg_scale = 0.0
        self.fc_drop_rate = 0.3
        self.lstm_drop_rate = 0.3
        
        self.print_period = int(self.num_epochs / 10)
        self.save_period = 1000
        self.save_dir = './models/'
        

class NN(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.prepare()

    def prepare(self):
        config = self.config

        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval = -config.kernel_init_scale,
            maxval = config.kernel_init_scale)

        if self.is_train and config.kernel_reg_scale > 0:
            self.fc_kernel_regularizer = layers.l2_regularizer(
                scale = config.kernel_reg_scale)
        else:
            self.fc_kernel_regularizer = None

        if self.is_train and config.activity_reg_scale > 0:
            self.fc_activity_regularizer = layers.l1_regularizer(
                scale = config.activity_reg_scale)
        else:
            self.fc_activity_regularizer = None

    def dense(self,
              inputs,
              units,
              activation = tf.tanh,
              use_bias = True,
              name = None):
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.dense(
            inputs = inputs,
            units = units,
            activation = activation,
            use_bias = use_bias,
            trainable = self.is_train,
            kernel_initializer = self.fc_kernel_initializer,
            kernel_regularizer = self.fc_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def dropout(self,
                inputs,
                name = None):
        return tf.layers.dropout(
            inputs = inputs,
            rate = self.config.fc_drop_rate,
            training = self.is_train)
        
        
class Captioning():
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.nn = NN(config)
        self.img_features = tf.placeholder(dtype=tf.float32, shape=[None, config.dim_img_features])
        
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.build()

    def build(self):
        print("Building the LSTM...")
        config = self.config

        if self.is_train:
            captions = tf.placeholder(
                dtype = tf.int32,
                shape = [None, config.caption_maxlen])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [None, config.caption_maxlen])

        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.word_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)

        batch_size = tf.shape(self.img_features)[0]
        initial_memory = tf.zeros([batch_size, lstm.state_size[0]])
        initial_output = tf.zeros([batch_size, lstm.state_size[1]])

        predictionsArr = []
        cross_entropies = []
        predictions_correct = []
        num_steps = config.caption_maxlen
        image_emb = self.img_features

        last_memory = initial_memory
        last_output = initial_output
        last_word = image_emb

        last_state = last_memory, last_output

        for idx in range(num_steps):
            if idx == 0:
                word_embed = image_emb
            else:
                with tf.variable_scope("word_embedding"):
                    word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                        last_word)
            with tf.variable_scope("lstm"):
                current_input = word_embed
                output, state = lstm(current_input, last_state)
                memory, _ = state

            with tf.variable_scope("decode"):
                expanded_output = output
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictionsArr.append(prediction)

                self.probs = probs

            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = captions[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(captions[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

            last_state = state
            if self.is_train:
                last_word = captions[:, idx]
            else:
                last_word = prediction

            tf.get_variable_scope().reuse_variables()
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis=1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis=1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

            self.captions = captions
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
        self.predictions = tf.stack(predictionsArr, axis=1)

        if self.is_train:
            learning_rate = tf.constant(config.learning_rate)
            if config.lr_decay_factor < 1.0:
                def _learning_rate_decay_fn(learning_rate, global_step):
                    return tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps = config.num_steps_per_decay,
                        decay_rate = config.lr_decay_factor,
                        staircase = True)
                learning_rate_decay_fn = _learning_rate_decay_fn
            else:
                learning_rate_decay_fn = None

            with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )

                opt_op = tf.contrib.layers.optimize_loss(
                    loss = self.total_loss,
                    global_step = self.global_step,
                    learning_rate = learning_rate,
                    optimizer = optimizer,
                    clip_gradients = config.clip_gradients,
                    learning_rate_decay_fn = learning_rate_decay_fn)

            self.opt_op = opt_op
            
        print("LSTM built.")
            
    def train(self, sess, img_features, captions):
        print("Training...")
        config = self.config
        batch_size = config.batch_size
        epochs = config.num_epochs
        num_data = img_features.shape[0]
        print_period = config.print_period
        masks = []
        for caption in captions:
            current_num_words = len(caption[caption!=0])
            current_masks = np.zeros(config.caption_maxlen)
            current_masks[:current_num_words] = 1.0
            masks.append(current_masks)
        masks = np.array(masks)
        
        if num_data % batch_size != 0:
            num_steps = int(num_data / batch_size + 1)
        else:
            num_steps = int(num_data / batch_size)
            
        for i in range(epochs):
            for j in range(num_steps):
                offset = j * batch_size
                img_features_batch = img_features[offset:(offset + batch_size), :]
                captions_batch = captions[offset:(offset + batch_size), :]
                masks_batch = masks[offset:(offset + batch_size), :]
                
                feed_dict = {self.img_features: img_features_batch,
                             self.captions: captions_batch,
                             self.masks: masks_batch}
                loss_val, global_step = sess.run([self.opt_op,
                                                    self.global_step],
                                                   feed_dict=feed_dict)
                    
            if i == 0 or (i+1) % print_period  == 0:
                print("Current Cost: %.2f \t Epoch %d/%d" % (loss_val, i + 1, epochs))
        self.save()
        print("Training done.")

    def predict(self, sess, img_features):
        config = self.config
        batch_size = config.batch_size
        num_data = img_features.shape[0]
        captions = []

        if num_data % batch_size != 0:
            num_steps = int(num_data / batch_size + 1)
        else:
            num_steps = int(num_data / batch_size)
            
        for j in range(num_steps):
            offset = j * batch_size
            img_features_batch = img_features[offset:(offset + batch_size), :]
            caption_data = sess.run(self.predictions, feed_dict={self.img_features:img_features_batch})[0]
            captions.append(caption_data)
        captions = np.array(captions)
        return captions


    def decode(self, expanded_output):
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            logits = self.nn.dense(expanded_output,
                                   units = config.word_size,
                                   activation = None,
                                   name = 'fc')
        else:
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.word_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits
        
    def save(self):
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, "model")

        print(" Saving the model...")
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("The model saved.")

    def load(self, sess, model_file=None):
        config = self.config
        save_path = model_file

        print("Loading the model...")
        
        np_load_old = np.load

        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        data_dict = np.load(save_path).item()
        np.load = np_load_old
        count = 0
        for v in tf.global_variables():
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("The model loaded.")
