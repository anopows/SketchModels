import numpy as np
import tensorflow as tf
import os
from time import time

from utility import in_jupyter, get_ckpt_vars, get_logger
from sketch_io import train_data, valid_data, test_data, write_results
from model_fns import classification
from loss_fns  import classification_loss
from loss_fns  import classification_summary
from sketch_util import batch_gather, mask_sketch


from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim      import conv2d

class SketchModel(object):
    def __init__(self, init_flags=True, my_flags=None):
        if init_flags:
            self._init_flags()
        if my_flags:
            for k,v in my_flags.items(): self.FLAGS[k].value = v

        self._init_model_params()
        self._init_environment()

        # Kaggle data
        self.tr_data = self.get_data('train')
        self.ev_data = self.get_data('eval')
        if not self.FLAGS.small:
            self.te_data = self.get_data('test')
        
        # Training, Validation, Prediction nodes
        if self.FLAGS.use_imgs:
            self.tr_data.update(self.convolutional_model(self.tr_data['num_images'], 
                                                         self.tr_data['images'], 
                                                         self.tr_data['image_indices'], training=True))
            self.ev_data.update(self.convolutional_model(self.ev_data['num_images'],
                                                         self.ev_data['images'], 
                                                         self.ev_data['image_indices'], training=False))
            self.te_data.update(self.convolutional_model(self.te_data['num_images'],
                                                         self.te_data['images'], 
                                                         self.te_data['image_indices'], training=False))

        self.tr_states, self.tr_losses, self.tr_metrics = \
            self.model_fn(self.tr_data, mode='train')
        self.ev_states, self.ev_losses, self.ev_metrics = \
            self.model_fn(self.ev_data, mode='evaluate')
        if not self.FLAGS.small:
            self.pr_states, _, _ = \
                self.model_fn(self.te_data, mode='predict')

    def _init_flags(self, cnn_only=False):
        self.flags = tf.app.flags
        self.FLAGS = self.flags.FLAGS

        # Parse command line arguments
        if in_jupyter(): 
            tf.app.flags.DEFINE_string('f', '', 'kernel')

        # Runtime options
        self.flags.DEFINE_string ('prefix', 'SketchModel',   "Model name prefix")
        self.flags.DEFINE_integer('gpu',                -1,   "If having multiple GPUs set to id. -1: do nothing")
        self.flags.DEFINE_integer('train_step',    250000,   "Number of training steps to train model")
        # Model parameters
        self.flags.DEFINE_boolean('small',           False,   "Use small dataset")
        self.flags.DEFINE_boolean('use_imgs',        False,   "Use image data too")
        self.flags.DEFINE_integer('cnn_size',         1024,   "CNN logits at last layer")
        self.flags.DEFINE_boolean('pretrained_cnn',   True,   "Use a pretrained CNN network")
        self.flags.DEFINE_boolean('finetuning_cnn',   True,   "Continuing finetuning the CNN network")
        self.flags.DEFINE_float  ('learning_rate',    4e-4,   "Learning rate")
        self.flags.DEFINE_boolean('anneal_lr',        True,   "Annealing learning rate by half every x steps")
        self.flags.DEFINE_integer('max_anneal',          5,   "How many times to half learning rate")
        self.flags.DEFINE_integer('clip_gradient',       1,   "Clip gradient magnitude")
        self.flags.DEFINE_integer('max_seqlen',        150,   "Clip lengths of sketches")

        # to delete
        self.flags.DEFINE_boolean('bnorm_before',       False,  "Use batch normalization before classification layers")
        self.flags.DEFINE_boolean('bnorm_middle',       False,  "Use batch normalization in between classification layers")
        self.flags.DEFINE_boolean('bnorm_eval_update',  False, "Update statistics in validating")

        self.cnn_only = False
        if cnn_only: 
            self.cnn_only = True
            return

        # RNN specifics
        self.flags.DEFINE_integer('batch_size',        30,   "Batch size")
        self.flags.DEFINE_integer('hidden_size',      512,   "LSTM state size")
        self.flags.DEFINE_integer('lstm_stack',         2,   "How many LSTMs to stack at every layer")

    def _init_model_params(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step') 
        self.train_learning_rate = tf.Variable(self.FLAGS.learning_rate, trainable=False, name='current_train_learning_rate')
        self.half_learning_rate  = tf.assign(self.train_learning_rate, 0.5*self.train_learning_rate)
        self.max_seqlen = self.FLAGS.max_seqlen if self.FLAGS.max_seqlen else None

        if self.cnn_only: return

        self.mean    = tf.constant([ 0.04, -0.67]) # TODO put this into input pipeline
        self.std_dev = tf.constant([34.53, 26.54])
        self.model_name     = '{}_hs{}x{}'.format(self.FLAGS.prefix, self.FLAGS.lstm_stack, self.FLAGS.hidden_size)
        if self.FLAGS.batch_size != 30:         self.model_name += '_bs' + str(self.FLAGS.batch_size)
        if self.FLAGS.small:                    self.model_name += '_smallModel'
        if self.FLAGS.learning_rate != 4e-4:    self.model_name += '_lr' + str(self.FLAGS.learning_rate)
        if self.FLAGS.anneal_lr:                self.model_name += '_anneal-lr'
        if self.FLAGS.max_seqlen:               self.model_name += '_maxlen' + str(self.FLAGS.max_seqlen)
        if self.FLAGS.use_imgs:
            self.model_name += '_usingImgs'
            if self.FLAGS.cnn_size != self.FLAGS.hidden_size:
                self.model_name += '_imglogits' + str(self.FLAGS.cnn_size)
            if self.FLAGS.pretrained_cnn: 
                self.model_name += '_pretrainedCNN'
            if self.FLAGS.finetuning_cnn:
                self.model_name += '_finetuningCNN'
        
        # Probably not set
        if self.FLAGS.bnorm_before:             self.model_name += '_bn-before'
        if self.FLAGS.bnorm_middle:             self.model_name += '_bn-middle'
        if self.FLAGS.bnorm_eval_update:        self.model_name += '_bnEvalUpdate'


    def _init_environment(self):
        # Folders for storage/retrival
        self.main_directory   = '../' 
        self.checkpoint_dir  = self.main_directory + 'checkpts/'  + self.model_name + '/'
        self.tensorboard_dir = self.main_directory + 'tb_graphs/' + self.model_name + '/'
        self.solutions_dir   = self.main_directory + 'solutions/' + self.model_name + '/'
        logging_directory = self.main_directory + 'logs/'
        for dir_name in [self.checkpoint_dir, self.tensorboard_dir, self.solutions_dir, logging_directory]:
            if not os.path.exists(dir_name): os.makedirs(dir_name)

        self.num_classes = 10 if self.FLAGS.small else 345
    
        if self.FLAGS.gpu != -1:
            os.environ["CUDA_DEVICE_ORDER"]    = 'PCI_BUS_ID'
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.FLAGS.gpu) # export CUDA_VISIBLE_DEVICES=5

        if in_jupyter():
            get_ipython().system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"') # pylint: disable=E0602
        else:
            os.system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')

        log_in_file=True
        if log_in_file:
            logger = get_logger(self.model_name, logging_directory)
            self.logging = logger.info
        else:
            self.logging = print

    def get_data(self, mode):
        data_fn = None
        if mode == 'train':
            data_fn = train_data
        elif mode == 'eval':
            data_fn = valid_data
        elif mode == 'test':
            data_fn = test_data
        else:
            raise Exception("Mode {} not known".format(mode))

        data_dict = {}
        if mode != 'test':
            data_dict['id'], _, data_dict['length'], data_dict['sketch'], *img, data_dict['labels'] = \
                data_fn(batch_size=self.FLAGS.batch_size,
                        epochs=None,
                        small=self.FLAGS.small,
                        with_imgs=self.FLAGS.use_imgs,
                        max_seqlen=self.max_seqlen
                        )
        else:
            data_dict['id'], _, data_dict['length'], data_dict['sketch'], *img = \
                data_fn(batch_size=self.FLAGS.batch_size,
                        small=self.FLAGS.small,
                        with_imgs=self.FLAGS.use_imgs,
                        max_seqlen=self.max_seqlen
                        )
        if img: 
            data_dict['num_images']    = img[0]
            data_dict['images']        = img[1]
            data_dict['image_indices'] = img[2]

        if not self.cnn_only:
            data_dict['sketch_prepr'] = self._preprocess(data_dict['sketch'], data_dict['length'])
        return data_dict

    def _revert_normalization(self, sketches, lengths=None):
        sketches_coords = (sketches[:,:,:2] * self.std_dev) + self.mean
        sketches = tf.concat([sketches_coords, sketches[:,:,2:]], axis=2)
        if lengths is not None:
            mask_sketch(sketches, lengths)
        return sketches

    def _preprocess(self, sketches, lengths):
        def normalize_movement(sketch_diff, lengths):
            vals = (sketch_diff[:,:,:2] - self.mean) / self.std_dev
            vals = mask_sketch(vals, lengths)
            return tf.concat([vals, sketch_diff[:,:,2:]], axis=2)

        def calc_diffs(sketches, lengths):
            sketches = tf.cast(sketches, tf.float32)
            batch_size  = tf.shape(sketches)[0]
            sketch_vals = sketches[:, :, 0:2]

            # add (122.5,122.5) pts as first pt. To preserve starting point when calculating diffs
            first_rows  = 122.5 * tf.ones(tf.stack([batch_size, 1, 2])) 
            sketch_vals = tf.concat([first_rows, sketch_vals], axis=1) 
            # Difference calculation
            sketch_vals = sketch_vals[:, 1:, 0:2] - sketch_vals[:, 0:-1, 0:2]
            # Throw away the last row of difference when it should be padding
            sketch_vals = mask_sketch(sketch_vals, lengths)
            return tf.concat([sketch_vals, sketches[:,:,2:]], axis=2)

        with tf.variable_scope('preprocessing'):
            sketches      = tf.cast(sketches, tf.float32)
            # Calculate differences
            sketches      = calc_diffs(sketches, lengths)
            # Normalize differnces/movements
            sketches      = normalize_movement(sketches, lengths)
            return sketches
        
    def model_fn(self, data, mode=None):
        """
        Args:
            data:   dict with attribute sketch & length
            labels: class labels if not in predict mode
            mode:   'train'/'evaluate'/'predict'
        
        Returns:
            dict of states
            dict of losses (+train_op)
            dict of metrics
        
        """
        raise NotImplementedError("Subclass must override this method")
        
    def convolutional_model(self, num_images, images, img_indices, training):
        with tf.variable_scope('cnn/resnet', reuse=tf.AUTO_REUSE):
            bs      = tf.shape(images)[0]
            imgsize = [224,224]
            # Flatten and cast to float
            images = tf.reshape(images, [-1, imgsize[0], imgsize[1], 1]) # num images x img_height x img_width x channels
            images = tf.cast(images, dtype=tf.float32)
            # Run through ResNet
            logits, _ = resnet_v2.resnet_v2_50(images, is_training=training) # ?x1x1x2048
            logits = conv2d(inputs=logits, num_outputs=self.FLAGS.cnn_size, kernel_size=[1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
            if not self.FLAGS.finetuning_cnn: 
                logits = tf.stop_gradient(logits)
            logits = tf.reshape(logits, [bs, -1, self.FLAGS.cnn_size])
            # Add first 0 logits, not yet an image available
            logits_0 = tf.concat([tf.zeros_like(logits[:,:1]), logits], axis=1)

            # propagate logits to the corresponding input (input el has last used snapshot img logits)
            # bs x num_snapshots x logit_size ---> bs x max_seqlen x logit_size
            def _propagate(logits, indices):
                bs   = tf.shape(logits)[0]
                # prepare indices to get the correct logits
                bs_ind = tf.range(bs)
                bs_ind = tf.tile(bs_ind[:,None], [1,self.max_seqlen])
                tog_ind = tf.stack([bs_ind, indices], axis=2)
                tog_ind = tf.reshape(tog_ind, (-1,2))
                # gather imgs from indices
                out = tf.gather_nd(logits, tog_ind)
                return tf.reshape(out, (bs,self.max_seqlen,self.FLAGS.cnn_size))

            output_dict = {'last_img_logits': batch_gather(logits, num_images), # img logits for finished sketches
                           'all_padded_img_logits': logits}                     # img logits for snapshot images
            if self.max_seqlen: # at every sketch point, corresponding img logits
                output_dict.update({'sketch_img_logits': _propagate(logits_0, img_indices)})

            return output_dict
                    

    def restore_latest(self, sess, scope=None, logging_fn=print, model_name=None):
            model_name = model_name or self.model_name
            checkpoint_dir  = self.main_directory + 'checkpts/' + model_name + '/'

            var_list = tf.global_variables(scope=scope)
            saver = tf.train.Saver(var_list=var_list)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path: # pylint: disable=E1101
                try:
                    saver.restore(sess, ckpt.model_checkpoint_path) # pylint: disable=E1101
                    logging_fn("Values were restored (scope: {})".format(scope))
                except:
                    logging_fn("Failure restoring variables (scope: {})".format(scope))
                    logging_fn("Variables in checkpoint:")
                    for v in get_ckpt_vars(self.checkpoint_dir): print(*v)
                    raise
                return True

            logging_fn("No values were restored for model {}\n\
                        Checkpoint directory: {} \n\
                        Starting with new weights".format(model_name, self.checkpoint_dir))
            return False

    def _eval_step(self, sess, eval_op, num_eval_steps):
        eval_sum = 0
        for _ in range(num_eval_steps):
            eval_sum += sess.run(eval_op)
        return eval_sum/num_eval_steps

    def _summary_step(self, i, sess, writer, log_in_tb):
        assert 'train_loss' in self.tr_losses
        assert 'train_loss' in self.ev_losses
        assert 'classification_accuracy' in self.ev_metrics

        summaries =      [sum for sum in self.tr_metrics.values() if (sum is not None) and sum.dtype == tf.string]
        summaries.extend([sum for sum in self.ev_metrics.values() if (sum is not None) and sum.dtype == tf.string])
        
        *summaries_out, tr_loss, ev_loss, ev_cl_acc = sess.run(summaries + [self.tr_losses['train_loss'], 
                                                                            self.ev_losses['train_loss'],
                                                                            self.ev_metrics['classification_accuracy']])
        if log_in_tb: 
            for summary in summaries_out:
                writer.add_summary(summary, i)
        
        self.logging("Step {:6d} | Train loss: {:6.4f} | Validation loss: {:6.4f} | Validation classification accuracy: {:6.4f}".format(
            i, tr_loss, ev_loss, ev_cl_acc))

    def get_session(self):
        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        sess.run(tf.tables_initializer())
        return sess

    def validate(self, sess, num_eval_batches=500):
        assert 'classification_accuracy' in self.ev_metrics and 'classification_accuracy_map3' in self.ev_metrics

        self.logging("Start validation ({} batches)".format(num_eval_batches))
        sum_acc_middle, sum_acc_last, sum_acc_top3   = 0.0, 0.0, 0.0
        
        summaries = [('classification_accuracy', 'accuracy for the last token'),
                     ('classification_accuracy_map3', 'top3 accuracy')]
        have_middle = False
        if 'classification_accuracy_middle' in self.ev_metrics: 
            have_middle = True
            summaries.insert(0, ('classification_accuracy_middle', 'accuracy of half finished sketches'))

        summary_text = ""
        for _, stext in summaries:
            stext = "\nThe average {:33}".format(stext) + ": {:6.4f}"
            summary_text += stext  
        summary_text += "\n========\n"

        for i in range(num_eval_batches):
            *eval_accuracy_middle, eval_accuracy_last, eval_accuracy_top3 = \
                sess.run([self.ev_metrics[metric_name] for metric_name, _ in summaries])
            output = []
            if have_middle:
                sum_acc_middle  += eval_accuracy_middle[0]
                output.append(sum_acc_middle)
            sum_acc_last    += eval_accuracy_last
            sum_acc_top3    += eval_accuracy_top3
            output.extend([sum_acc_last, sum_acc_top3])

            if i%100 == 0: 
                self.logging("After {:4d} steps:".format(i+1))
                self.logging(summary_text.format(*output))

        output = []
        if have_middle:
            output.append(sum_acc_middle/num_eval_batches)
        output.extend([sum_acc_last/num_eval_batches, sum_acc_top3/num_eval_batches])

        self.logging("\n\nFinal accuracy:\n=========\n" + summary_text.format(*output))


    def train(self, 
              log_in_tb=True, 
              log_in_file=True, 
              save_checkpts=True, 
              restore=True,
              finalize_graph=True):
        
        assert 'train_loss' in self.ev_losses
        assert 'train_op'   in self.tr_losses

        if save_checkpts and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if log_in_tb and not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)

        self.logging("Current model:\n\t {}".format(self.model_name))

        with self.get_session() as sess: 
            if restore:
                restore = self.restore_latest(sess)
                if not restore and self.FLAGS.use_imgs and self.FLAGS.pretrained_cnn:
                    self.restore_latest(sess, scope='cnn/resnet', model_name='conv_units' + str(self.FLAGS.cnn_size))
            else: 
                self.logging("No values were restored as requested. Starting with new weights")

            # Early stopping params
            best_eval_loss        = tf.Variable(initial_value=tf.float32.max, trainable=False, dtype=tf.float32)
            update_placeholder    = tf.placeholder(tf.float32, shape=[])
            update_best_loss_op   = tf.assign(best_eval_loss, update_placeholder)
            improvement_ratio     = 0.02
            evals_wo_improvement  = 0
            max_eval_wo_improvement = 3
            max_annealing           = self.FLAGS.max_anneal
            num_annealing           = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
            inc_num_annealing       = tf.assign(num_annealing, num_annealing+1)
            sess.run(tf.variables_initializer([best_eval_loss, num_annealing]))
            # Saver
            if save_checkpts or restore: saver = tf.train.Saver(max_to_keep=5)
            # Log in Tensorboard
            if log_in_tb: writer = tf.summary.FileWriter(self.tensorboard_dir, sess.graph)

            if finalize_graph:
                sess.graph.finalize()

            num_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            self.logging("Number of trainable variables: {}".format(num_vars))
            
            starti = self.global_step.eval()
            seconds, last_step = time(), starti

            for i in range(starti, self.FLAGS.train_step):
                # Summary
                summary_every    = 100
                evaluation_every = 2000
                evaluation_steps = 500

                # Summary
                if log_in_tb and (i+1) % summary_every    == 0:
                    self._summary_step(i, sess, writer, log_in_tb)
                # Evaluation
                if (i+1) % evaluation_every == 0: 
                    cur_eval_loss = self._eval_step(sess, self.ev_losses['train_loss'], evaluation_steps)
                    best_eval_loss_out = sess.run(best_eval_loss)
                    # Improvement, store checkpoint
                    if (best_eval_loss_out-cur_eval_loss) > np.abs(best_eval_loss_out*improvement_ratio):
                        sess.run(update_best_loss_op, feed_dict={update_placeholder: cur_eval_loss})
                        evals_wo_improvement = 0
                        # Store checkpoint
                        saver.save(sess, self.checkpoint_dir + 'checkpoint', self.global_step.eval())
                        self.logging("New checkpoint stored at step {}. Evaluation loss {:5.3f}".format(
                                    i+1, cur_eval_loss))
                    else:
                        evals_wo_improvement += 1
                        self.logging("x No new checkpoint stored at step {}. Evaluation loss {:5.3f}".format(
                                    i+1, cur_eval_loss))

                    # 5 steps no improvement, half the learning rate or finish
                    if evals_wo_improvement >= max_eval_wo_improvement:
                        num_annealing_out = sess.run(num_annealing)
                        if num_annealing_out >= max_annealing: 
                            self.logging("Early stopping after {} steps. Current loss {:5.3f} | Best loss {:5.3f}".format(
                                        i, cur_eval_loss, best_eval_loss_out
                            ))
                            break
                        else: 
                            sess.run(inc_num_annealing)
                            evals_wo_improvement = 0
                            sess.run(self.half_learning_rate)
                            self.logging("Decrease learning rate through annealing, lr now: {:.2e}".format(
                                sess.run(self.train_learning_rate)
                            ))

                # Regular training
                sess.run(self.tr_losses['train_op'])


            self.logging("Training finished")
            self.logging("Validation:")
            self.validate(sess)
            self.post_train(sess)
            self.predict(sess, validate_before=False)

    
    def post_train(self, sess):
        pass

    def predict(self, sess=None, restore=False, validate_before=True):
        assert 'id' in self.te_data and 'prediction' in self.pr_states

        def _predict(sess):
            test_preds = []
            test_ids   = []
            if validate_before:
                self.validate(sess)

            i = 0
            while True:
                try:
                    i += 1
                    if (i+1)%150 == 0: print("Step", i+1)
                    cur_id, cur_pred = sess.run([self.te_data['id'], self.pr_states['prediction']])
                    test_ids.append(cur_id)
                    test_preds.append(cur_pred)
                except tf.errors.OutOfRangeError:
                    break

            return test_preds, test_ids

        if sess is not None: 
            if restore: self.restore_latest(sess)
            test_preds, test_ids = _predict(sess)
        else: # do testing in new session
            with self.get_session() as sess:
                if restore: self.restore_latest(sess)
                test_preds, test_ids = _predict(sess)
                
        preds = np.concatenate(test_preds, axis=0)
        ids   = np.concatenate(test_ids,   axis=0)

        write_results(ids, preds, path=self.solutions_dir)

