import tensorflow as tf

from sketch_util import batch_gather
from model_fns import classification
from loss_fns  import classification_loss, classification_summary

from SketchModels import SketchModel

class BLSTM(SketchModel):
    def __init__(self, init_flags=True, my_flags=None):
        super(BLSTM, self).__init__()

    def _init_flags(self):
        super(BLSTM, self)._init_flags()
        self.FLAGS['prefix'].value = 'BLSTM'
    
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
        assert mode in ['train', 'evaluate', 'predict']
        assert mode != 'predict' or not self.FLAGS.small

        sketch = data['sketch_prepr']
        length = data['length']
        if mode != 'predict':
            labels = data['labels']

        # Dicts to return
        dstates  = {}
        dlosses  = {}
        dmetrics = {}

        def _average(values, length, max_seqlen=None):
            values = tf.squeeze(values)
            mask   = tf.cast(tf.sequence_mask(length), tf.float32)
            values = mask * values
            return tf.reduce_sum(values) / tf.reduce_sum(mask)

        #########
        # BLSTM #
        #########
        
        with tf.variable_scope('blstm', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('forward_cell', reuse=tf.AUTO_REUSE):
                fw_rnn_cell   = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.FLAGS.hidden_size) for _ in range(self.FLAGS.lstm_stack)])
            with tf.variable_scope('backward_cell', reuse=tf.AUTO_REUSE):
                bw_rnn_cell   = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.FLAGS.hidden_size) for _ in range(self.FLAGS.lstm_stack)])

            (fw_output, bw_output), (fw_last_state, bw_last_state) = \
                tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, sketch, sequence_length=length, dtype=tf.float32)
                
            # Forward output:   (bs, len, 512), dtype=float32
            # Backwards output: (bs, len, 512), dtype=float32

            # Forward state:  ((c=(bs, 512), h=(bs, 512)),  (c=(bs, 512), h=(bs, 512)))
            # Backward state: ((c=(bs, 512), h=(bs, 512)),  (c=(bs, 512), h=(bs, 512))

            output = tf.concat((fw_output, bw_output), axis=2)
            dstates.update({'output': output, 'fw_output': fw_output, 'bw_output': bw_output,
                            'fw_state': fw_last_state, 'bw_state': bw_last_state})

        ##################
        # Classification #
        ##################
        
        classification_fn =     lambda logits : classification(logits, layer_sizes=[64,self.num_classes], 
                                           bnorm_before=self.FLAGS.bnorm_before, bnorm_middle=self.FLAGS.bnorm_middle, # if: apply after first dense layer
                                           training=(mode=='training' or self.FLAGS.bnorm_eval_update), trainable=True, name='classifier')
        
        # blstm output logits --> logits for probabilities
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            dstates['logits_for_cl'] = batch_gather(output, length) # TODO: correct also for VRNN
            # [batch_size x max_len x unit_size] --> [batch_size*max_len x unit_size]
            bs = tf.shape(output)[0]
            cl_logits_all = tf.reshape(output, [-1, 2*self.FLAGS.hidden_size])
            cl_logits_all = classification_fn(cl_logits_all)
            # turn back to batches
            cl_logits_all  = tf.reshape(cl_logits_all, [bs, -1, self.num_classes])
            cl_logits = batch_gather(cl_logits_all, length)
            dstates['cl_logits'] = cl_logits
        
        ########
        # Loss #
        ########

        if mode in ['train', 'evaluate']:
            with tf.variable_scope('blstm/classification_loss', reuse=tf.AUTO_REUSE):
                # Train on last states
                *cl_vals, = classification_loss(cl_logits, labels, scope_to_train=None, 
                                            training=True, clip_gradient=self.FLAGS.clip_gradient,
                                            learning_rate=self.FLAGS.learning_rate, anneal_lr_every=None,
                                            mask=None, additional_loss=None, 
                                            batch_normalization=self.FLAGS.bnorm_before or self.FLAGS.bnorm_middle)
                cl_loss, cl_avg_loss = (cl_vals[0], cl_vals[1])

            dlosses.update({'cl_loss': cl_avg_loss, 'train_loss': cl_avg_loss})

            summary_scope = 'train' if mode == 'train' else 'eval'
            dmetrics.update({'train_loss': tf.summary.scalar(summary_scope + '/train_loss', cl_avg_loss),
                             'cl_loss':    tf.summary.scalar(summary_scope + '/classification_loss', cl_avg_loss)})

        ############
        # Training #
        ############

        if mode == 'train':
            self.cl_global_step, cl_train_op = (cl_vals[2], cl_vals[3]) # unpack from classification loss calculations
            with tf.variable_scope('blstm/loss', reuse=tf.AUTO_REUSE):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.train_learning_rate)
                train_vars = None 
                gradients = optimizer.compute_gradients(cl_avg_loss, var_list=train_vars)
                gradients = [(tf.clip_by_value(grad, -1*self.FLAGS.clip_gradient, self.FLAGS.clip_gradient), var) 
                                    for grad, var in gradients if grad is not None]
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # case of batch norm used
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(
                        grads_and_vars=gradients, global_step=self.global_step
                    )
            dlosses.update({'train_op': train_op})

        #############
        # Summaries #
        #############

        if mode in ['train', 'evaluate']:
            summary_scope = 'train' if mode == 'train' else 'eval'
            
            with tf.variable_scope('classification_summary/' + summary_scope, reuse=tf.AUTO_REUSE):
                # Classification part
                summary_op, summary_vars = classification_summary(cl_logits, labels, 'accuracy')

                # Validation from middle logits
                middle_logits = batch_gather(cl_logits_all, length, progress=0.5)
                middle_summary_op, middle_summary_vars = \
                    classification_summary(middle_logits, labels, 'middlepoint_accuracy')

                # Validation in top 3
                top3_summary_op, top3_summary_vars = \
                    classification_summary(cl_logits, labels,  'map3_accuracy',
                                           top_k=3, avg_top_k=True)
                cl_summary = tf.summary.merge_all(scope='classification_summary/' + summary_scope)
                dmetrics.update({'classification_summary': cl_summary, 
                                 'classification_accuracy':         summary_vars['accuracy'],
                                 'classification_accuracy_middle':  middle_summary_vars['middlepoint_accuracy'],
                                 'classification_accuracy_map3':    top3_summary_vars['map3_accuracy']})

        #######################
        # Inference Test Data #
        #######################

        if mode == 'predict':
            predictions = tf.nn.top_k(cl_logits, k=3)[1]
            dstates['prediction'] = predictions

        return dstates, dlosses, dmetrics

if __name__ == '__main__':
    model = BLSTM()
    model.train()