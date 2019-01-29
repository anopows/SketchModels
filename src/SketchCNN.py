import tensorflow as tf

from SketchModels import SketchModel
from model_fns import classification
from loss_fns  import classification_loss, classification_summary

class SketchCNN(SketchModel):
    def __init__(self, init_flags=True, my_flags=None):
        super(SketchCNN, self).__init__(init_flags=init_flags, my_flags=my_flags)

    def _init_flags(self, cnn_only=True):
        super(SketchCNN, self)._init_flags(cnn_only=True)

        self.flags.DEFINE_integer('batch_size', 30, "Batch size")
        self.FLAGS['use_imgs'].value   = True
        self.FLAGS['prefix'].value     = 'CNN'

    def _init_model_params(self):
        super()._init_model_params()
        self.model_name = self.FLAGS.prefix + '_imglogits' + str(self.FLAGS.cnn_size)

    def model_fn(self, data, mode=None):
        """
        Args:
            data:   dict with attribute sketch,length and labels
            mode:   'train'/'evaluate'/'predict'
        
        Returns:
            dict of states
            dict of losses (+train_op)
            dict of metrics
        
        """
        assert mode in ['train', 'evaluate', 'predict']
        assert mode != 'predict' or not self.FLAGS.small

        # Dicts to return
        dstates  = {}
        dlosses  = {}
        dmetrics = {}

        # Already have code for imgs --> logits
        logits = data['last_img_logits']
        if mode != 'predict':
            labels = data['labels']

        ##################
        # Classification #
        ##################
        
        classification_fn =     lambda logits : classification(logits, layer_sizes=[64,self.num_classes], 
                                           bnorm_before=False, bnorm_middle=False, # if: apply after first dense layer
                                           training=(mode=='training'), trainable=True, name='classifier')
        
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            dstates['logits_for_cl'] = logits
            cl_logits = classification_fn(logits)
            dstates['cl_logits'] = cl_logits
        
        ########
        # Loss #
        ########

        if mode in ['train', 'evaluate']:
            with tf.variable_scope('conv_model/classification_loss', reuse=tf.AUTO_REUSE):
                # Train on last states
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=cl_logits)
                avg_loss = tf.reduce_mean(loss)
                dlosses.update({'cl_loss': avg_loss, 'train_loss': avg_loss})
                summary_scope = 'train' if mode == 'train' else 'eval'
                dmetrics.update({'train_loss': tf.summary.scalar(summary_scope + '/train_loss', avg_loss),
                                 'cl_loss':    tf.summary.scalar(summary_scope + '/classification_loss', avg_loss)})

        ############
        # Training #
        ############

        if mode == 'train':
            with tf.variable_scope('conv_model/loss', reuse=tf.AUTO_REUSE):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.train_learning_rate)
                train_vars = None
                gradients = optimizer.compute_gradients(avg_loss, var_list=train_vars)
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
            
            with tf.variable_scope('conv_model/summary/' + summary_scope, reuse=tf.AUTO_REUSE):
                # Classification part
                summary_op, summary_vars = classification_summary(cl_logits, labels, 'accuracy')

                # Validation in top 3
                top3_summary_op, top3_summary_vars = \
                    classification_summary(cl_logits, labels, 'map3_accuracy',
                                           top_k=3, avg_top_k=True)
                cl_summary = tf.summary.merge_all(scope='classification_summary/' + summary_scope)
                dmetrics.update({'classification_summary': cl_summary, 
                                 'classification_accuracy':         summary_vars['accuracy'],
                                 'classification_accuracy_map3':    top3_summary_vars['map3_accuracy']})

        #######################
        # Inference Test Data #
        #######################

        if mode == 'predict':
            predictions = tf.nn.top_k(cl_logits, k=3)[1]
            dstates['prediction'] = predictions

        return dstates, dlosses, dmetrics

if __name__ == '__main__':
    model = SketchCNN()
    model.train()