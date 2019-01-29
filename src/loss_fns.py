import tensorflow as tf

def classification_loss(logits, labels, name='classifier_loss', scope_to_train='classifier/', training=True,
                        learning_rate=1e-3, anneal_lr_every=None,
                        batch_normalization=False, clip_gradient=None, mask=None,
                        additional_loss=None,
                        return_gradients=False):
    """Classification from logits and sparse labels
    
    Args:
        logits:         Batch of logits to classify
        labels:         Batch of labels
        name:           Current scope
        scope_to_train: What weights should be trained
        training:       If yes: returning also global_step & train_op
        clip_gradient
        lengths:        If wanting to apply classification loss, only to subset of values
        clip_value
    Returns:
        loss
        avg_loss
        (if training)global_step
        (if training)train_op
    """
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        
        if additional_loss is not None:
            assert loss.get_shape().as_list() == additional_loss.get_shape().as_list(), "Same shapes for comparison"
            loss = (loss + additional_loss)/2

        if mask is not None:
            loss = loss * mask
            avg_loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        else:
            avg_loss = tf.reduce_mean(loss)

        if not training:
            return loss, avg_loss
        
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='classification_global_step')

        if anneal_lr_every is not None:
            learning_rate = tf.train.exponential_decay(
                   learning_rate, global_step, anneal_lr_every, 0.5, staircase=True
               )
            
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # only apply gradients to classifier weights
        if scope_to_train:
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_to_train)
        else:
            train_vars = None
        
        gradients = optimizer.compute_gradients(avg_loss, var_list=train_vars)
        if clip_gradient:
            gradients = [(tf.clip_by_value(grad, -1*clip_gradient, clip_gradient), var) 
                         for grad, var in gradients if grad is not None]
        

        if batch_normalization: # train_op depdendent on updating batch normalization params
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            assert len(update_ops) == 2 # Mean and variance
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(
                    grads_and_vars=gradients, global_step=global_step
                )
        else:
            train_op = optimizer.apply_gradients(
                    grads_and_vars=gradients, global_step=global_step
            )
        
        if return_gradients:
            return loss, avg_loss, global_step, train_op, gradients
        else:
            return loss, avg_loss, global_step, train_op
        
def mapk_acc(labels, predictions, k=3):
    bs = tf.shape(predictions)[0]
    es = tf.shape(predictions)[1]

    predictions = tf.cast(predictions, tf.int32)
    labels      = tf.cast(labels,      tf.int32) 
    equal_els = tf.cast(tf.equal(predictions, labels[:,None]), tf.float32)
    weighting = tf.range(1, tf.cast(es+1, tf.float32), dtype=tf.float32)
    weighting = tf.tile(weighting[None], [bs, 1])
    
    max_eq_els = tf.reduce_max(equal_els / weighting, axis=1)
    acc  = tf.reduce_mean(max_eq_els)
    return acc
    
def classification_summary(logits, labels, name=None, top_k=1, avg_top_k=True, **to_log):
    """ Creates summaries for Tensorboard based on classification logits and labels
    Args: 
        logits:      Batch of logits
        labels:      Label ground truth
        name:        Scope name of summary
    Returns:
        summary:     Summary
        var_dict:    Calculated metrics
    """
    var_dict = {}
    if not name:
        name = 'classification_accuracy'
        if top_k > 1: 
            if avg_top_k: 
                name += '_avgtop' + str(top_k)
            else:
                name += '_top' + str(top_k)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):   
        if avg_top_k and top_k>1:
            classification_accuracy = mapk_acc(labels, tf.nn.top_k(logits, k=top_k)[1])
        else:
            correct_preds = tf.nn.in_top_k(logits, labels, top_k)
            classification_accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

        tf.summary.scalar(name, classification_accuracy)
        var_dict[name] = classification_accuracy
        
        # Additonal tensors to log
        for tname, tensor in to_log.items():
            tf.summary.histogram(tname, tensor)
            
        return tf.summary.merge_all(scope=name), var_dict