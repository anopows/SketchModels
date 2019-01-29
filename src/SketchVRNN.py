import tensorflow as tf
from SketchModels import SketchModel
from sketch_util  import batch_gather
from model_fns    import classification
from loss_fns     import classification_loss, classification_summary
from vrnn.tf_rnn_cells   import GaussianLatentCell
from vrnn.constants      import Constants as C
from vrnn.tf_loss        import logli_normal_isotropic, logli_bernoulli

class VRNN(SketchModel):
    def _init_flags(self):
        super(VRNN, self)._init_flags()
        self.flags.DEFINE_integer('latentsize',       512,   "Latent size z")
        self.flags.DEFINE_boolean('train_together',          True,  "Train classification together with VRNN")
        self.flags.DEFINE_string ('logits_option', 'hidden+latent', 
            "What logits to use for classification: 'hidden', 'latent', 'latent_phi', 'hidden+latent', 'hidden+latent_phi'")
        self.flags.DEFINE_float  ('kldweight',        1.0,   "Weighting of KL Divergence")
        self.flags.DEFINE_boolean('anneal_klw',      True,   "Anneal KL weigthts from 0.1 to kld weight")
        self.flags.DEFINE_float  ('anneal_klw_start', 0.4,   "Start with this weighting of the KL loss")
        self.flags.DEFINE_float  ('anneal_klw_end',   1.0,   "End with this weigthing of the KL loss")
        self.flags.DEFINE_integer('anneal_klw_steps',10000,  "Anneal in KL weighting from start to end in this many steps")
        self.flags.DEFINE_boolean('post_training',   False,  "Finetune classifier at end")
        self.FLAGS['prefix'].value = 'VRNN'

    def _init_model_params(self):
        super(VRNN, self)._init_model_params()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='vrnn_global_step') # for older version prepend 'vrnn_loss/' or 'vrnn/'
        self.model_name    += '_kldw{}'.format(self.FLAGS.kldweight)
        if self.FLAGS.train_together: self.model_name += '_trainTogether'
        if self.FLAGS.anneal_klw:             
            self.model_name += '_annealKL' \
                            + '-from'  + str(self.FLAGS.anneal_klw_start) \
                            + '-end'   + str(self.FLAGS.anneal_klw_end) \
                            + '-steps' + str(self.FLAGS.anneal_klw_steps)
        self.model_name += '_cln-' + self.FLAGS.logits_option
        if self.FLAGS.post_training:
            self.model_name += "_postTrainClassifier"

    @staticmethod
    def _infer(logits, img_logits=None):
        def infer_mu_sig(states, units, scope, reuse=tf.AUTO_REUSE):
            """
            Given input logits parameterizes a Normal distribution
            Args:
                states: hidden states, form (batch_size, sequence_length, dim)
                scope: 
            Returns:
                (mu,sigma)
            """

            with tf.variable_scope(scope, reuse=reuse):
                with tf.variable_scope(scope + '_mu', reuse=reuse):
                    mu_t = tf.layers.dense(inputs=states, units=units)
                with tf.variable_scope(scope + '_sigma', reuse=reuse):
                    sigma_t = tf.layers.dense(inputs=states, units=units, activation=tf.nn.softplus)
                return mu_t, sigma_t
                
        if img_logits is not None:
            logits = tf.concat((logits, img_logits), axis=2)
        # Log likelihood position/movement. 
        with tf.variable_scope('coord', reuse=tf.AUTO_REUSE):
            mu, sig= infer_mu_sig(logits, units=2, scope='coord_inference')
        
        # Log likelihood for pen up event
        with tf.variable_scope('pen_up', reuse=tf.AUTO_REUSE):
            penup_p = tf.layers.dense(logits, units=1, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE)
            
        return mu, sig, penup_p

    @staticmethod
    def _sample_step(mu_coord, sig_coord, penup_p, prob_weight=1, sample_p=False, threshold_penup=0.35):
        # Assume shapes [batch_size x max_length x 2/1] for coord/penup
        mov_reconstructed   = tf.random_normal(tf.shape(mu_coord), mu_coord, prob_weight * sig_coord)
        if sample_p:
            penup_reconstructed = tf.distributions.Bernoulli(probs=penup_p).sample()
            penup_reconstructed = tf.cast(penup_reconstructed, tf.float32)
        else: # everything over threshold: take
            penup_reconstructed = penup_p > threshold_penup
            penup_reconstructed = tf.cast(penup_reconstructed, tf.float32)
        return tf.concat([mov_reconstructed, penup_reconstructed], axis=2)

    def _vrnn(self, sketch, mode, use_imgs=False):
        with tf.variable_scope('vrnn', reuse=tf.AUTO_REUSE):  
            kld_weight = self.FLAGS.kldweight
            # kl annealing
            if self.FLAGS.anneal_klw:
                # start, end, increment eg: [0.1, 1, 9e-5] --> 10k steps
                start, end = [self.FLAGS.anneal_klw_start, self.FLAGS.anneal_klw_end]
                increment = (end-start) / self.FLAGS.anneal_klw_steps
                kld_weight = tf.minimum(tf.cast(self.global_step, tf.float32)*increment + start, end)
            self.config = {
                'use_temporal_kld': False,              # False for regular KL-Divergence
                'latent_size': self.FLAGS.latentsize,   # Size of z
                'kld_weight' : kld_weight,
                'num_hidden_units': self.FLAGS.latentsize,  # Phi function, feature extractor
                'num_hidden_layers': 2,                     # Phi function, layers
                # LSTM part
                'cell_type': C.LSTM,
                'cell_num_layers': self.FLAGS.lstm_stack,   # Stacked 2 times
                'cell_size': self.FLAGS.hidden_size,
                #
                'hidden_activation_fn': tf.nn.relu,
                # img option
                'use_imgs': use_imgs
            } 

            # Setup sketch input: Add 0 token at start: To inference first token
            batch_size   = tf.shape(sketch)[0] 
            feature_size = tf.shape(sketch)[2]
            zero_token = tf.zeros((batch_size, 1, feature_size), dtype=tf.float32)
            sketch     = tf.cast(sketch, tf.float32)
            sketch_input = tf.concat((zero_token, sketch), axis=1)
            # Setup Cell
            vrnn_cell = GaussianLatentCell(config=self.config, mode=mode, reuse=tf.AUTO_REUSE)
            # init states
            init_state = vrnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32) 

            with tf.variable_scope('dynamic_rnn', reuse=tf.AUTO_REUSE): 
                # Setup VRNN - Training
                (*pq, states, z_t, z_t_phi) , last_state = \
                    tf.nn.dynamic_rnn(vrnn_cell, sketch_input, initial_state=init_state)
                vrnn_cell.register_sequence_components(pq)
            
            return pq, states, z_t, z_t_phi, vrnn_cell, last_state


    def model_fn(self, data, mode=None): # data: dict with sketch and length. mode: 'train'/'evaluate'/'predict'
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

        _mode = 'training' # Names for mode for VRNN code
        if   mode == 'evaluate': _mode = 'validation'
        elif mode == 'predict':  _mode = 'validation' # use validation mode to calculate state

        sketch = data['sketch_prepr']
        length = data['length']
        if mode != 'predict':
            labels = data['labels']
        sketch_img_logits = data.get('sketch_img_logits', None)
        if self.FLAGS.use_imgs:
            sketch = tf.concat([sketch, sketch_img_logits], axis=2) # bs x maxlen x (3+img_logit_size)

        # Dicts to return
        dstates  = {}
        dlosses  = {}
        dmetrics = {}

        def _average(values, length, max_seqlen=None):
            values = tf.squeeze(values)
            mask   = tf.cast(tf.sequence_mask(length, maxlen=self.max_seqlen), tf.float32)
            values = mask * values
            return tf.reduce_sum(values) / tf.reduce_sum(mask)

        ########
        # VRNN #
        ########
        
        pq, states, z_t, z_t_phi, vrnn, _ = \
            self._vrnn(sketch, mode=_mode, use_imgs=self.FLAGS.use_imgs)
        dstates.update({'pq': pq, 'states': states, 'z_t': z_t, 'z_t_phi':z_t_phi, 'vrnn':vrnn})

        with tf.variable_scope('vrnn', reuse=tf.AUTO_REUSE):
            # Concat. Ignore first entries: 0 token input/init state
            logits_hz       = tf.concat([states[:,1:], z_t_phi[:,1:]], axis=2)
            # Logits -> (mu,sig),p
            mu, sig, penup_p = VRNN._infer(logits_hz, img_logits=sketch_img_logits)
            dstates.update({'logits_hz': logits_hz, 'mu': mu, 'sig': sig})
    
            # Log likelihood position/movement. 
            with tf.variable_scope('coord', reuse=tf.AUTO_REUSE):
                logli_coord = logli_normal_isotropic(sketch[:,:,:2], mu, sig)
            
            # Log likelihood for pen up event
            with tf.variable_scope('pen_up', reuse=tf.AUTO_REUSE):
                logli_penup = logli_bernoulli(sketch[:,:,2:3], penup_p)
            
            # KL Loss
            with tf.variable_scope('KL'):
                kl_loss = vrnn.build_loss(sequence_mask=1, 
                                          reduce_loss_fn=lambda x:x,
                                          loss_ops_dict=dict())
                kl_loss = kl_loss['loss_kld'][:,1:] # first kl loss term not needed
                
            dmetrics.update({'penup_p': penup_p, 'logli_coord': logli_coord, 'logli_penup': logli_penup, 'kl_loss': kl_loss})
            dmetrics.update({'avg_penup_p': _average(penup_p, length, max_seqlen=self.max_seqlen), 
                             'avg_logli_coord': _average(logli_coord, length, max_seqlen=self.max_seqlen), 
                             'avg_logli_penup': _average(logli_penup, length, max_seqlen=self.max_seqlen), 
                             'avg_kl_loss': _average(kl_loss, length, max_seqlen=self.max_seqlen)})

        ##################
        # Classification #
        ##################
        
        classification_fn =     lambda logits : classification(logits, layer_sizes=[64,self.num_classes], 
                                           bnorm_before=self.FLAGS.bnorm_before, bnorm_middle=self.FLAGS.bnorm_middle, # if: apply after first dense layer
                                           training=(mode=='training' or self.FLAGS.bnorm_eval_update), trainable=True, name='classifier')
        
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            if self.FLAGS.logits_option == 'hidden':
                unit_size = self.FLAGS.hidden_size
                logits_for_cl = states[:,1:]
            elif self.FLAGS.logits_option == 'latent':
                unit_size = self.FLAGS.latentsize
                logits_for_cl = z_t[:,1:]
            elif self.FLAGS.logits_option == 'latent_phi':
                unit_size = self.FLAGS.latentsize
                logits_for_cl = z_t_phi[:,1:]
            elif self.FLAGS.logits_option == 'hidden+latent':
                unit_size = self.FLAGS.hidden_size + self.FLAGS.latentsize
                logits_for_cl = tf.concat([states[:,1:], z_t[:,1:]], axis=2)
            elif self.FLAGS.logits_option == 'hidden+latent_phi':
                unit_size = self.FLAGS.hidden_size + self.FLAGS.latentsize
                logits_for_cl = tf.concat([states[:,1:], z_t_phi[:,1:]], axis=2)
            else: raise NotImplementedError
            
            # [batch_size x max_len x unit_size] --> [batch_size*max_len x unit_size]
            cl_logits_all = tf.reshape(logits_for_cl, [-1, unit_size])
            dstates['logits_for_cl'] = batch_gather(cl_logits_all, length)
            cl_logits_all  = classification_fn(cl_logits_all)
            # turn back to batches
            bs  = tf.shape(z_t)[0]
            cl_logits_all  = tf.reshape(cl_logits_all, [bs, -1, self.num_classes])
            cl_logits = batch_gather(cl_logits_all, length)
            dstates['cl_logits'] = cl_logits
        
        ########
        # Loss #
        ########

        if mode in ['train', 'evaluate']:
            with tf.variable_scope('classification_loss', reuse=tf.AUTO_REUSE):
                # Train on last states
                *cl_vals, = classification_loss(cl_logits, labels, scope_to_train='logits', 
                                            training=True, clip_gradient=self.FLAGS.clip_gradient,
                                            learning_rate=self.FLAGS.learning_rate, anneal_lr_every=None,
                                            mask=None, additional_loss=None, 
                                            batch_normalization=self.FLAGS.bnorm_before or self.FLAGS.bnorm_middle)
                cl_loss, cl_avg_loss = (cl_vals[0], cl_vals[1])

            with tf.variable_scope('vrnn_loss', reuse=tf.AUTO_REUSE):
                # loss part
                train_mask = tf.cast(tf.sequence_mask(length, maxlen=self.max_seqlen), tf.float32)
                vrnn_loss = -logli_coord - logli_penup + kl_loss
                vrnn_loss = train_mask * tf.squeeze(vrnn_loss)

                avg_vrnn_loss = tf.reduce_sum(vrnn_loss) / tf.reduce_sum(train_mask)

                if self.FLAGS.train_together:
                    avg_loss = avg_vrnn_loss + cl_avg_loss 
                else: 
                    avg_loss = avg_vrnn_loss

            dlosses.update({'vrnn_loss': avg_vrnn_loss, 'cl_loss': cl_avg_loss, 
                            'vrnn_cl_loss': avg_vrnn_loss + cl_avg_loss, 'train_loss': avg_loss})
            summary_scope = 'train' if mode == 'train' else 'eval'
            dmetrics.update({'train_loss': tf.summary.scalar(summary_scope + '/train_loss', avg_loss),
                             'vrnn_loss':  tf.summary.scalar(summary_scope + '/vrnn_loss',  avg_vrnn_loss),
                             'cl_loss':    tf.summary.scalar(summary_scope + '/classification_loss', cl_avg_loss)})

        ############
        # Training #
        ############

        if mode == 'train':
            self.cl_global_step, cl_train_op = (cl_vals[2], cl_vals[3]) # unpack from classification loss calculations
            with tf.variable_scope('vrnn_loss', reuse=tf.AUTO_REUSE):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.train_learning_rate)
                train_vars = None # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vrnn')
                gradients = optimizer.compute_gradients(avg_loss, var_list=train_vars)
                gradients = [(tf.clip_by_value(grad, -1*self.FLAGS.clip_gradient, self.FLAGS.clip_gradient), var) 
                                    for grad, var in gradients if grad is not None]
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # case of batch norm used
                with tf.control_dependencies(update_ops):
                    vrnn_train_op = optimizer.apply_gradients(
                        grads_and_vars=gradients, global_step=self.global_step
                    )
            dlosses.update({'train_op': vrnn_train_op, 'cl_train_op': cl_train_op})

        #############
        # Summaries #
        #############

        if mode in ['train', 'evaluate']:
            summary_scope = 'train' if mode == 'train' else 'eval'
            
            # VRNN part
            with tf.variable_scope('vrnn_summary/' + summary_scope, reuse=tf.AUTO_REUSE):
                summary_loss = tf.summary.scalar('loss', avg_loss)
                # loss (- log-likelihood + kl-loss)
                summary_loss = tf.summary.scalar('vrnn_loss', avg_vrnn_loss)
                # loglikelihood
                avg_loglikelihood = _average(logli_coord + logli_penup, length, max_seqlen=self.max_seqlen)
                summary_logli = tf.summary.scalar('vrnn_loglikelihood', avg_loglikelihood)
                # loglikelihood coord
                avg_loglikelihood_coord = _average(logli_coord, length, max_seqlen=self.max_seqlen)
                summary_logli_coord = tf.summary.scalar('vrnn_loglikelihood_coord', avg_loglikelihood_coord)
                # loglikelihood penup
                avg_loglikelihood_penup = _average(logli_penup, length, max_seqlen=self.max_seqlen)
                summary_logli_penup = tf.summary.scalar('vrnn_loglikelihood_penup', avg_loglikelihood_penup)
                # kl loss
                avg_kl = _average(kl_loss, length, max_seqlen=self.max_seqlen)
                summary_kl = tf.summary.scalar('vrnn_KLloss', avg_kl)

                vrnn_summary = tf.summary.merge_all(scope='vrnn_summary/' + summary_scope)
                dmetrics.update({'vrnn_summary': vrnn_summary})

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

    def _summary_step(self, i, sess, writer, log_in_tb):
        summary_ops = [self.ev_metrics['vrnn_summary'], 
                       self.tr_metrics['train_loss'],
                       self.ev_metrics['train_loss'],
                       self.tr_metrics['vrnn_loss'],
                       self.ev_metrics['vrnn_loss'],
                       self.tr_metrics['cl_loss'],
                       self.ev_metrics['cl_loss'],
                       self.tr_losses['train_loss'], 
                       self.ev_losses['train_loss']]
        vrnn_summary_out, \
            tr_train_loss_summary, ev_train_loss_summary, \
            tr_vrnn_loss_summary,  ev_vrnn_loss_summary, \
            tr_cl_loss_summary,    ev_cl_loss_summary, \
            avg_vrnn_loss_out, avg_eval_vrnn_loss_out = \
                sess.run(summary_ops)

        avg_kl_train_out, avg_loglikelihood_coord_train_out, avg_loglikelihood_penup_train_out = \
            sess.run([self.tr_metrics['avg_kl_loss'], self.tr_metrics['avg_logli_coord'], self.tr_metrics['avg_logli_penup']])
        avg_kl_eval_out, avg_loglikelihood_coord_eval_out, avg_loglikelihood_penup_eval_out = \
            sess.run([self.ev_metrics['avg_kl_loss'], self.ev_metrics['avg_logli_coord'], self.ev_metrics['avg_logli_penup']])
        self.logging("Step {:5d}. Train loss: {:5.2f} | Valid. loss: {:5.2f}   ||   ".format(
                    i+1, avg_vrnn_loss_out, avg_eval_vrnn_loss_out) + \
                "Train KL Loss: {:5.2f}. Train Logli coord/penup: {:5.2f}/{:5.2f} | ".format(
                    avg_kl_train_out, avg_loglikelihood_coord_train_out, avg_loglikelihood_penup_train_out) + \
                "Valid. KL Loss: {:5.2f}. Valid. Logli coord/penup: {:5.2f}/{:5.2f}".format(
                    avg_kl_eval_out, avg_loglikelihood_coord_eval_out, avg_loglikelihood_penup_eval_out
                )
        )
        if log_in_tb: 
            writer.add_summary(vrnn_summary_out, global_step=i)
            writer.add_summary(tr_train_loss_summary, global_step=i)
            writer.add_summary(ev_train_loss_summary,  global_step=i)
            writer.add_summary(tr_vrnn_loss_summary, global_step=i)
            writer.add_summary(ev_vrnn_loss_summary, global_step=i)
            writer.add_summary(tr_cl_loss_summary, global_step=i)
            writer.add_summary(ev_cl_loss_summary, global_step=i)
    
        if self.FLAGS.train_together:
            cl_summary_ops = [self.ev_metrics['classification_summary'],
                              self.ev_metrics['classification_accuracy'],
                              self.ev_metrics['classification_accuracy_map3']]
            cl_summary_out, cl_la_acc_out, cl_top3_acc_out = \
                sess.run(cl_summary_ops)
            self.logging("\t\tClassification accuracy: {:7.2f}. Top 3 accuracy: {:7.2f}".format(cl_la_acc_out, cl_top3_acc_out))
            if log_in_tb:
                writer.add_summary(cl_summary_out,      global_step=i)

    def post_train(self, sess):
        if self.FLAGS.post_training:
            self.train_classifier(sess)
            self.validate(sess)

    def train_classifier(self, sess, num_steps_classification=5000):
        self.logging("Training classifier")
        
        for i in range(num_steps_classification):
            _, eval_cl_acc, eval_cl_map3 = \
                sess.run([self.tr_losses['cl_train_op'], 
                self.ev_metrics['classification_accuracy'], 
                self.ev_metrics['classification_accuracy_map3']])
            if i<20 or (i+1)%20 == 0:
                self.logging("Classification Training step {:4d}. Validation accuracy: {:6.4f}, map3: {:6.4f}".format(
                             i+1, eval_cl_acc, eval_cl_map3))


    def psample_after_half(self, length=25):
        def state_after_half(length, sketch):
            half_length = (length+1) // 2
            sketch = sketch[:half_length]
            *_, output_state = self._vrnn(sketch[None], mode="validation")
            return output_state

        def merge_states(states):
            c0s = []
            h0s = []
            c1s = []
            h1s = []
            z_s = []
            for state in states:
                c0s.append(state[0][0].c)
                h0s.append(state[0][0].h)
                c1s.append(state[0][1].c)
                h1s.append(state[0][1].h)
                z_s.append(state[1])
            
            c0 = tf.concat(c0s, axis=0)
            h0 = tf.concat(h0s, axis=0)
            c1 = tf.concat(c1s, axis=0)
            h1 = tf.concat(h1s, axis=0)
            z  = tf.concat(z_s, axis=0)

            return (tf.contrib.rnn.LSTMStateTuple(c0, h0), tf.contrib.rnn.LSTMStateTuple(c1, h1)), z
        lengths  = tf.unstack(self.ev_data['length'],       num=self.FLAGS.batch_size)
        sketches = tf.unstack(self.ev_data['sketch_prepr'], num=self.FLAGS.batch_size)
        
        # calculate states
        states = []
        for l,sk in zip(lengths, sketches): states.append(state_after_half(l,sk))
        # merge states together into one batch
        # every state: ((c0,h0), (c1,h1)), z
        state = merge_states(states)
        return self.ev_data['length'], self.ev_data['labels'], self.ev_data['sketch'], self.psample(prev_state=state, length=length)


    def psample(self, prev_state=None, length=None, keepfirst=False, threshold_penup=0.35):
        with tf.variable_scope('vrnn', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('dynamic_rnn/rnn', reuse=tf.AUTO_REUSE):
                vrnn_sampling_cell  = GaussianLatentCell(config=self.config, mode="sampling", reuse=tf.AUTO_REUSE)
                # init state sampling, make already a step for z
                if prev_state is None:
                    prev_state = vrnn_sampling_cell.zero_state(batch_size=self.FLAGS.batch_size, dtype=tf.float32)
                sampling_init_state = vrnn_sampling_cell.vae_step(prev_state) # update z state
                _, sampling_h, sampling_z = sampling_init_state
                sampling_h = tf.expand_dims(sampling_h, axis=1) 
                sampling_z = tf.expand_dims(sampling_z, axis=1)

            # Concat. Ignore first entries: 0 token input/init state
            sampling_logits = tf.concat([sampling_h, sampling_z], axis=2)
            mu0, sig0, p0 = VRNN._infer(sampling_logits)
            x0            = VRNN._sample_step(mu0, sig0, p0, threshold_penup=threshold_penup)

            xs = [x0[:,0,:]]
            last_state = sampling_init_state
            length = 50 if not length else length
            for _ in range(length):
                with tf.variable_scope('dynamic_rnn/rnn', reuse=tf.AUTO_REUSE):
                    (*_, h_t, z_t, z_t_phi), state = vrnn_sampling_cell(xs[-1], last_state)
                h_t, z_t_phi = (tf.expand_dims(h_t, axis=1), tf.expand_dims(z_t_phi, axis=1))
                logits_t = tf.concat([h_t, z_t_phi], axis=2)
                mu_t, sig_t, p_t = VRNN._infer(logits_t)
                x_t              = VRNN._sample_step(mu_t, sig_t, p_t, threshold_penup=threshold_penup)
                xs.append(x_t[:,0,:])
                last_state = state
            
            xs = tf.stack(xs, axis=1)
            if keepfirst:   xs = xs[:, :-1]
            else:           xs = xs[:, 1:] 

            lengths = length * tf.ones(shape=[self.FLAGS.batch_size], dtype=tf.int32)
            xs = self._revert_normalization(xs, lengths)

            return xs, lengths

    def qsample(self, prob_weight=1):
        # reconstruct sketch: [batch_size x max_length x 3]
        qsketch = VRNN._sample_step(self.ev_states['mu'], self.ev_states['sig'], self.ev_metrics['penup_p'], prob_weight)
        # mask only until real data's progress
        qsketch = qsketch* tf.cast(tf.sequence_mask(self.ev_data['length'], maxlen=self.FLAGS.max_seqlen), tf.float32)[:,:,None]
        qsketch    = self._revert_normalization(qsketch, self.ev_data['length'])
        return self.ev_data['labels'], self.ev_data['length'], qsketch, self.ev_data['sketch']  

        
if __name__ == '__main__':
    vrnn = VRNN()
    vrnn.train()
