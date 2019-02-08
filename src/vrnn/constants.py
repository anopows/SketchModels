# Source: https://github.com/emreaksan/deepwriting/tree/master/source
class Constants(object):
    SEED = 23

    # Run modes.
    TRAIN = 'training'
    TEST = 'test'
    VALID = 'validation'
    EVAL = 'evaluation'
    SAMPLE = 'sampling'

    # RNN cells
    GRU = 'gru'
    LSTM = 'lstm'
    BLSTM = 'blstm'
    LayerNormLSTM = 'LayerNormBasicLSTMCell'

    # Activation functions
    RELU = 'relu'
    ELU = 'elu'
    SIGMOID = 'sigmoid'
    SOFTPLUS = 'softplus'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    LRELU = 'lrelu'
    CLRELU = 'clrelu'  # Clamped leaky relu.

    # Losses
    NLL_BERNOULLI = 'nll_bernoulli'
    NLL_NORMAL = 'nll_normal'
    NLL_BINORMAL = 'nll_binormal'
    NLL_GMM = 'nll_gmm'
    NLL_BIGMM = 'nll_bigmm'
    NLL_CENT = 'nll_cent'  # Cross-entropy.
    KLD = 'kld'
    L1 = 'l1'
    MSE = 'mse'

    # Model output names
    OUT_MU = 'out_mu'
    OUT_SIGMA = 'out_sigma'
    OUT_RHO = 'out_rho'
    OUT_COEFFICIENT = 'out_coefficient'  # For GMM outputs only.
    OUT_BINARY = 'out_binary'  # For binary outputs and bernoulli loss

    # Suffix for output names.
    SUF_MU = '_mu'
    SUF_SIGMA = '_sigma'
    SUF_RHO = '_rho'
    SUF_COEFFICIENT = '_coefficient'  # For GMM outputs only.
    SUF_BINARY = '_binary'  # For binary outputs and bernoulli loss
    SUF_CENT = '_logit'  # For cross-entropy loss

    # Reduce function types
    R_MEAN_STEP = 'mean_step_loss'  # Take average of average step loss per sample over batch. Uses sequence length.
    R_MEAN_SEQUENCE = 'mean_sequence_loss'  # Take average of sequence loss (summation of all steps) over batch. Uses sequence length.
    R_MEAN = 'mean'  # Take mean of the whole tensor.
    R_SUM = 'sum'  # Take mean of the whole tensor.
    B_MEAN_STEP = 'batch_mean_step_loss'  # Keep the loss per sample. Uses sequence length.
    R_IDENTITY = 'identity'

    # Models
    MODEL_RNN = 'rnn'
    MODEL_BIRNN = 'birnn'  # Bidirectional RNN
    MODEL_CCN = 'ccn'  # Causal Convolutional Network (i.e. Wavenet Model)
    MODEL_TCN = 'tcn'  # Temporal Convolutional Network
    MODEL_STCN = 'stcn'  # Stochastic Temporal Convolutional Network
    MODEL_BISTCN = 'bistcn'  # Bidirectional stochastic Temporal Convolutional Network
    MODEL_VRNN = 'vrnn'  # Variational Recurrent Neural Network.
    MODEL_ZFORCE = 'zforce'
    MODEL_BITCN = 'bitcn'
    MODEL_RNN_LATENT_AE = 'rnnlatentae'
    MODEL_RNN_LATENT_CELL = 'rnnlatentcell'
    MODEL_EMBEDDED_LSTCN = 'elstcn'  # Stochastic TCN with ladder latent layer embedded into layers.

    # Motion Datasets
    HOLDEN = 'holden'
    SMPL = 'smpl'
    H36M = 'h36m'
    IMU = 'imu'

    # Digital Ink Datasets
    IAMONDB = "iam"
    DEEPWRITING = "dw"

    # Benchmark datasets
    TIMIT = "timit"

    # Dataset I/O keys for TF placeholders.
    PL_INPUT = "pl_input"
    PL_TARGET = "pl_target"
    PL_SEQ_LEN = "pl_seq_len"
    PL_IDX = "pl_idx"

    PL_ORI = "pl_imu_orientation"
    PL_ACC = "pl_imu_acceleration"
    PL_SMPL = "pl_imu_smpl_pose"

    # Latent components.
    Q_MU = 'q_mu'
    Q_SIGMA = 'q_sigma'
    P_MU = 'p_mu'
    P_SIGMA = 'p_sigma'
    Z_LATENT = 'z_latent'
    Q_PI = 'q_pi'
    P_PI = 'p_pi'
    LATENT_Q = 'q'  # approximate_posterior
    LATENT_P = 'p'  # prior

    # Preprocessing operations.
    PP_SHIFT = "pp_shift"
    PP_ZERO_MEAN_NORM = "pp_zero_mean_normalization"
    PP_ZERO_MEAN_NORM_SEQ = "pp_zero_mean_norm_seq_stats"
    PP_EXPLICIT_ANGLE_AXIS = "pp_explicit_angle_axis"
    PP_NORM_IGNORE_ZEROS = "pp_norm_ignore_zeros"
    PP_IGNORE_ZEROS = "pp_ignore_zeros"
    PP_RELATIVE_STEP = "pp_relative_step"
    PP_RANDOM_WINDOWS = "pp_random_windows"
    PP_QUANTIZE = "pp_quantize"
    PP_SCALE_ZERO_ONE = "pp_scale_zero_one"
    PP_SCALE_MINUS_ONE_ONE = "pp_scale_minus_one_one"

    # Preprocessing operations: IMU
    PP_IMU_ORI_NORM = "norm_ori"
    PP_IMU_ACC_NORM = "norm_acc"
    PP_IMU_SMPL_NORM = "norm_smpl"

    # Preprocessing operator side-effects.
    SE_PP_SEQ_LEN_DIFF = "sequence_length_diff"
    SE_PP_SEQ_LEN = "sequence_length"
    SE_PP_INPUT_SIZE = "input_feature_size"
    SE_PP_TARGET_SIZE = "target_feature_size"

    # Zforcing Cell
    AUX_BACKWARD = "aux_backward"
    SUF_BACKWARD = "_backward"

    # Latent layers.
    LATENT_GAUSSIAN = "latent_gaussian"
    LATENT_STOCHASTIC_CODEBOOK = "latent_stochastic_codebook"
    LATENT_VARIATIONAL_CODEBOOK = "latent_variational_codebook"
    LATENT_CATEGORICAL = "latent_categorical"
    LATENT_LADDER_GAUSSIAN = "latent_ladder_gaussian"

    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"

    LAYER_TCN = "tcn"
    LAYER_FC = "fc"
    LAYER_RNN = "rnn"

    DECAY_PC = "piecewise_constant"
    DECAY_EXP = "exponential_decay"
