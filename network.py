from keras import backend as K
from keras.layers import CuDNNGRU,Bidirectional,Lambda
from keras.losses import mse, binary_crossentropy
from sklearn.metrics import hamming_loss
def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D 
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"])(layer)
    return layer


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer, **params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add 
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer

def add_GCN(K_CPT,K_X):
    from keras.layers import Input, Dropout, Lambda
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.regularizers import l2

    from graph import GraphConvolution
    import scipy.sparse as sparse
    # from ecg.graph_utils import *

    # K_CPT = Lambda(K.constant)(CPT)
    # K_X = Lambda(K.constant)(diag_matrix)

    support = 1
    # graph = [K_X, K_CPT ]
    # K_CPT = Input(shape=(None, None), batch_shape=(None, None), sparse=True)

    # K_X= Input(shape=(diag_matrix.shape[1],))

    # Define model architecture
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
    H = Dropout(0.5)(K_X)
    H = GraphConvolution(64, support, activation='relu', kernel_regularizer=l2(5e-4))([H] + [K_CPT])
    H = Dropout(0.5)(H)
    layer = GraphConvolution(128, support, activation='softmax')([H] + [K_CPT])
    return layer

def layer_dot(layers):
    return K.dot(layers[0], layers[1])

def add_output_layer(layer, GCN_layer,**params):
    from keras.layers.core import Dense, Activation
    from keras.layers import Multiply,Dot,Lambda
    from keras.layers.wrappers import TimeDistributed
    layer = Bidirectional(CuDNNGRU(64,  return_sequences=True, return_state=False))(layer)
    GCN_layer = Lambda(K.transpose)(GCN_layer)
    # layer = K.dot(layer,GCN_layer)
    layer = Lambda(layer_dot)([layer, GCN_layer])
    # layer = Dot()([layer,GCN_layer])
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    layer = Activation('sigmoid')(layer)
    return layer

def hn_multilabel_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def add_compile(model, **params):
    from keras.optimizers import Adam
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss=hn_multilabel_loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

def pre_process(x,mean,std):
    # mean = params['mean']
    # std = params['std']
    x = (x - mean) / std
    return x

def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    processed_inputs = Lambda(pre_process,arguments={'mean':params['mean'],'std':params['std']})(inputs)
    layer = add_resnet_layers(processed_inputs, **params)
    CPT = params['CPT']
    # CPT = sparse.csr_matrix(CPT)
    diag_matrix = params['diag_matrix']
    cons_CPT = K.constant(CPT)
    cons_X = K.constant(diag_matrix)
    input_CPT = Input(tensor=cons_CPT,name='wefew')
    input_X = Input(tensor=cons_X,name='ef2')
    GCN_layer = add_GCN(input_CPT,input_X)
    output = add_output_layer(layer,GCN_layer, **params)
    model = Model(inputs=[inputs,input_CPT,input_X], outputs=[output])
    if params.get("compile", True):
        add_compile(model, **params)
    return model
