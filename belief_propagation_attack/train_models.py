import os.path
import sys
import h5py
import numpy as np
import argparse
import timing
from time import time
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, LSTM, Dropout
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
try:
    from keras.applications.imagenet_utils import _obtain_input_shape
except ImportError:
    from keras_applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
from utility import *

###########################################################################

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

#### MLP Best model (6 layers of 200 units)
def mlp_best(mlp_nodes=200,layer_nb=6, input_length=700):
    model = Sequential()
    model.add(Dense(mlp_nodes, input_dim=input_length, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(mlp_nodes, activation='relu'))
    model.add(Dense(256, activation='softmax'))
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

### CNN Best model
def cnn_best(classes=256, input_length=700):
    # From VGG16 design
    input_shape = (input_length, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### CNN Previously Trained model
def cnn_pretrained(classes=256, input_length=700):
    # load model
    cnn_previous = load_model(CNN_ASCAD_FILEPATH)
    for layer in cnn_previous.layers[:-6]:
        layer.trainable = False

    model = Sequential()
    model.add(cnn_previous)
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

### LSTM Best model
def lstm_best(classes=256, input_length=700, layer_nb=1, lstm_nodes=64, use_dropout=True):
    # From VGG16 design
    input_shape = (input_length, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    if layer_nb == 1:
        x = LSTM(lstm_nodes)(img_input)

    else:
        x = LSTM(lstm_nodes, return_sequences=True)(img_input)
        for i in range(2, layer_nb):

            x = LSTM(lstm_nodes, return_sequences=True)(x)
        x = LSTM(lstm_nodes)(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lstm_best')
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def load_sca_model(model_file):
    check_file_exists(model_file)
    try:
            model = load_model(model_file)
    except:
        print("Error: can't load Keras model file '%s'" % model_file)
        sys.exit(-1)
    return model

#### Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, validation_data=None, progress_bar=1):
    check_file_exists(os.path.dirname(save_file_name))
    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    callbacks=[save_model, TrainValTensorBoard(write_graph=False)]
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
        Reshaped_validation_data = validation_data[0]
    elif len(input_layer_shape) == 3:
        # This is a CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        Reshaped_validation_data = validation_data[0].reshape((validation_data[0].shape[0], validation_data[0].shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)

    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), batch_size=batch_size, verbose = progress_bar, epochs=epochs, callbacks=callbacks, validation_data=(Reshaped_validation_data, to_categorical(validation_data[1], num_classes=256)))
    return history

# def train_svm()


def train_variable_model(variable, X_profiling, Y_profiling, X_attack, Y_attack, mlp=True, cnn=True, cnn_pre=False, lstm=True, svm=False, add_noise=False, input_length=700, normalise_traces=True, epochs=None, training_traces=50000, mlp_layers=6, lstm_layers=1, batch_size=200, sd=100, augment_method=0, progress_bar=1, mlp_nodes=200, lstm_nodes=64):

    if add_noise:
        standard_deviation = 10
        seed = 1
        np.random.seed(seed)
        X_profiling = X_profiling + np.round(np.random.normal(0, standard_deviation, X_profiling.shape)).astype(int)
        X_attack = X_attack + np.round(np.random.normal(0, standard_deviation, X_attack.shape)).astype(int)

    ### CNN training
    if cnn:
        cnn_best_model = cnn_best(input_length=input_length)
        cnn_epochs = epochs if epochs is not None else 75
        cnn_batchsize = batch_size
        train_model(X_profiling, Y_profiling, cnn_best_model, MODEL_FOLDER +
                    "{}_cnn_window{}_epochs{}_batchsize{}_sd{}_traces{}_aug{}.h5".format(
                        variable, input_length, cnn_epochs, cnn_batchsize, sd, training_traces, augment_method),
                    epochs=cnn_epochs, batch_size=cnn_batchsize, validation_data=(X_attack, Y_attack),
                    progress_bar=progress_bar)

    ### CNN pre-trained training
    if cnn_pre:
        cnn_pretrained_model = cnn_pretrained(input_length=input_length)
        cnn_epochs = epochs if epochs is not None else 75
        cnn_batchsize = batch_size
        train_model(X_profiling, Y_profiling, cnn_pretrained_model, MODEL_FOLDER +
                    "{}_cnnpretrained_window{}_epochs{}_batchsize{}_sd{}_traces{}_aug{}.h5".format(
                        variable, input_length, cnn_epochs, cnn_batchsize, sd, training_traces, augment_method),
                    epochs=cnn_epochs, batch_size=cnn_batchsize, validation_data=(X_attack, Y_attack),
                    progress_bar=progress_bar)

    ### MLP training
    if mlp:
        mlp_best_model = mlp_best(input_length=input_length, layer_nb=mlp_layers)
        mlp_epochs = epochs if epochs is not None else 200
        mlp_batchsize = batch_size
        train_model(X_profiling, Y_profiling, mlp_best_model, MODEL_FOLDER +
                    "{}_mlp{}_nodes{}_window{}_epochs{}_batchsize{}_sd{}_traces{}_aug{}.h5".format(
                        variable, mlp_layers, mlp_nodes, input_length, mlp_epochs, mlp_batchsize, sd,
                        training_traces, augment_method), epochs=mlp_epochs, batch_size=mlp_batchsize,
                    validation_data=(X_attack, Y_attack), progress_bar=progress_bar)

    ### LSTM training
    if lstm:
        lstm_best_model = lstm_best(input_length=input_length, layer_nb=lstm_layers)
        lstm_epochs = epochs if epochs is not None else 75
        lstm_batchsize = batch_size
        train_model(X_profiling, Y_profiling, lstm_best_model, MODEL_FOLDER +
                    "{}_lstm{}_nodes{}_window{}_epochs{}_batchsize{}_sd{}_traces{}_aug{}.h5".format(
                        variable, lstm_layers, lstm_nodes, input_length, lstm_epochs, lstm_batchsize, sd,
                        training_traces, augment_method), epochs=lstm_epochs, batch_size=lstm_batchsize,
                    validation_data=(X_attack, Y_attack), progress_bar=progress_bar)

    ### SVM training
    if svm:
        svm_best_model = svm_best(input_length=input_length, layer_nb=svm_layers)
        svm_epochs = epochs if epochs is not None else 75
        svm_batchsize = batch_size
        train_model(X_profiling, Y_profiling, svm_best_model, MODEL_FOLDER +
                    "{}_svm{}_nodes{}_window{}_epochs{}_batchsize{}_sd{}_traces{}_aug{}.h5".format(
                        variable, svm_layers, svm_nodes, input_length, svm_epochs, svm_batchsize, sd,
                        training_traces, augment_method), epochs=svm_epochs, batch_size=svm_batchsize,
                    validation_data=(X_attack, Y_attack), progress_bar=progress_bar)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--MLP', action="store_true", dest="USE_MLP", help='Trains Multi Layer Perceptron',
                        default=False)
    parser.add_argument('--NORM', action="store_false", dest="NORMALISE", help='Toggles Normalisation Off',
                        default=True)
    parser.add_argument('--CNN', action="store_true", dest="USE_CNN",
                        help='Trains Convolutional Neural Network', default=False)
    parser.add_argument('--CNNP', '--CNN_PRE', '--CNN_PRETRAINED', action="store_true", dest="USE_CNN_PRETRAINED",
                        help='ReTrains Pretrained Convolutional Neural Network', default=False)
    parser.add_argument('--LSTM', action="store_true", dest="USE_LSTM",
                        help='Trains Long Short Term Memory Neural Network', default=False)
    parser.add_argument('--N', '--NOISE', action="store_true", dest="ADD_NOISE",
                        help='Adds noise to the profiling step', default=False)
    parser.add_argument('-v', '-var', '-variable', action="store", dest="VARIABLE", help='Variable to train',
                        default='s001')
    parser.add_argument('-l', '-length', '-input', action="store", dest="INPUT_LENGTH", help='Input Length (default: 700)',
                        type=int, default=700)
    parser.add_argument('-e', '-epochs', action="store", dest="EPOCHS", help='Number of Epochs in Training (default: 75 CNN, 6000 MLP)',
                        type=int, default=6000)
    parser.add_argument('-t', '-traces', action="store", dest="TRAINING_TRACES", help='Number of Traces in Training (default: 200000)',
                        type=int, default=200000)
    parser.add_argument('-mlp_layers', action="store", dest="MLP_LAYERS", help='Number of Layers in MLP (default: 5)',
                        type=int, default=5)
    parser.add_argument('-mlp_nodes', action="store", dest="MLP_NODES", help='Number of Nodes in MLP Layer (default: 200)',
                        type=int, default=200)
    parser.add_argument('-lstm_layers', action="store", dest="LSTM_LAYERS",
                        help='Number of Layers in LSTM (default: 1)',
                        type=int, default=1)
    parser.add_argument('-lstm_nodes', action="store", dest="LSTM_NODES",
                        help='Number of Nodes in LSTM Layer (default: 64)',
                        type=int, default=64)
    parser.add_argument('-b', '-batch', '-batch_size', action="store", dest="BATCH_SIZE", help='Size of Training Batch (default: 200)',
                        type=int, default=200)
    parser.add_argument('-allvar', '-av', action="store", dest="ALL_VARIABLE",
                        help='Train all Variables that match (default: None)', default=None)
    parser.add_argument('-sd', action="store", dest="STANDARD_DEVIATION",
                        help='Standard Deviation for Data Augmentation (default: 100)',
                        type=int, default=100)
    parser.add_argument('-am', '-augment', '-aug', action="store", dest="AUGMENT_METHOD",
                        help='Method for Data Augmentation: 0 Noise, 1 Shift, 2 Average (default: 0)',
                        type=int, default=0)
    parser.add_argument('--PB', '--PROG', '--PROGRESS', action="store_true", dest="PROGRESS_BAR",
                        help='Prints Progress Bar', default=False)
    parser.add_argument('--ALLVARS', action="store_true", dest="ALL_VARS",
                        help='Trains all Variable Nodes', default=False)


    # Target node here
    args            = parser.parse_args()
    USE_MLP         = args.USE_MLP
    USE_CNN         = args.USE_CNN
    USE_CNN_PRETRAINED = args.USE_CNN_PRETRAINED
    USE_LSTM        = args.USE_LSTM
    VARIABLE        = args.VARIABLE
    ADD_NOISE       = args.ADD_NOISE
    INPUT_LENGTH    = args.INPUT_LENGTH
    EPOCHS          = args.EPOCHS
    TRAINING_TRACES = args.TRAINING_TRACES
    MLP_LAYERS      = args.MLP_LAYERS
    MLP_NODES       = args.MLP_NODES
    LSTM_LAYERS     = args.LSTM_LAYERS
    LSTM_NODES      = args.LSTM_NODES
    BATCH_SIZE      = args.BATCH_SIZE
    NORMALISE       = args.NORMALISE
    STANDARD_DEVIATION = args.STANDARD_DEVIATION
    ALL_VARIABLE    = args.ALL_VARIABLE
    AUGMENT_METHOD  = args.AUGMENT_METHOD
    ALL_VARS        = args.ALL_VARS
    PROGRESS_BAR = 1 if args.PROGRESS_BAR else 0

    # Handle dodgy input
    if (INPUT_LENGTH % 2):
        print "|| Error: input length must be even, adding 1 to fix ({} -> {})".format(INPUT_LENGTH, INPUT_LENGTH+1)
        INPUT_LENGTH += 1

    if ALL_VARIABLE is None:
        variable_list = [VARIABLE]
    else:
        variable_list = ['{}{}'.format(ALL_VARIABLE, pad_string_zeros(i+1)) for i in range(variable_dict[ALL_VARIABLE])]

    for variable in variable_list:

        print "$$$ Training Neural Networks $$$\nVariable {}, MLP {} ({} layers, {} nodes per layer), CNN {} (Pretrained {}), LSTM {} ({} layers, {} nodes per layer), Input Length {}, Noise {}, Normalising {}\n{} Epochs, Batch Size {}, Training Traces {}".format(
            variable, USE_MLP, MLP_LAYERS, MLP_NODES, USE_CNN, USE_CNN_PRETRAINED, USE_LSTM, LSTM_LAYERS, LSTM_NODES, INPUT_LENGTH, ADD_NOISE, NORMALISE, EPOCHS, BATCH_SIZE, TRAINING_TRACES)

        # Load the profiling traces and the attack traces
        (X_profiling, Y_profiling), (X_attack, Y_attack) = load_bpann(variable, normalise_traces=NORMALISE,
                                                                      input_length=INPUT_LENGTH, training_traces=TRAINING_TRACES, sd = STANDARD_DEVIATION, augment_method=AUGMENT_METHOD)

        train_variable_model(variable, X_profiling, Y_profiling, X_attack, Y_attack, mlp=USE_MLP, cnn=USE_CNN, cnn_pre=USE_CNN_PRETRAINED, lstm=USE_LSTM, input_length=INPUT_LENGTH, add_noise=ADD_NOISE, epochs=EPOCHS,
            training_traces=TRAINING_TRACES, mlp_layers=MLP_LAYERS, mlp_nodes=MLP_NODES, lstm_layers=LSTM_LAYERS, lstm_nodes=LSTM_NODES, batch_size=BATCH_SIZE, sd=STANDARD_DEVIATION, augment_method=AUGMENT_METHOD, progress_bar=PROGRESS_BAR)

    # for var, length in variable_dict.iteritems():
    #     for i in range(length):
    #         variable = "{}{}".format(var, pad_string_zeros(i+1))
    #         # print variable
    #         print "$$$ Training Neural Networks $$$\nVariable {}, MLP {}, CNN {}, Input Length {}, Noise {}\n".format(VARIABLE, USE_MLP,
    #                                                                                                  USE_CNN, INPUT_LENGTH, ADD_NOISE)
    #         train_variable_model(VARIABLE, mlp=USE_MLP, cnn=USE_CNN, input_length=INPUT_LENGTH, add_noise=ADD_NOISE)

    print "$ Done!"
