import os.path
import sys
import h5py
import numpy as np
import argparse
import timing
from time import time
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, LSTM, Dropout, BatchNormalization
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
from keras.utils.vis_utils import plot_model

from keras import backend as K
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

############# Loss functions #############

#### MLP Best model (6 layers of 200 units)
def mlp_ascad(node=200,layer_nb=6):
	model = Sequential()
	model.add(Dense(node, input_dim=700, activation='relu'))
	for i in range(layer_nb-2):
		model.add(Dense(node, activation='relu'))
	model.add(Dense(256, activation='softmax'))
	optimizer = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

### CNN Best model
def cnn_ascad(classes=256):
	# From VGG16 design
	input_shape = (700,1)
	img_input = Input(shape=input_shape)
	# Block 1
	x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
	print x.get_shape()
	x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
	print x.get_shape()
	# Block 2
	x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
	print x.get_shape()
	x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
	print x.get_shape()
	# Block 3
	x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
	print x.get_shape()
	x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
	print x.get_shape()
	# Block 4
	x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
	print x.get_shape()
	x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
	print x.get_shape()
	# Block 5
	x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
	print x.get_shape()
	x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
	print x.get_shape()
	# Classification block
	x = Flatten(name='flatten')(x)
	print x.get_shape()
	x = Dense(4096, activation='relu', name='fc1')(x)
	print x.get_shape()
	x = Dense(4096, activation='relu', name='fc2')(x)
	print x.get_shape()
	x = Dense(classes, activation='softmax', name='predictions')(x)
	print x.get_shape()

	inputs = img_input
	# Create model.
	model = Model(inputs, x, name='cnn_best')
	optimizer = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


#### MLP Weighted bit model (6 layers of 200 units)
def mlp_weighted_bit(mlp_nodes=200,layer_nb=6, input_length=700, learning_rate=0.00001, classes=256, loss_function='binary_crossentropy'):
    if loss_function is None:
        loss_function='binary_crossentropy'
    model = Sequential()
    model.add(Dense(mlp_nodes, input_dim=input_length, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(mlp_nodes, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    optimizer = RMSprop(lr=learning_rate)
    try:
        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    except ValueError:
        print "!!! Loss Function '{}' not recognised, aborting\n".format(loss_function)
        raise
    return model


#### MLP Best model (6 layers of 200 units)
def mlp_best(mlp_nodes=200,layer_nb=6, input_length=700, learning_rate=0.00001, classes=256, loss_function='categorical_crossentropy'):
    if loss_function is None:
        loss_function='categorical_crossentropy'
    model = Sequential()
    model.add(Dense(mlp_nodes, input_dim=input_length, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(mlp_nodes, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    # Save image!
    plot_model(model, to_file='output/model_plot.png', show_shapes=True, show_layer_names=True)

    optimizer = RMSprop(lr=learning_rate)
    if loss_function=='rank_loss':
        model.compile(loss=tf_rank_loss, optimizer=optimizer, metrics=['accuracy'])
    elif loss_function=='median_probability_loss':
        model.compile(loss=tf_median_probability_loss, optimizer=optimizer, metrics=['accuracy'])
    else:
        try:
            model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        except ValueError:
            print "!!! Loss Function '{}' not recognised, aborting\n".format(loss_function)
            raise
    return model

### CNN From MAKE SOME NOISE (AES_HD)
def cnn_aes_hd(input_length=700, learning_rate=0.00001, classes=256, dense_units=512):
    # From VGG16 design
    input_shape = (input_length, 1)
    img_input = Input(shape=input_shape)

    # # Initial Batch Normalisation
    # x = BatchNormalization(name='initial_batchnorm')(img_input)

    # Block 1 (700)
    x = Conv1D(8, 3, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_batchnorm')(x)
    x = MaxPooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2 (350)
    x = Conv1D(16, 3, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3 (175)
    x = Conv1D(32, 3, activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization(name='block3_batchnorm')(x)
    x = MaxPooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4 (87)
    x = Conv1D(64, 3, activation='relu', padding='same', name='block4_conv1')(x)
    x = MaxPooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5 (43)
    x = Conv1D(64, 3, activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization(name='block5_batchnorm')(x)
    x = MaxPooling1D(2, strides=2, name='block5_pool')(x)
    # Block 6 (21)
    x = Conv1D(128, 3, activation='relu', padding='same', name='block6_conv1')(x)
    x = MaxPooling1D(2, strides=2, name='block6_pool')(x)
    # Block 7 (10)
    x = Conv1D(128, 3, activation='relu', padding='same', name='block7_conv1')(x)
    x = BatchNormalization(name='block7_batchnorm')(x)
    x = MaxPooling1D(2, strides=2, name='block7_pool')(x)
    # Block 8 (5)
    x = Conv1D(256, 3, activation='relu', padding='same', name='block8_conv1')(x)
    x = MaxPooling1D(2, strides=2, name='block8_pool')(x)
    # Block 9 (2)
    x = Conv1D(256, 3, activation='relu', padding='same', name='block9_conv1')(x)
    x = BatchNormalization(name='block9_batchnorm')(x)
    x = MaxPooling1D(2, strides=2, name='block9_pool')(x)

    # Now 1!

    # First Dropout Layer
    x = Dropout(0.5, name='dropout1')(x)
    # Classification block
    x = Flatten(name='flatten')(x)

    # One Dense layer
    x = Dense(dense_units, activation='relu', name='fc')(x)
    # Second Dropout Layer
    x = Dropout(0.5, name='dropout2')(x)

    # Output layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

### CNN Best model
def cnn_best(input_length=700, learning_rate=0.00001, classes=256, dense_units=4096):
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
    # Two Dense layers
    x = Dense(dense_units, activation='relu', name='fc1')(x)
    x = Dense(dense_units, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### CNN Previously Trained model
def cnn_pretrained(input_length=700, learning_rate=0.00001, classes=256):
    # load model
    cnn_previous = load_model(CNN_ASCAD_FILEPATH)
    for layer in cnn_previous.layers[:-6]:
        layer.trainable = False
    model = Sequential()
    model.add(cnn_previous)
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

### LSTM Best model
def lstm_best(input_length=700, layer_nb=1, lstm_nodes=64, use_dropout=True, learning_rate=0.00001, classes=256):
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
    optimizer = RMSprop(lr=learning_rate)
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
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, validation_data=None, progress_bar=1, hamming_distance_encoding=False, one_hot=True, multilabel=False, hammingweight=False):

    check_file_exists(os.path.dirname(save_file_name))
    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    callbacks=[save_model, TrainValTensorBoard(write_graph=True)]
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

    # Split up for debug
    if multilabel:
        # print "Before: {} {} {}".format(Y_profiling.shape, type(Y_profiling), Y_profiling)
        reshaped_y = np.unpackbits( np.expand_dims(Y_profiling, 1).astype(np.uint8), axis=1)
        reshaped_val = np.unpackbits( np.expand_dims(validation_data[1], 1).astype(np.uint8), axis=1)
    elif hamming_distance_encoding:
        reshaped_y = hamming_distance_encode_bulk(Y_profiling)
        reshaped_val = hamming_distance_encode_bulk(validation_data[1])
    elif one_hot:
        reshaped_y = to_categorical(Y_profiling, num_classes=9 if hammingweight else 256)
        reshaped_val = to_categorical(validation_data[1], num_classes=9 if hammingweight else 256)
    else:
        reshaped_y = Y_profiling
        reshaped_val = validation_data[1]

    history = model.fit(x=Reshaped_X_profiling, y=reshaped_y, batch_size=batch_size, verbose = progress_bar, epochs=epochs, callbacks=callbacks, validation_data=(Reshaped_validation_data, reshaped_val))
    return history

# def train_svm()


def train_variable_model(variable, X_profiling, Y_profiling, X_attack, Y_attack, mlp=True, cnn=True, cnn_pre=False, lstm=True, svm=False, add_noise=False, input_length=700, normalise_traces=True, epochs=None, training_traces=50000, mlp_layers=6, lstm_layers=1, batch_size=200, sd=100, augment_method=0, jitter=None, progress_bar=1, mlp_nodes=200, lstm_nodes=64, learning_rate=0.00001, multilabel=False, hammingweight=False, loss_function=None, hamming_distance_encoding=False, scratch_storage=False, use_ascad=False):

    store_directory = NEURAL_MODEL_FOLDER if scratch_storage else MODEL_FOLDER

    classes = 9 if hammingweight else 256
    hammingweight_flag = '_hw' if hammingweight else ''
    hammingdistance_flag = '_hamdistencode' if hamming_distance_encoding else ''

    if add_noise:
        standard_deviation = 10
        seed = 1
        np.random.seed(seed)
        X_profiling = X_profiling + np.round(np.random.normal(0, standard_deviation, X_profiling.shape)).astype(int)
        X_attack = X_attack + np.round(np.random.normal(0, standard_deviation, X_attack.shape)).astype(int)

    if use_ascad:
        # Done slightly differently
        ascad_model = cnn_ascad() if cnn else mlp_ascad()
        train_model(X_profiling, Y_profiling, ascad_model, store_directory + '{}_{}_ASCAD.h5'.format(variable, 'cnn' if cnn else 'mlp'), epochs=75 if cnn else 200, batch_size=200 if cnn else 100, validation_data=(X_attack, Y_attack),
        progress_bar=progress_bar, hammingweight=hammingweight, hamming_distance_encoding=hamming_distance_encoding)

    ### CNN training
    elif cnn:
        # TODO: Test New CNN!
        # cnn_best_model = cnn_best(input_length=input_length, learning_rate=learning_rate, classes=classes)
        cnn_best_model = cnn_aes_hd(input_length=input_length, learning_rate=learning_rate, classes=classes)
        cnn_epochs = epochs if epochs is not None else 75
        cnn_batchsize = batch_size
        train_model(X_profiling, Y_profiling, cnn_best_model, store_directory +
                    "{}_cnn{}{}_window{}_epochs{}_batchsize{}_lr{}_sd{}_traces{}_aug{}_jitter{}.h5".format(
                        variable, hammingweight_flag, hammingdistance_flag, input_length, cnn_epochs, cnn_batchsize, learning_rate, sd, training_traces, augment_method, jitter),
                    epochs=cnn_epochs, batch_size=cnn_batchsize, validation_data=(X_attack, Y_attack),
                    progress_bar=progress_bar, hammingweight=hammingweight, hamming_distance_encoding=hamming_distance_encoding)

    ### CNN pre-trained training
    elif cnn_pre:
        cnn_pretrained_model = cnn_pretrained(input_length=input_length, learning_rate=learning_rate, classes=classes)
        cnn_epochs = epochs if epochs is not None else 75
        cnn_batchsize = batch_size
        train_model(X_profiling, Y_profiling, cnn_pretrained_model, store_directory +
                    "{}_cnnpretrained{}_window{}_epochs{}_batchsize{}_lr{}_sd{}_traces{}_aug{}_jitter{}.h5".format(
                        variable, hammingweight_flag, input_length, cnn_epochs, cnn_batchsize, learning_rate, sd, training_traces, augment_method, jitter),
                    epochs=cnn_epochs, batch_size=cnn_batchsize, validation_data=(X_attack, Y_attack),
                    progress_bar=progress_bar, hammingweight=hammingweight, hamming_distance_encoding=hamming_distance_encoding)

    ### MLP training
    elif mlp:
        if multilabel:
            mlp_best_model = mlp_weighted_bit(input_length=input_length, layer_nb=mlp_layers, learning_rate=learning_rate, classes=classes, loss_function=loss_function)
        else:
            mlp_best_model = mlp_best(input_length=input_length, layer_nb=mlp_layers, learning_rate=learning_rate, classes=classes, loss_function=loss_function)
        mlp_epochs = epochs if epochs is not None else 200
        mlp_batchsize = batch_size

        train_model(X_profiling, Y_profiling, mlp_best_model, store_directory +
                    "{}_mlp{}{}{}{}_nodes{}_window{}_epochs{}_batchsize{}_lr{}_sd{}_traces{}_aug{}_jitter{}_{}.h5".format(
                        variable, mlp_layers, '_multilabel' if multilabel else '', hammingweight_flag, hammingdistance_flag, mlp_nodes, input_length, mlp_epochs, mlp_batchsize, learning_rate, sd,
                        training_traces, augment_method, jitter, 'defaultloss' if loss_function is None else loss_function.replace('_','')), epochs=mlp_epochs, batch_size=mlp_batchsize,
                    validation_data=(X_attack, Y_attack), progress_bar=progress_bar, multilabel=multilabel, hammingweight=hammingweight, hamming_distance_encoding=hamming_distance_encoding)

    ### LSTM training
    elif lstm:
        lstm_best_model = lstm_best(input_length=input_length, layer_nb=lstm_layers, learning_rate=learning_rate, classes=classes)
        lstm_epochs = epochs if epochs is not None else 75
        lstm_batchsize = batch_size
        train_model(X_profiling, Y_profiling, lstm_best_model, store_directory +
                    "{}_lstm{}{}_nodes{}_window{}_epochs{}_batchsize{}_lr{}_sd{}_traces{}_aug{}_jitter{}.h5".format(
                        variable, lstm_layers, hammingweight_flag, lstm_nodes, input_length, lstm_epochs, lstm_batchsize, learning_rate, sd,
                        training_traces, augment_method, jitter), epochs=lstm_epochs, batch_size=lstm_batchsize,
                    validation_data=(X_attack, Y_attack), progress_bar=progress_bar, hammingweight=hammingweight, hamming_distance_encoding=hamming_distance_encoding)

    ### SVM training
    elif svm:
        svm_best_model = svm_best(input_length=input_length, layer_nb=svm_layers, classes=classes)
        svm_epochs = epochs if epochs is not None else 75
        svm_batchsize = batch_size
        train_model(X_profiling, Y_profiling, svm_best_model, store_directory +
                    "{}_svm{}{}_nodes{}_window{}_epochs{}_batchsize{}_sd{}_traces{}_aug{}_jitter{}.h5".format(
                        variable, svm_layers, hammingweight_flag, svm_nodes, input_length, svm_epochs, svm_batchsize, sd,
                        training_traces, augment_method, jitter), epochs=svm_epochs, batch_size=svm_batchsize,
                    validation_data=(X_attack, Y_attack), progress_bar=progress_bar, hammingweight=hammingweight, hamming_distance_encoding=hamming_distance_encoding)



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
    parser.add_argument('-l', '-length', '-input', '-window', action="store", dest="INPUT_LENGTH", help='Input Length (default: 2000)',
                        type=int, default=2000)
    parser.add_argument('-lr', '-learn', '-learning_rate', action="store", dest="LEARNING_RATE", help='Learning Rate (default: 0.00001)',
                        type=float, default=0.00001)
    parser.add_argument('-e', '-epochs', action="store", dest="EPOCHS", help='Number of Epochs in Training (default: 75 CNN, 100 MLP)',
                        type=int, default=100)
    parser.add_argument('-t', '-traces', action="store", dest="TRAINING_TRACES", help='Number of Traces in Training (default: 200000)',
                        type=int, default=200000)
    parser.add_argument('-vt', '-validation_traces', action="store", dest="VALIDATION_TRACES", help='Number of Validation Traces in Testing, taken from Training Traces (default: 10000)',
                        type=int, default=10000)
    parser.add_argument('-mlp_layers', action="store", dest="MLP_LAYERS", help='Number of Layers in MLP (default: 5)',
                        type=int, default=5)
    parser.add_argument('-mlp_nodes', action="store", dest="MLP_NODES", help='Number of Nodes in MLP Layer (default: 200)',
                        type=int, default=100)
    parser.add_argument('-lstm_layers', action="store", dest="LSTM_LAYERS",
                        help='Number of Layers in LSTM (default: 1)',
                        type=int, default=1)
    parser.add_argument('-lstm_nodes', action="store", dest="LSTM_NODES",
                        help='Number of Nodes in LSTM Layer (default: 64)',
                        type=int, default=64)
    parser.add_argument('-b', '-batch', '-batch_size', action="store", dest="BATCH_SIZE", help='Size of Training Batch (default: 200)',
                        type=int, default=50)
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
    parser.add_argument('-j', '-jitter', action="store", dest="JITTER",
                        help='Clock Jitter to use on real traces (default: None)',
                        type=int, default=None)
    parser.add_argument('--TEST', '--TEST_VARIABLES', action="store_true", dest="TEST_VARIABLES",
                        help='Trains only specified Testing Variable Nodes', default=False)
    parser.add_argument('--MULTILABEL', '--ML', '--M', action="store_true", dest="MULTILABEL",
                        help='Uses multilabels in binary form', default=False)
    parser.add_argument('--RV', '--RKV', '--RANDOMKEY_VALIDATION', action="store_false", dest="RANDOMKEY_VALIDATION",
                        help='Takes validation traces from randomkey set (subtracting from training traces!), default True', default=True)
    parser.add_argument('--HW', '--HAMMINGWEIGHT', '--HAMMING_WEIGHT', action="store_true", dest="HAMMINGWEIGHT",
                        help='Trains to match Hamming Weight rather than identity', default=False)
    parser.add_argument('--HD', '--HAMMINGDISTANCE', '--HAMMING_DISTANCE', action="store_true", dest="HAMMING_DISTANCE_ENCODING",
                        help='Encodes to scaled Hamming Weight Distance, rather than one hot encoding', default=False)
    parser.add_argument('--META', '--LOAD_META', '--LOAD_METADATA', action="store_false", dest="LOAD_METADATA",
                        help='Toggles loading of metadata, default True (bad for mig!)', default=True)
    parser.add_argument('--SCRATCH', '--SCRATCH_STORAGE', '--S', action="store_true", dest="SCRATCH_STORAGE",
                        help='Stores neural networks on scratch storage (external hard drive)', default=False)
    parser.add_argument('--ASCAD', '--USE_ASCAD', action="store_true", dest="USE_ASCAD",
                        help='Uses ASCAD Default Model, CNN or MLP', default=False)

    parser.add_argument('-loss', '-loss_function', action="store", dest="LOSS_FUNCTION", help='Loss Function (default: None (uses standard depending on model structure, usually categorical cross entropy))',
                        default=None)

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
    JITTER          = args.JITTER
    TEST_VARIABLES  = args.TEST_VARIABLES
    LEARNING_RATE   = args.LEARNING_RATE
    MULTILABEL      = args.MULTILABEL
    VALIDATION_TRACES = args.VALIDATION_TRACES
    RANDOMKEY_VALIDATION = args.RANDOMKEY_VALIDATION
    HAMMINGWEIGHT = args.HAMMINGWEIGHT
    LOSS_FUNCTION = args.LOSS_FUNCTION
    HAMMING_DISTANCE_ENCODING = args.HAMMING_DISTANCE_ENCODING
    LOAD_METADATA = args.LOAD_METADATA
    SCRATCH_STORAGE = args.SCRATCH_STORAGE
    USE_ASCAD = args.USE_ASCAD

    if not USE_MLP and not USE_CNN and not USE_CNN_PRETRAINED and not USE_LSTM:
        print "|| No models set to run - setting USE_MLP to True"
        USE_MLP = True

    PROGRESS_BAR = 1 if args.PROGRESS_BAR else 0

    # Handle dodgy input
    if (INPUT_LENGTH % 2) and INPUT_LENGTH != 1 and INPUT_LENGTH != -1:
        print "|| Error: input length must be even, adding 1 to fix ({} -> {})".format(INPUT_LENGTH, INPUT_LENGTH+1)
        INPUT_LENGTH += 1

    # Handle ASCAD Defaults
    if USE_ASCAD:
        print "|| Setting to ASCAD Default Values (epochs etc):"
        print "|| * INPUT_LENGTH {} -> 700".format(INPUT_LENGTH)
        INPUT_LENGTH = 700


    if TEST_VARIABLES:
        variable_list = ['k001','s001','t001','k004']
    if ALL_VARS:
        variable_list = get_variable_list()
    elif ALL_VARIABLE is None:
        variable_list = [VARIABLE]
    else:
        variable_list = ['{}{}'.format(ALL_VARIABLE, pad_string_zeros(i+1)) for i in range(variable_dict[ALL_VARIABLE])]

    if RANDOMKEY_VALIDATION:
        TRAINING_TRACES -= VALIDATION_TRACES

    for variable in variable_list:

        print "$$$ Training Neural Networks $$$\nVariable {}, Hamming Weight {} Hamming Distance Encoding {}, MLP {} ({} layers, {} nodes per layer), CNN {} (Pretrained {}), LSTM {} ({} layers, {} nodes per layer), Input Length {}, Learning Rate {}, Noise {}, Jitter {}, Normalising {}\n{} Epochs, Batch Size {}, Training Traces {}, Validation Traces {}, ASCAD {}".format(
            variable, HAMMINGWEIGHT, HAMMING_DISTANCE_ENCODING, USE_MLP, MLP_LAYERS, MLP_NODES, USE_CNN, USE_CNN_PRETRAINED, USE_LSTM, LSTM_LAYERS, LSTM_NODES, INPUT_LENGTH, LEARNING_RATE, ADD_NOISE, JITTER, NORMALISE, EPOCHS, BATCH_SIZE, TRAINING_TRACES, VALIDATION_TRACES, USE_ASCAD)

        # Load the profiling traces and the attack traces
        (X_profiling, Y_profiling), (X_attack, Y_attack) = load_bpann(variable, normalise_traces=NORMALISE,
                                                                      input_length=INPUT_LENGTH, training_traces=TRAINING_TRACES, sd = STANDARD_DEVIATION, augment_method=AUGMENT_METHOD, jitter=JITTER, validation_traces=VALIDATION_TRACES, randomkey_validation=RANDOMKEY_VALIDATION,
                                                                      hammingweight=HAMMINGWEIGHT,
                                                                      load_metadata=LOAD_METADATA)

        # Handle Input Length of -1
        if INPUT_LENGTH < 0:
            # Set to length of X_profiling
            print "|| Changing Input Length from {} to {} (max samples)".format(INPUT_LENGTH, X_profiling.shape[1])
            INPUT_LENGTH = X_profiling.shape[1]

        train_variable_model(variable, X_profiling, Y_profiling, X_attack, Y_attack, mlp=USE_MLP, cnn=USE_CNN, cnn_pre=USE_CNN_PRETRAINED, lstm=USE_LSTM, input_length=INPUT_LENGTH, add_noise=ADD_NOISE, epochs=EPOCHS,
            training_traces=TRAINING_TRACES, mlp_layers=MLP_LAYERS, mlp_nodes=MLP_NODES, lstm_layers=LSTM_LAYERS, lstm_nodes=LSTM_NODES, batch_size=BATCH_SIZE, sd=STANDARD_DEVIATION, augment_method=AUGMENT_METHOD, jitter=JITTER, progress_bar=PROGRESS_BAR,
            learning_rate=LEARNING_RATE, multilabel=MULTILABEL, hammingweight=HAMMINGWEIGHT, loss_function=LOSS_FUNCTION, hamming_distance_encoding=HAMMING_DISTANCE_ENCODING, scratch_storage=SCRATCH_STORAGE, use_ascad=USE_ASCAD)

    # for var, length in variable_dict.iteritems():
    #     for i in range(length):
    #         variable = "{}{}".format(var, pad_string_zeros(i+1))
    #         # print variable
    #         print "$$$ Training Neural Networks $$$\nVariable {}, MLP {}, CNN {}, Input Length {}, Noise {}\n".format(VARIABLE, USE_MLP,
    #                                                                                                  USE_CNN, INPUT_LENGTH, ADD_NOISE)
    #         train_variable_model(VARIABLE, mlp=USE_MLP, cnn=USE_CNN, input_length=INPUT_LENGTH, add_noise=ADD_NOISE)

    print "$ Done!"
