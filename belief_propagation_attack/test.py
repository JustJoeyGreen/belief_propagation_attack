import leakageSimulatorAESFurious as lSimF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis
from utility import *
import realTraceHandler as rTH
import argparse
import matplotlib.pyplot as plt
import trsfile
import operator
import timing
import factorGraphAES as fG



# KERAS AND TENSORFLOW
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







def plot_results(testname='ranks_'):

    # Get results files
    file_prefix = OUTPUT_FOLDER+'new_results/'
    results_files = get_files_in_folder(folder=file_prefix, substring=testname)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Test Name: {}'.format(testname))

    graph_connections = ['IND', 'SEQ', 'LFG']
    snrs = ['-1', '-7']
    graphs = ['G1_', 'G1A', 'G2']

    # for id_x, string_x in enumerate():

    for snr_i, snr_bool in enumerate([True, False]):
        for acyclic_i, acyclic_bool in enumerate([True, False]):
            print "\nSnr {}, Acyclic {}:".format(snr_bool, acyclic_bool)
            for result_file in [rfile for rfile in results_files if (string_contains(rfile, '-1') == snr_bool and (string_contains(rfile, 'G1A') == acyclic_bool))]:
                current_results = np.mean(np.load(file_prefix + result_file), axis=0)
                axs[snr_i][acyclic_i].plot(current_results, label=get_graph_connection_method(result_file))
            axs[snr_i][acyclic_i].legend()
    plt.show()


def tf_get_median(v):
    v = tf.reshape(v, [-1])
    mid = v.get_shape()[0]//2 + 1
    return tf.nn.top_k(v, mid).values[-1]

def tf_rank(input):


    argmaxed_onehot = K.argmax(input[0])
    argsort1 = tf.argsort(input[1], direction='DESCENDING')
    argsort2 = tf.argsort(argsort1, direction='ASCENDING')
    return tf.gather_nd(argsort2[0], argmaxed_onehot)


    # print '\n\n\n\n\n\n\n\ntype1', type(argmaxed_onehot), '\n\n\n\n\n\n\n\n'
    # # casted_to_int = tf.cast(argmaxed_onehot, dtype=int)
    # casted_to_int = argmaxed_onehot.eval()
    # print '\n\n\n\n\n\n\n\ntype3', type(casted_to_int), '\n\n\n\n\n\n\n\n'
    #
    # return tf.size(tf.where(input[1] >= input[1][argmaxed_onehot]))
    # return tf.size(tf.where(input[1] >= input[1][0]))

def tf_rank_loss(y_true, y_pred):
    argsort1 = tf.argsort(y_pred, direction='DESCENDING')
    argsort2 = tf.argsort(argsort1, direction='ASCENDING')
    argmaxed_onehot = tf.argmax(y_true, output_type=tf.int32, axis=1)
    # reshaped_onehot = tf.reshape(argmaxed_onehot, [tf.shape(argsort2)[0], 1])
    reshaped_onehot = tf.expand_dims(argmaxed_onehot, 1)
    tf_range = tf.range(tf.shape(argsort2)[0], dtype=tf.int32)
    reshaped_tf_range = tf.expand_dims(tf_range, 1)
    concatenated_onehot = tf.concat([reshaped_tf_range, reshaped_onehot], 1)
    gathered = tf.gather_nd(argsort2, concatenated_onehot)
    mean = tf.cast(tf.reduce_mean(gathered), tf.float32)
    # print "Mean, type {} ({}), shape {}".format(type(mean), mean.dtype, mean.get_shape())
    return mean

    # out = tf.constant(1, dtype=tf.float32)
    # print "Here, type {} ({}), shape {}".format(type(out), out.dtype, out.get_shape())
    # return out


def tf_cross_entropy_test(y_true, y_pred):
    cross = tf.losses.softmax_cross_entropy(y_true, y_pred)
    # print "Here, type {} ({}), shape {}".format(type(cross), cross.dtype, cross.get_shape())
    return cross

def tensors():

    # my_values = [0, 1, 3]
    # my_values = range(256)
    my_values = [3, 7]

    true_onehot = np.array([get_plaintext_array(255-i) for i in my_values])

    true = np.array([(i) for i in my_values])
    # print "True:\n{}\n\n".format(true)
    # pred = np.array([get_hamming_weight_array(i) for i in range(hw_limit)])
    pred = np.array([get_hamming_weight_array(get_hw(i)) for i in my_values])
    # pred = np.array([get_hamming_weight_array(1)])
    y_true = tf.constant(true, dtype=tf.float32)
    y_true_onehot = tf.constant(true_onehot)
    y_pred = tf.constant(pred, dtype=tf.float32)



    argsort1 = tf.argsort(y_pred, direction='DESCENDING')
    argsort2 = tf.argsort(argsort1, direction='ASCENDING')
    argmaxed_onehot = tf.argmax(y_true_onehot, output_type=tf.int32, axis=1)
    # reshaped_onehot = tf.reshape(argmaxed_onehot, [tf.shape(argsort2)[0], 1])
    reshaped_onehot = tf.expand_dims(argmaxed_onehot, 1)
    tf_range = tf.range(tf.shape(argsort2)[0], dtype=tf.int32)
    reshaped_tf_range = tf.expand_dims(tf_range, 1)
    concatenated_onehot = tf.concat([reshaped_tf_range, reshaped_onehot], 1)
    gathered = tf.gather_nd(argsort2, concatenated_onehot)
    mean = tf.reduce_mean(gathered)


    # with tf.Session() as sess:
    #     print "y_pred ({}, {}):\n{}\n".format(y_pred.dtype, y_pred.get_shape(), sess.run(y_pred))
    #     print "argsort1 ({}, {}):\n{}\n".format(argsort1.dtype, argsort1.get_shape(), sess.run(argsort1))
    #     print "argsort2 ({}, {}):\n{}\n".format(argsort2.dtype, argsort2.get_shape(), sess.run(argsort2))
    #     print "argmaxed_onehot ({}, {}):\n{}\n".format(argmaxed_onehot.dtype, argmaxed_onehot.get_shape(), sess.run(argmaxed_onehot))
    #     print "reshaped_onehot ({}, {}):\n{}\n".format(reshaped_onehot.dtype, reshaped_onehot.get_shape(), sess.run(reshaped_onehot))
    #     print "tf_range ({}, {}):\n{}\n".format(tf_range.dtype, tf_range.get_shape(), sess.run(tf_range))
    #     print "reshaped_tf_range ({}, {}):\n{}\n".format(reshaped_tf_range.dtype, reshaped_tf_range.get_shape(), sess.run(reshaped_tf_range))
    #     print "concatenated_onehot ({}, {}):\n{}\n".format(concatenated_onehot.dtype, concatenated_onehot.get_shape(), sess.run(concatenated_onehot))
    #     print "gathered ({}, {}):\n{}\n".format(gathered.dtype, gathered.get_shape(), sess.run(gathered))
    #     print "mean ({}, {}):\n{}\n".format(mean.dtype, mean.get_shape(), sess.run(mean))
    #
    # exit(1)

    loss1 = tf_rank_loss(y_true_onehot, y_pred)
    loss2 = tf.losses.softmax_cross_entropy(y_true_onehot, y_pred)

    # Rank list of predictions
    # argsorted = tf.argsort(pred, direction='DESCENDING')
    # reversed = K.reverse(argsorted, axes=1)
    # ranks =

    # result = argsorted
    # print result

    # with tf.Session() as sess:
    #     print "\nTrue:\n"
    #     print(sess.run(y_true))
    #     print "\nPred:\n"
    #     print(sess.run(y_pred))
    #     print "\nCross Entropy Loss Function:\n"
    #     print(sess.run(loss2))
    #     print "\nRank Loss Function:\n"
    #     print(sess.run(loss1))

    # Model time
    mlp_nodes = 200
    input_length = 700
    layer_nb = 5
    learning_rate = 0.00001

    model = Sequential()
    model.add(Dense(mlp_nodes, input_dim=input_length, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(mlp_nodes, activation='relu'))
    model.add(Dense(256, activation='softmax'))
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=tf_rank_loss, optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss=tf_cross_entropy_test, optimizer=optimizer, metrics=['accuracy'])


    print "Model compiled successfully..."


def numpy_tests():

    my_values = range(3) # 256
    true_onehot = np.array([get_plaintext_array(i) for i in my_values])
    pred = np.array([get_hamming_weight_array(get_hw(i)) for i in my_values])

    for i, value in enumerate(zip(true_onehot, pred)):

        print "\n\n\ni: {}\n".format(i)

        # Step 1: Decode one hot
        decoded_onehot = np.argmax(value[0])
        print "Decoded One Hot: {}".format(decoded_onehot)

        # Step 2: ???
        argsorted_pred = np.argsort(value[1])


def weighted_bits():

    multilabel_probabilities = np.array([0.2,0.3,0.1,0.01,0.2,0.1,0.05,0.4])

    out = multilabel_probabilities_to_probability_distribution(multilabel_probabilities)

    # out = np.zeros(256)

    # for i in range(256):
    #     bits = np.unpackbits(np.array(i, dtype=np.uint8))
    #     bit_corrected = (np.ones(8) - bits) - multilabel_probabilities
    #     out[i] = np.abs(np.prod(bit_corrected))

    print out


def value_occurance_checker(var_name='s', var_num=1, randomkey_extra = False, extra_size = 10000):

    print "* Value Occurance Checker, {} {} *".format(var_name, var_num)

    for extra_file in [False, True]:
        filename = '{}{}{}.npy'.format(REALVALUES_FOLDER, 'extra_' if extra_file and not randomkey_extra else '', var_name)
        real_values = np.load(filename)[var_num-1]
        if randomkey_extra:
            if extra_file:
                real_values = real_values[-extra_size:]
            else:
                real_values = real_values[:-extra_size]
        unique, counts = np.unique(real_values, return_counts=True)
        if unique.shape[0] != 256 and not (var_name == 'k' and unique.shape[0] == 1):
            print "GOTCHA! Unique is only size {}:\n{}\n".format(unique.shape[0], unique)
            raise
        print "> {} Values ({}):".format('Attack' if extra_file else 'Profile', real_values.shape[0])
        print_statistics(counts, mode=extra_file)





if __name__ == "__main__":

    for var_name in ['s', 'k']:
        for var_num in [1,4,8]:
            value_occurance_checker(var_name, var_num, randomkey_extra=True)
    exit(1)

    # Testnames: GraphStructure, ReducedGraphs, GroundTruth
    # plot_results(testname='GraphStructure')

    # numpy_tests()

    # plot_results(testname='GraphStructure')

    # exit(1)

    # fig, ax = plt.subplots()

    plt.figure(figsize=(20,5))

    for var, length in variable_dict.iteritems():
        print "\n* Var {}".format(var)
        timepoint = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var))
        # ax.scatter(var, timepoint, label=var)
        var_l = [var for i in range(len(timepoint))]
        plt.scatter(timepoint, var_l)


    vertical_lines = [0, 4650, 12100, 19690, 23400, 27800, 35000, 42800, 48000]

    for vertical_line in vertical_lines:
        plt.axvline(x=vertical_line, linewidth=0.1)
    # ax.legend()

    plt.xlabel("Samples")
    plt.ylabel("Variables")

    # plt.show()
    plt.savefig("output/hw_timepoints.pdf")
    # plt.savefig("output/identity_timepoints.pdf")

    exit(1)

    for var, length in variable_dict.iteritems():
        print "\n* Var {}".format(var)
        for j in range(length):
            timepoint = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var))[j]
            hw_timepoint = np.argmax(np.load("{}{}_{}_HW.npy".format(COEFFICIENT_FOLDER, var, j)))
            print "TP: {:6} HWTP: {:6} RANGE: {:6}".format(timepoint, hw_timepoint, np.abs(timepoint - hw_timepoint))


    TEST = False

    # USE_REAL_TRACE_HANDLER = True
    USE_REAL_TRACE_HANDLER = False

    SHIFT_TRACES = True
    SHIFT_VAL = 2
    SHIFT_EXTRA = False

    my_var = 'k004'
    var_name, var_num, _ = split_variable_name(my_var)
    hw = False

    (X_profiling, Y_profiling), (X_attack, Y_attack) = load_bpann(my_var, load_metadata=False, normalise_traces=True, input_length=1, training_traces=200000, sd = 100, augment_method=2, jitter=None)

    powerval_list = [list() for i in range(256 if not hw else 9)]

    for i, label in enumerate(Y_profiling):
        v = X_profiling[i][0]
        powerval_list[get_hw(label) if hw else label].append(v)

    max_list = np.array([max(i) for i in powerval_list])
    min_list = np.array([min(i) for i in powerval_list])
    avg_list = np.array([get_average(i) for i in powerval_list])

    # plt.plot(max_list, label='max_list')
    # plt.plot(min_list, label='min_list')
    # plt.plot(avg_list, label='avg_list')

    plt.errorbar(np.arange(256 if not hw else 9), avg_list, [avg_list - min_list, max_list - avg_list], fmt='.k', ecolor='gray', lw=1)

    # plt.legend()
    plt.show()

    exit(1)
