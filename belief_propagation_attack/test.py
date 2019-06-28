import leakageSimulatorAESFurious as lSimF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis
from utility import *
import realTraceHandler as rTraceH
import argparse
import matplotlib.pyplot as plt
from matplotlib import rc
import trsfile
import operator
import timing
import factorGraphAES as fG
import csv
import os


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

    # Testnames: GraphStructure, ReducedGraphs, GroundTruth

    # Get results files
    file_prefix = OUTPUT_FOLDER+'new_results/'
    results_files = get_files_in_folder(folder=file_prefix, substring=testname)

    print 'Files:', results_files

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    if testname == 'GraphStructure':
        # fig, axs = plt.subplots(2, 2)
        pass
    elif testname == 'ReducedGraphs':
        # fig, axs = plt.subplots(1, 2)
        pass


    # fig.suptitle('Test Name: {}'.format(testname))

    graph_connections = ['IND', 'SEQ', 'LFG']
    snrs = ['-1', '-7']
    graphs = ['G1_', 'G1A', 'G2']

    # for id_x, string_x in enumerate():

    for snr_i, snr_bool in enumerate([True, False]):

        if testname == 'GraphStructure':
            for acyclic_i, acyclic_bool in enumerate([True, False]):
                print ("\nSnr {}, Acyclic {}:").format(snr_bool, acyclic_bool)

                # LFG
                lfg_x = list()
                lfg_y = list()

                for result_file in [rfile for rfile in results_files if (string_contains(rfile, '-1') == snr_bool and (string_contains(rfile, 'G1A') == acyclic_bool)) and not string_contains(rfile, 'KEYAVG')]:

                    print "> {}".format(result_file)

                    # LFG HANDLE
                    if get_graph_connection_method(result_file) == 'LFG':
                        lfg_traces = get_traces_used(result_file)

                        try:
                            current_results = np.mean(np.load(file_prefix + result_file, allow_pickle=True), axis=0)
                            print ("Plot for LFG {:3}: {}".format(lfg_traces, current_results))
                            # print 'LFG Shape: {}'.format(current_results.shape)
                            lfg_x.append(eval(lfg_traces))
                            lfg_y.append(current_results)
                            # axs[snr_i][acyclic_i].plot(lfg_traces, current_results, label=get_graph_connection_method(result_file))
                        except ValueError:
                            print "Could not load file {}".format(result_file)
                        continue
                    else:

                        try:
                            current_results = np.mean(np.load(file_prefix + result_file, allow_pickle=True), axis=0)
                            # print 'IFG / LFG Shape: {}'.format(current_results.shape)

                            # axs[snr_i][acyclic_i].plot(current_results, label=get_graph_connection_method(result_file))
                            # plt.plot(current_results, label=get_graph_connection_method(result_file))

                            np.savetxt('output/{}_{}_acyclic{}_{}.csv'.format(testname, get_graph_connection_method(result_file), acyclic_bool, '-1' if snr_bool else '-7'), current_results, delimiter=",")

                        except ValueError:
                            print "Could not load file {}".format(result_file)

                # Finish with LFG Traces
                lfg_x_sorted = sorted(lfg_x)
                lfg_y_sorted = [x for _,x in sorted(zip(lfg_x,lfg_y))]

                print lfg_x, lfg_y
                print lfg_x_sorted, lfg_y_sorted

                interpolated = np.interp(np.arange(1,101), lfg_x_sorted, lfg_y_sorted)

                # Make into length 100 by interpolating

                np.savetxt('output/{}_{}_acyclic{}_{}.csv'.format(testname, 'LFG', acyclic_bool, '-1' if snr_bool else '-7'), interpolated, delimiter=",")

                # axs[snr_i][acyclic_i].plot(lfg_x_sorted, lfg_y_sorted, label='LFG')
                # plt.plot(lfg_x_sorted, lfg_y_sorted, label='LFG')

                # Legend
                # axs[snr_i][acyclic_i].legend()
                # plt.xlabel(r'\textbf{Traces}', fontsize=11)
                # plt.ylabel(r'\textbf{Median Rank}', fontsize=11)
                # plt.legend()
                # plt.savefig('output/{}_acyclic{}_{}.svg'.format(testname, acyclic_bool, '-1' if snr_bool else '-7'), format='svg', dpi=1200)
                # plt.clf()

        elif testname == 'ReducedGraphs':
            print ("\nSnr {}:".format(snr_bool))
            for result_file in [rfile for rfile in results_files if (string_contains(rfile, '-1') == snr_bool)]:

                try:
                    loaded = np.load(file_prefix + result_file, allow_pickle=True)
                    current_results = np.mean(loaded, axis=0)

                    if snr_i == False:
                        current_results = current_results[:8]

                    # axs[snr_i].plot(current_results, label=get_graph_size_and_structure(result_file))

                    np.savetxt('output/{}_{}_{}.csv'.format(testname, get_graph_size_and_structure(result_file), '-1' if snr_bool else '-7'), current_results, delimiter=",")

                except ValueError:
                    print "Could not load file {}".format(result_file)
            # axs[snr_i].legend()

    # try:
    #     axs[0].set_xlabel('Traces', fontsize=11)
    #     axs[0].set_ylabel('Classification Rank', fontsize=11)
    # except AttributeError:
    #     axs[0][0].set_xlabel('Traces', fontsize=11)
    #     axs[0][0].set_ylabel('Classification Rank', fontsize=11)
    #
    #
    # plt.show()
    #
    # fig.savefig("output/{}.pdf".format(testname), bbox_inches='tight')



def tf_get_median(v):
    v = tf.reshape(v, [-1])
    mid = v.get_shape()[0]//2 + 1
    return tf.nn.top_k(v, mid).values[-1]

def tf_rank(input):


    argmaxed_onehot = K.argmax(input[0])
    argsort1 = tf.argsort(input[1], direction='DESCENDING')
    argsort2 = tf.argsort(argsort1, direction='ASCENDING')
    return tf.gather_nd(argsort2[0], argmaxed_onehot)

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
    return mean

def tf_cross_entropy_test(y_true, y_pred):
    cross = tf.losses.softmax_cross_entropy(y_true, y_pred)
    return cross

def tensors():

    # my_values = [0, 1, 3]
    # my_values = range(256)
    my_values = [3, 7]

    true_onehot = np.array([get_plaintext_array(255-i) for i in my_values])

    true = np.array([(i) for i in my_values])
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

    loss1 = tf_median_probability_loss(y_true_onehot, y_pred)
    loss2 = tf.losses.softmax_cross_entropy(y_true_onehot, y_pred)

    # Rank list of predictions
    # argsorted = tf.argsort(pred, direction='DESCENDING')
    # reversed = K.reverse(argsorted, axes=1)
    # ranks =

    # result = argsorted
    # print result

    # with tf.Session() as sess:
    #     print ("\nTrue:\n"
    #     print(sess.run(y_true))
    #     print ("\nPred:\n"
    #     print(sess.run(y_pred))
    #     print ("\nCross Entropy Loss Function:\n"
    #     print(sess.run(loss2))
    #     print ("\nRank Loss Function:\n"
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
    model.compile(loss=tf_median_probability_loss, optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss=tf_cross_entropy_test, optimizer=optimizer, metrics=['accuracy'])


    print ("Model compiled successfully...")

def numpy_tests():

    my_values = range(3) # 256
    true_onehot = np.array([get_plaintext_array(i) for i in my_values])
    pred = np.array([get_hamming_weight_array(get_hw(i)) for i in my_values])

    for i, value in enumerate(zip(true_onehot, pred)):

        print ("\n\n\ni: {}\n".format(i))

        # Step 1: Decode one hot
        decoded_onehot = np.argmax(value[0])
        print ("Decoded One Hot: {}".format(decoded_onehot))

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

    print (out)

def value_occurance_checker(var_name='s', var_num=1, randomkey_extra = False, extra_size = 10000, hw=False):

    print ("* Value Occurance Checker, {} {} *".format(var_name, var_num))

    for extra_file in [False, True]:
        filename = '{}{}{}.npy'.format(REALVALUES_FOLDER, 'extra_' if extra_file and not randomkey_extra else '', var_name)
        real_values = np.load(filename, allow_pickle=True)[var_num-1]
        if randomkey_extra:
            if extra_file:
                real_values = real_values[-extra_size:]
            else:
                real_values = real_values[:-extra_size]

        if hw:
            real_values = linear_get_hamming_weights(real_values)


        unique, counts = np.unique(real_values, return_counts=True)
        if unique.shape[0] != (9 if hw else 256) and not (var_name == 'k' and unique.shape[0] == 1):
            print ("GOTCHA! Unique is only size {}:\n{}\n".format(unique.shape[0], unique))
            raise
        print ("> {} Values ({}):".format('Attack' if extra_file else 'Profile', real_values.shape[0]))
        print_statistics(real_values, mode=True)
        if hw:
            print (">> Counts: {}".format(counts))

def stretch_and_shrink(vector, start, length=300, stretch_length=20):
    section = vector[start:start+length]

    stretched = np.repeat(section[:stretch_length], 2)

    new_section = np.concatenate((stretched, section[stretch_length:]))

    count = -1
    while len(new_section) > len(section):
        new_section = np.delete(new_section, count)
        count -= 1

    return np.concatenate((vector[:start], new_section, vector[start+length:]))

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    print ("TODO: Fix latexify for python3")

    # try:
    #     params = {'backend': 'ps',
    #               'text.latex.preamble': ['\usepackage{gensymb}'],
    #               'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    #               'axes.titlesize': 8,
    #               'text.fontsize': 8, # was 10
    #               'legend.fontsize': 8, # was 10
    #               'xtick.labelsize': 8,
    #               'ytick.labelsize': 8,
    #               'text.usetex': True,
    #               'figure.figsize': [fig_width,fig_height],
    #               'font.family': 'serif'
    #     }
    # except SyntaxError:
    #     print ("Doesn't work with Python3")
    #     exit(1)
    # matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

def dtw_plot(dtw=True):

    start, length = 10000, 500
    trace_data = load_trace_data()[:, start:start+length]

    trace1 = trace_data[0]
    trace2 = stretch_and_shrink(trace_data[1], length / 5)

    f = plt.figure()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(trace1)
    plt.plot(trace2)

    if dtw:
        _, path = fastdtw(trace1, trace2, dist=euclidean)
        for [map_x, map_y] in path:
            plt.plot([map_x, map_y], [trace1[map_x], trace2[map_y]], 'r')

    # plt.title(r'\textbf{Dynamic Time Warping Path}', fontsize=11)
    plt.xlabel(r'\textbf{Samples}', fontsize=11)
    plt.ylabel(r'\textbf{Power Consumption}', fontsize=11)

    plt.show()

    f.savefig("output/dynamic_time_warping_{}.pdf".format('on' if dtw else 'off'), bbox_inches='tight')

def timepoint_plot():
    # f = plt.figure(figsize=(20,5))
    f = plt.figure()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for var, length in variable_dict.iteritems():
        print ("\n* Var {}".format(var))
        timepoint = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var), allow_pickle=True)
        # ax.scatter(var, timepoint, label=var)
        var_l = [var for i in range(len(timepoint))]
        plt.scatter(timepoint, var_l)


    vertical_lines = [0, 4650, 12100, 19690, 23400, 27800, 35000, 42800, 48000]

    for vertical_line in vertical_lines:
        plt.axvline(x=vertical_line, linewidth=0.1)
    # ax.legend()

    plt.xlabel(r'\textbf{Samples}', fontsize=11)
    plt.ylabel(r'\textbf{Variables}', fontsize=11)

    plt.show()

    # f.savefig("output/hw_timepoints.pdf")
    f.savefig("output/identity_timepoints.pdf")

def error_bar():
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

def get_best_templates():

    rtrace_uni = rTraceH.RealTraceHandler(no_print=False, use_nn=False, use_lda=False, use_best=False, tprange=1, jitter=None, use_extra=False)
    rtrace_lda = rTraceH.RealTraceHandler(no_print=False, use_nn=False, use_lda=True, use_best=False, tprange=200, jitter=None, use_extra=False)
    rtrace_nn = rTraceH.RealTraceHandler(no_print=False, use_nn=True, use_lda=False, use_best=False, tprange=2000, jitter=None, use_extra=False) #TODO CHANGE BACK

    template_dict = dict()

    variables = ['{}{}'.format(k, pad_string_zeros(i+1)) for k, v in variable_dict.iteritems() for i in range(v)]

    my_csv = 'output/template_results.csv'
    clear_csv(my_csv)
    append_csv(my_csv, 'Variable,MaxRankUni,MinRankUni,MeanRankUni,MedianRankUni,RangeRankUni,VarianceRankUni,MaxProbUni,MinProbUni,MeanProbUni,MedianProbUni,RangeProbUni,VarianceProbUni,MaxRankLda,MinRankLda,MeanRankLda,MedianRankLda,RangeRankLda,VarianceRankLda,MaxProbLda,MinProbLda,MeanProbLda,MedianProbLda,RangeProbLda,VarianceProbLda,MaxRankNN,MinRankNN,MeanRankNN,MedianRankNN,RangeRankNN,VarianceRankNN,MaxProbNN,MinProbNN,MeanProbNN,MedianProbNN,RangeProbNN,VarianceProbNN,\n')

    for var in variables:

        print ("\n\n\n*** VARIABLE {} ***".format(var))

        ranklist_uni, problist_uni = rtrace_uni.get_performance_of_handler(var)
        ranklist_lda, problist_lda = rtrace_lda.get_performance_of_handler(var)
        ranklist_nn, problist_nn = rtrace_nn.get_performance_of_handler(var)

        print ("\n\n* Ranks *")
        print ("\n> Uni")
        print_statistics(ranklist_uni)
        print ("\n> Lda")
        print_statistics(ranklist_lda)
        print ("\n> NN")
        print_statistics(ranklist_nn)

        print ("\n\n* Probabilities *")
        print ("\n> Uni")
        print_statistics(problist_uni)
        print ("\n> Lda")
        print_statistics(problist_lda)
        print ("\n> NN")
        print_statistics(problist_nn)

        template_dict[var] = dict()
        template_dict[var]['uni'] = (np.median(ranklist_uni), np.median(problist_uni))
        template_dict[var]['lda'] = (np.median(ranklist_lda), np.median(problist_lda))
        template_dict[var]['nn'] = (np.median(ranklist_nn), np.median(problist_nn))



        append_csv(my_csv,'{},{},{},{},{},{},{},\n'.format(var,get_statistics_string(ranklist_uni), get_statistics_string(problist_uni), get_statistics_string(ranklist_lda), get_statistics_string(problist_lda), get_statistics_string(ranklist_nn), get_statistics_string(problist_nn)))

    # Save template
    save_object(template_dict, BEST_TEMPLATE_DICT, suffix=False)

def timepoint_test(tp_threshold = 100):

    count = 0

    for variable, length in variable_dict.iteritems():

        print "* variable {} ({}) *".format(variable, length)

        id_timepoints = np.load('{}{}{}.npy'.format(TRACE_FOLDER, 'identity_timepoints/', variable))
        hw_timepoints = np.load('{}{}{}.npy'.format(TRACE_FOLDER, 'timepoints/', variable))

        for i in range(length):

            id_tp = int(id_timepoints[i])
            hw_tp = int(hw_timepoints[i])
            diff = id_tp - hw_tp

            if abs(id_tp - hw_tp) > tp_threshold:

                print "n: {:3}, diff: {:4} HW {:6} ID {:6}".format(i+1, diff, hw_tp, id_tp)

                count += 1


    print "Total: {}".format(count)


def get_best_models():

    # Load mig vs bcp4 csv
    csv_file = TRACE_FOLDER + 'mig_vs_bcp4.csv'
    current_probability = 0
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            # Get variable, and whether bcp4 or mig
            model_name, median_prob = row[0], row[7]

            try:
                variable = re.search('models\/([a-z]+[0-9]+).*', model_name).group(1)
                source = re.search('.*loss_([a-z]+[0-9]*).h5', model_name).group(1)
                model_name_stripped = re.search('models\/(.*)_.*.h5', model_name).group(1)
                print variable, source, median_prob

                if source == 'bcp4':
                    current_probability = median_prob
                else:
                    model_name_remove = '{}_{}.h5'.format(model_name_stripped, 'mig' if median_prob < current_probability else 'bcp4')
                    if median_prob < current_probability:
                        print "Variable {:5}: REMOVE MIG".format(variable)
                    else:
                        print "Variable {:5}: REMOVE BCP4".format(variable)
                    print "> removing {}".format(model_name_remove)
                    # os.remove(NEURAL_MODEL_FOLDER + model_name_remove)

            except AttributeError:
                print "! Ignoring model '{}'".format(model_name)

def load_new_result(filename, file_prefix = OUTPUT_FOLDER+'new_results/'):
    return np.load('{}{}'.format(file_prefix, filename), allow_pickle=True)

if __name__ == "__main__":

    shifted_l = [2,10,50,100,500,1000]
    extra_l = [False,True]

    for extra in extra_l:
        for shifted in shifted_l:
            print "*** Warping extra {} shift {}".format(extra, shifted)
            realign_traces(extra=extra, shifted=shifted)


    exit(1)

    # get_best_templates()

    # get_best_models()

    # file_prefix = OUTPUT_FOLDER+'new_results/'

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    prefix = 'ranks_NeuralNetworks_IND_'
    classifiers = ['NN2000', 'LDA200', '']
    # graphs = ['G2', 'G1', 'G1A']
    graphs = ['G1']
    midfix = '_KEYAVG_100T_'
    suffix = '_REALNone.npy'

    for classifier in classifiers:
        for graph in graphs:
            plot_name = '{}{}{}{}{}{}{}'.format(prefix, classifier, '_' if classifier is not '' else '', graph, midfix, '100I' if graph is not 'G1A' else '8I', suffix)
            try:
                load_plot = np.mean(load_new_result(plot_name), axis=0)
                plt.plot(load_plot, label='{} {}'.format('Uni' if classifier is '' else classifier, graph))
                np.savetxt('output/bpacomp_{}.csv'.format('Uni' if classifier is '' else classifier, graph), load_plot, delimiter=",")
            except:
                print "!! No plot called {}".format(plot_name)
                raise



    # nn_g2 = load_new_result('ranks_NeuralNetworks_IND_NN2000_G2_KEYAVG_100T_100I_REALNone.npy')
    # nn_g1 = load_new_result('ranks_NeuralNetworks_IND_NN2000_G1_KEYAVG_100T_100I_REALNone.npy')
    # nn_g1a = load_new_result('ranks_NeuralNetworks_IND_NN2000_G1A_KEYAVG_100T_8I_REALNone.npy')
    # lda = load_new_result('ranks_NeuralNetworks_IND_LDA200_G2_KEYAVG_100T_100I_REALNone.npy')
    # uni = load_new_result('ranks_NeuralNetworks_IND_G2_KEYAVG_100T_100I_REALNone.npy')
    #
    # nn_g2 = np.mean(nn_g2, axis=0)
    # nn_g1 = np.mean(nn_g1, axis=0)
    # nn_g1a = np.mean(nn_g1a, axis=0)
    # uni = np.mean(uni, axis=0)
    # lda = np.mean(lda, axis=0)
    #
    # plt.plot(uni, label='Uni BAD G2')
    # plt.plot(lda, label='LDA G2')
    # plt.plot(nn_g2, label='NN G2')
    # plt.plot(nn_g1, label='NN G1')
    # plt.plot(nn_g1a, label='NN G1A')


    plt.xlabel(r'\textbf{Traces}', fontsize=11)
    plt.ylabel(r'\textbf{Mean Rank}', fontsize=11)

    plt.legend()
    plt.show()

    exit(1)

    uni = np.load('{}{}'.format(file_prefix, 'ranks_Standard_IND_G2_KEYAVG_100T_100I_REALNone.npy'), allow_pickle=True)
    lda = np.load('{}{}'.format(file_prefix, 'ranks_Standard_IND_LDA200_G2_KEYAVG_100T_100I_REALNone.npy'), allow_pickle=True)

    uni = np.mean(uni, axis=0)
    lda = np.mean(lda, axis=0)

    np.savetxt('output/uni.csv', uni, delimiter=",")
    np.savetxt('output/lda.csv', lda, delimiter=",")

    # plt.plot(uni, label='uni')
    # plt.plot(lda, label='lda')
    # plt.legend()
    # plt.show()

    exit(1)

    for var, length in variable_dict.iteritems():
        print ("\n* Var {}".format(var))
        for j in range(length):
            timepoint = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var), allow_pickle=True)[j]
            hw_timepoint = np.argmax(np.load("{}{}_{}_HW.npy".format(COEFFICIENT_FOLDER, var, j), allow_pickle=True))
            print ("TP: {:6} HWTP: {:6} RANGE: {:6}".format(timepoint, hw_timepoint, np.abs(timepoint - hw_timepoint)))
