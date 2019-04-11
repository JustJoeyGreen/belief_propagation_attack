from os import listdir
from os.path import isfile, join, expanduser
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utility import *
from random import shuffle
import realTraceHandler as rTH


###########################################################################

###########################################################################

class TestModels:

    def __init__(self, jitter=None, use_extra = True, no_print=True):
        # Real Trace Handler
        self.real_trace_handler = rTH.RealTraceHandler(no_print = no_print, use_nn = True, use_lda = False, memory_mapped=True, tprange=700, debug=True, jitter=jitter, use_extra = use_extra)


    AES_Sbox = np.array([
                0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
                0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
                0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
                0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
                0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
                0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
                0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
                0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
                0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
                0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
                0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
                0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
                0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
                0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
                0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
                0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
                ])


    # Compute the rank of the real key for a give set of predictions
    def rank(self, predictions, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba):
        # Compute the rank
        if len(last_key_bytes_proba) == 0:
            # If this is the first rank we compute, initialize all the estimates to zero
            key_bytes_proba = np.zeros(256)
        else:
            # This is not the first rank we compute: we optimize things by using the
            # previous computations to save time!
            key_bytes_proba = last_key_bytes_proba

        for p in range(0, max_trace_idx-min_trace_idx):
            # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.

            # plaintext = metadata[min_trace_idx + p]['plaintext'][2]
            plaintext = np.load('{}p.npy'.format(REALVALUES_FOLDER))[2,p]

            for i in range(0, 256):
                # Our candidate key byte probability is the sum of the predictions logs
                proba = predictions[p][AES_Sbox[plaintext ^ i]]
                if proba != 0:
                    key_bytes_proba[i] += np.log(proba)
                else:
                    # We do not want an -inf here, put a very small epsilon
                    # that correspondis to a power of our min non zero proba
                    min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                    if len(min_proba_predictions) == 0:
                        print("Error: got a prediction with only zeroes ... this should not happen!")
                        sys.exit(-1)
                    min_proba = min(min_proba_predictions)
                    key_bytes_proba[i] += np.log(min_proba**2)
        # Now we find where our real key candidate lies in the estimation.
        # We do this by sorting our estimates and find the rank in the sorted array.
        sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
        real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
        return (real_key_rank, key_bytes_proba)

    def full_ranks(self, model_name, model, dataset, labels, min_trace_idx, max_trace_idx, rank_step, template_attack = False, model_variable='s001'):
        # Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.

        # real_key = metadata[0]['key'][2]
        var_name, var_number, _ = split_variable_name(model_variable)
        real_key = np.load('{}extra_k.npy'.format(REALVALUES_FOLDER))[var_number-1]
        real_p = np.load('{}extra_p.npy'.format(REALVALUES_FOLDER))[var_number-1]

        print 'Real Key: {}\nReal Ptx: {}\n'.format(real_key, real_p)


        # Check for overflow
        if max_trace_idx > dataset.shape[0]:
            print("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))
            sys.exit(-1)
        # Get the input layer shape
        input_layer_shape = model.get_layer(index=0).input_shape
        # Sanity check
        if input_layer_shape[1] != len(dataset[0, :]):
            print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(dataset[0, :])))
            sys.exit(-1)
        # Adapt the data shape according our model input
        if len(input_layer_shape) == 2:
            # This is a MLP
            input_data = dataset[min_trace_idx:max_trace_idx, :]
        elif len(input_layer_shape) == 3:
            # This is a CNN: reshape the data
            input_data = dataset[min_trace_idx:max_trace_idx, :]
            input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
        else:
            print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
            sys.exit(-1)

        predictions = model.predict(input_data) #TODO: Use this!!

        if len(np.unique(predictions, axis=0)) < 3:
            # print len(np.unique(predictions, axis=0))
            print "! All attack traces predicted same value {}".format(get_value_from_plaintext_array(predictions[0]))
        else:

            # Loop and check
            total_ranks = np.empty(len(predictions))
            probability_distributions = np.ones((len(predictions), 256))

            for i, prediction_vector in enumerate(predictions):
                try:
                    predicted = np.argmax(prediction_vector)
                except IndexError:
                    print "! Error: can't get argmax of {}".format(predicted)
                    exit(1)
                # Get rank

                if template_attack:

                    if var_name not in ['k','t','s']:
                        print "! Error: Can't perform Template Attack on unkown varname {}".format(var_name)
                        raise
                    # if s001
                    temp = prediction_vector

                    if var_name == 's':
                        temp = array_inv_sbox(temp)
                    if var_name == 's' or var_name == 't':
                        temp = array_xor_permutate(temp, real_p[i])

                    probability_distributions[i] = temp
                    total_ranks[i] = get_rank_from_prob_dist(probability_distributions[i], real_key[i])

                    # print 'i: {}, real key {}, real plaintext {}\nranked dist: {}\n'.format(i, real_key[i], real_p[i], zip(range(256), get_rank_list_from_prob_dist(probability_distributions[i])))
                    # exit(1)
                    # rank_after_traces[i] = get_rank_from_prob_dist(np.prod(probability_distributions, axis=0), real_key[i])

                else:
                    correct_val = labels[i] # EDIT
                    rank = get_rank_from_prob_dist(prediction_vector, correct_val)
                    # if i == 0:
                    #     print "Trace {:8}  Label: {:3}  Predicted: {:3}  Rank: {}".format(i, correct_val, predicted, rank)
                    #     print prediction_vector
                    total_ranks[i] = rank
                    # predictions[i][predicted], predictions[i][labels[i]]

                    # if i <= 1:
                    #     print "Correct Val: {}, Rank: {}, Probability: {}, Highest Probability: {} (index {})\n\n{}".format(correct_val, rank, prediction_vector[correct_val], np.max(prediction_vector), np.argmax(prediction_vector), prediction_vector)

            if template_attack:

                max_traces_required = 5000
                max_repeats = 1000
                consistency_max = 10
                traces_required = list()
                failed_traces = 0
                for repeat in range(max_repeats):

                    consistency_checker = 0
                    solved_placeholder = 0

                    # Get random order
                    order = range(len(predictions))
                    shuffle(order)

                    for i, t in enumerate(order):
                        probdist = probability_distributions[t]
                        if i == 0:
                            current_probdist = np.log(probdist)
                        else:
                            current_probdist += np.log(probdist)

                        current_rank = get_rank_from_prob_dist(current_probdist, real_key[0])
                        # print "i: {:3}  t: {:3}  rank: {}".format(i, t, current_rank)

                        if current_rank == 1:

                            if consistency_checker == consistency_max:
                                # print "Repeat {}, Required {}".format(repeat, solved_placeholder)
                                traces_required.append(solved_placeholder)
                                break
                            elif consistency_checker == 0:
                                solved_placeholder = i + 1
                                consistency_checker += 1
                            else:
                                consistency_checker += 1
                        else:
                            consistency_checker = 0

                        if i > max_traces_required:
                            # print "! Repeat {} Took over 1000 Traces, currently rank {}".format(repeat, current_rank)
                            failed_traces += 1
                            break

                print "* Percentage Success within {} traces: {}%".format(max_traces_required, ((max_repeats - failed_traces) * 100) / (max_repeats + 0.0))
                print "* Traces Required for First Order Template Attack *"
                print_statistics(traces_required)

            else:
                print "* Printing Classification Rank Statistics *"
                print_statistics(total_ranks, top=True)
                if SAVE:
                    save_statistics(model_name, total_ranks)




            # # Get average rank
            # print "* Predition Statistics *"
            # print_statistics(total_ranks)
            #
            # # current_probdist = [1] * 256
            # # rank_after_traces = [256] * len(predictions)
            # # for i, probdist in enumerate(probability_distributions):
            # #     # current_probdist = normalise_array(current_probdist * normalise_array(probdist))
            # #     temp_probdist = [a*b for a,b in zip(current_probdist, probdist)]
            # #     if np.sum(temp_probdist) < 1e-50:
            # #         temp_probdist = [a * b for a, b in zip(normalise_array(np.array(current_probdist)).tolist(), probdist)]
            # #     current_probdist = temp_probdist
            # #     rank_after_traces[i] = get_rank_from_prob_dist(np.array(current_probdist), real_key[0])
            #
            # # LOG PROBABILITIES
            # rank_after_traces = [256] * len(predictions)
            # for i, probdist in enumerate(probability_distributions):
            #     if i == 0:
            #         current_probdist = np.log(probdist)
            #     else:
            #         current_probdist += np.log(probdist)
            #
            #     rank_after_traces[i] = get_rank_from_prob_dist(current_probdist, real_key[0])

            # print "* Rank after Traces *"
            # print rank_after_traces
            # print_statistics(rank_after_traces)

            # plt.title(VARIABLE + ' performance')
            # plt.xlabel('number of traces')
            # plt.ylabel('rank')
            # plt.grid(True)
            # plt.plot(rank_after_traces)
            # plt.show(block=False)

            # print "* Multiplying All Probabilities: *"
            # mult = np.prod(probability_distributions, axis=0)
            # print mult
            # print "Rank", get_rank_from_prob_dist(mult, real_key[0])
            # print "* Averaging All Probabilities: *"
            # av = np.mean(probability_distributions, axis=0)
            # print "Rank", get_rank_from_prob_dist(av, real_key[0])


        # index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
        # f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
        # key_bytes_proba = []
        # for t, i in zip(index, range(0, len(index))):
        #     real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], real_key, t-rank_step, t, key_bytes_proba)
        #     f_ranks[i] = [t - min_trace_idx, real_key_rank]
        # return f_ranks

    # Check a saved model against one of the bpann databases Attack traces
    def check_model(self, model_file, num_traces=10000, template_attack=False, random_key=False):

        print "Check model, random key {}".format(random_key)
        try:
            rank_list, predicted_values = self.real_trace_handler.get_leakage_rank_list_with_specific_model(model_file, traces=num_traces, from_end=random_key)
            if rank_list is not None:
                print "\n\nModel: {}".format(model_file)
                print_statistics(rank_list, mode=False)
                print "> Top Predicted Indices:"
                print_statistics(predicted_values)
        except Exception as e:
            print "! Uh oh, couldn't check the model! Need to resubmit (in test_models)" #PASSING OVER..."
            print e

        # model_name = model_file.replace(MODEL_FOLDER, '')
        # model_variable = model_name.split('_')[0]
        #
        # print "\n* Checking model {} (variable {}) *\n".format(model_name, model_variable)
        #
        # check_file_exists(model_file)
        # # Load profiling and attack data and metadata from the bpann database
        #
        #
        #
        # # Get inputs size from file name!
        # input_length = get_window_size_from_model(model_file)
        #
        # (_, _), (X_attack, Y_attack) = load_bpann(model_variable, input_length=input_length)
        # # print 'X_attack for variable {}:\n{}\n\n'.format(model_variable, X_attack)
        #
        # # Load model
        # model = load_sca_model(model_file)
        # # We test the rank over traces of the Attack dataset, with a step of 10 traces
        # ranks = full_ranks(model_name, model, X_attack, Y_attack, 0, num_traces, 10, template_attack=template_attack, model_variable=model_variable)






        # # We plot the results
        # x = [ranks[i][0] for i in range(0, ranks.shape[0])]
        # y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        # plt.title(VARIABLE + ' Performance of '+model_file)
        # plt.xlabel('number of traces')
        # plt.ylabel('rank')
        # plt.grid(True)
        # plt.plot(x, y)
        # plt.show(block=False)
        # plt.figure()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--ALL', '--ALL_VARS', '--TEST_ALL', action="store_true", dest="TEST_ALL", help='Tests all available models',
                        default=False)
    parser.add_argument('--MLP', action="store_true", dest="USE_MLP", help='Tests Multi Layer Perceptron',
                        default=False)
    parser.add_argument('--CNN', action="store_true", dest="USE_CNN",
                        help='Tests Convolutional Neural Network', default=False)
    parser.add_argument('--N', '--NOISE', action="store_true", dest="ADD_NOISE",
                        help='Adds noise to the profiling step', default=False)
    parser.add_argument('-v', '-var', '-variable', action="store", dest="VARIABLE", help='Variable to train',
                        default='s001')
    parser.add_argument('-l', '-length', '-input', action="store", dest="INPUT_LENGTH",
                        help='Input Length (default: 700)',
                        type=int, default=700)
    parser.add_argument('-t', '-traces', '-test_traces', action="store", dest="TEST_TRACES",
                        help='Number of Traces to Test with (default 10000)',
                        type=int, default=10000)
    parser.add_argument('--S', '--SAVE', action="store_true", dest="SAVE",
                        help='Saves Output', default=False)
    parser.add_argument('--T', '--TEMPLATE', '--TEMPLATE_ATTACK', action="store_true", dest="TEMPLATE_ATTACK",
                        help='Performs Template Attack on Data', default=False)
    parser.add_argument('-j', '-jitter', action="store", dest="JITTER",
                        help='Clock Jitter to use on real traces (default: None)',
                        type=int, default=None)
    parser.add_argument('--E', '--EX', '--EXTRA', '--USE_EXTRA', action="store_false", dest="USE_EXTRA",
                        help='Toggle to Turn USE EXTRA Off (Attack Trained Traces)', default=True)

    parser.add_argument('--RK', '--RKV', '--RANDOMKEY', action="store_true", dest="RANDOM_KEY",
                        help='Toggle to Turn RANDOM_KEY On (Attack Validation Traces)', default=False)

    parser.add_argument('--D', '--DEBUG', action="store_true", dest="DEBUG",
                        help='Turns no_print off', default=False)

    # Target node here
    args = parser.parse_args()
    USE_MLP = args.USE_MLP
    USE_CNN = args.USE_CNN
    VARIABLE = args.VARIABLE
    ADD_NOISE = args.ADD_NOISE
    INPUT_LENGTH = args.INPUT_LENGTH
    TEST_TRACES = args.TEST_TRACES
    TEST_ALL = args.TEST_ALL
    SAVE = args.SAVE
    TEMPLATE_ATTACK = args.TEMPLATE_ATTACK
    JITTER = args.JITTER
    USE_EXTRA = args.USE_EXTRA
    RANDOM_KEY = args.RANDOM_KEY
    DEBUG = args.DEBUG

    # var_list = list()
    # for v in variable_dict:
    #     var_list.append('{}001'.format(v))
    #
    # for i in range(variable_dict['s']):
    #     var_list.append('s{}'.format(pad_string_zeros(i+1)))

    # print "*** TEST VARIABLE {} ***".format(VARIABLE)

    model_tester = TestModels(jitter=JITTER, use_extra=(not RANDOM_KEY) and USE_EXTRA, no_print=not DEBUG)

    if TEST_ALL:
        # Clear statistics
        if SAVE:
            clear_statistics()
        # Check all models
        for (m) in sorted(listdir(MODEL_FOLDER)):
            if string_ends_with(m, '.h5'):
                # print 'm: {}'.format(m)
                model_tester.check_model(MODEL_FOLDER + m, TEST_TRACES, template_attack=TEMPLATE_ATTACK, random_key=RANDOM_KEY)
    else:
        # Check specific model
        # TODO
        print "Todo: Check specific model"
        pass

# # No argument: check all the trained models
# if (len(sys.argv) == 1) or (len(sys.argv) == 2):
#     if len(sys.argv) == 2:
#         num_traces = int(sys.argv[1])
#     else:
#         num_traces = 2000
#
#     for (m) in to_check_all:
#         check_model(m, num_traces)
#
# else:
#     if len(sys.argv) == 4:
#         num_traces = int(sys.argv[3])
#     else:
#         num_traces = 2000
#
#     check_model(sys.argv[1], sys.argv[2], num_traces)

# try:
#     input("Press enter to exit ...")
# except SyntaxError:
#     pass
