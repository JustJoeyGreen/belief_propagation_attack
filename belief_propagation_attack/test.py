import leakageSimulatorAESFurious as lSimF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis
from utility import *
import realTraceHandler as rTH
import argparse

parser = argparse.ArgumentParser(description='Trains Neural Network Models')
parser.add_argument('-t', '-traces', '-test_traces', action="store", dest="TRACES",
                    help='Number of Traces to Test with (default 10000)',
                    type=int, default=10000)
args = parser.parse_args()
TRACES = args.TRACES

# USE_REAL_TRACE_HANDLER = True
USE_REAL_TRACE_HANDLER = True

TEST = False
if TEST:
    todo = list()
    for v, length in variable_dict.iteritems():
        for i in range(length):
            variable = '{}{}'.format(v, pad_string_zeros(i+1))
            if not check_file_exists('{}{}_mlp5_nodes200_window{}_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, variable, 700)):
                todo.append(variable)
    print len(todo), todo
    exit(1)

if USE_REAL_TRACE_HANDLER:

    # variable_list = ['k001', 'k002', 'p001', 'p002', 't001', 't002', 's001', 's002']
    # variable_list = ['k001', 's001']
    variable_list = ['k001']
    traces = TRACES
    nn = True
    lda = False
    print "* LDA {}".format(lda)

    rth = rTH.RealTraceHandler(no_print = False, use_nn = nn, use_lda = lda, memory_mapped=True, tprange=200 if lda else 700, debug=True)

    for var in variable_list:
        print "\n\n* Variable {}".format(var)
        rank_list = rth.get_leakage_rank_list(var, traces=traces)
        print_statistics(rank_list)

else:

    profile_traces = 200000
    attack_traces = 10000
    # variable = 'k001'
    variables = ['k001', 'k002', 'p001', 'p002', 't001', 't002', 's001', 's002']
    # window = 200
    normalise = False
    # windows = [2, 10, 20, 50, 100, 200, 500, 1000]
    window = 200
    one_take = False
    baseline = 0.00390625

    # profile_trace_data = load_trace_data(filepath=TRACEDATA_FILEPATH, memory_mapped=True)[:profile_traces]
    attack_trace_data = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH, memory_mapped=True)[:attack_traces]

    for variable in variables:

        var_name, var_number, _ = split_variable_name(variable)

        profile_labels = np.load('{}{}.npy'.format(REALVALUES_FOLDER, var_name))[var_number-1][:profile_traces]
        attack_labels = np.load('{}extra_{}.npy'.format(REALVALUES_FOLDER, var_name))[var_number-1][:attack_traces]

        time_point = np.load('{}{}.npy'.format(TIMEPOINTS_FOLDER, var_name))[var_number-1]

        print "* Var Name {}".format(variable)

        # my_profile_trace_data = profile_trace_data[:,time_point-(window/2):time_point+(window/2)]
        my_attack_trace_data = attack_trace_data[:,time_point-(window/2):time_point+(window/2)]

        if normalise:
            my_profile_trace_data = normalise_neural_traces(my_profile_trace_data)
            my_attack_trace_data = normalise_neural_traces(my_attack_trace_data)

        # print profile_trace_data.shape, attack_trace_data.shape, profile_labels.shape, attack_labels.shape
        # lda = linDisAnalysis()

        # lda.fit(my_profile_trace_data, profile_labels)

        # print 'Params: {}, Score: {}'.format(lda.get_params(), lda.score(my_attack_trace_data, attack_labels))

        lda = pickle.load(open('{}{}_{}_{}.p'.format(LDA_FOLDER,
            window, var_name, var_number-1),'ro'))

        # print "var_name {} var_number {} traces {}".format(var_name, var_number, attack_traces)#debug

        predictions = lda.predict_proba(my_attack_trace_data)
        my_rank_list = list()

        for i in range(attack_traces):
            # my_predictions = normalise_array(1 - predictions[i])
            my_predictions = predictions[i]
            if one_take:
                my_predictions = 1 - my_predictions
            # my_predictions = normalise_array(my_predictions)
            rank = get_rank_from_prob_dist(my_predictions, attack_labels[i])

            # print "trace {} real_val {} rank {}".format(i, attack_labels[i], rank)
            # exit()


            # print "\nTrace {}. Correct label for {}: {}".format(i, variable, attack_labels[i])
            # print "Predicted:\n{}\n".format(predictions[0])
            # print "Rank List:\n{}\n".format(rank_list)
            # print "Rank: {}, Probability: {}, Max: {}".format(rank_list[attack_labels[i]], my_predictions[attack_labels[i]], np.max(my_predictions))
            if i == 0:
                my_p = my_predictions[attack_labels[i]]
                print "Sample Prediction: {} ({} from baseline)".format(my_p, np.abs(my_p - baseline))

            my_rank_list.append(rank)
        print "\nStatistics"
        print_statistics(my_rank_list)
