import leakageSimulatorAESFurious as lSimF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis
from utility import *
import realTraceHandler as rTH
import argparse
import matplotlib.pyplot as plt
import trsfile
import operator


parser = argparse.ArgumentParser(description='Trains Neural Network Models')
parser.add_argument('-t', '-traces', '-test_traces', action="store", dest="TRACES",
                    help='Number of Traces to Test with (default 10000)',
                    type=int, default=10000)
args = parser.parse_args()
TRACES = args.TRACES

TEST = False

# USE_REAL_TRACE_HANDLER = True
USE_REAL_TRACE_HANDLER = True

# a = load_trace_data()
# for i in range(20):
#     plt.plot(a[i,1000:2000])
# plt.show()
# exit(1)


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

    traces = TRACES

    rth_best = rTH.RealTraceHandler(no_print = False, use_best = True, memory_mapped=True, tprange=700, debug=True)
    for var in ['k00{}'.format(i) for i in range(1,10)]:
        rth_best.get_leakage(var)
    exit(1)
    # rth_none = rTH.RealTraceHandler(no_print = False, use_nn = False, use_lda = False, memory_mapped=True, tprange=1, debug=True)
    # rth_lda = rTH.RealTraceHandler(no_print = False, use_nn = False, use_lda = True, memory_mapped=True, tprange=200, debug=True)
    # rth_nn = rTH.RealTraceHandler(no_print = False, use_nn = True, use_lda = False, memory_mapped=True, tprange=700, debug=True)
    # csv_file = 'output/variable_template_comparison.csv'
    # dict_file = 'output/best_template_dict.dict'
    # clear_csv(csv_file)
    # append_csv(csv_file, 'VarName,VarNumber,TopTimepoint,TopCoefficient,AriMean,Median,SensibleTimepoint,SensibleCoefficient,Ranked,CoefficientDifference,\n')
    # append_csv(csv_file, 'Variable Name,Variable Number,Template Mean Rank,Template Median Rank,LDA Mean Rank,LDA Median Rank,Neural Mean Rank,Neural Median Rank,\n')
    template_dict = {}
    # for var_name in ['k','t','s']:
    # for var_name in ['k']:
    for var_name, length in variable_dict.iteritems():

        # Assume first timepoint is sensible?
        # current_timepoint = -1
        # tp_window = 300
        # for var_number in range(1, 17):
        for var_number in range(1, length+1):
            var = '{}{}'.format(var_name, pad_string_zeros(var_number))
            print "\n\n* Variable {} {}".format(var_name, var_number)

            rank_list_none = np.array(rth_none.get_leakage_rank_list(var, traces=traces, invert=False))
            rank_list_lda = np.array(rth_lda.get_leakage_rank_list(var, traces=traces, invert=False))
            rank_list_nn = np.array(rth_nn.get_leakage_rank_list(var, traces=traces, invert=False))

            # append_csv(csv_file, '{},{},{},{},{},{},{},{},\n'.format(var_name, var_number, get_average(rank_list_none), array_median(rank_list_none), get_average(rank_list_lda), array_median(rank_list_lda), get_average(rank_list_nn), array_median(rank_list_nn)))

            template_dict[var] = {'uni':array_median(rank_list_none),'lda':array_median(rank_list_lda),'nn':array_median(rank_list_nn)}

            if var_number == 1:
                print "* Standard Template"
                print_statistics(rank_list_none)
                print "* LDA 200"
                print_statistics(rank_list_lda)
                print "* Neural 700"
                print_statistics(rank_list_nn)

            #
            # print "+ Non-inverted"
            # print_statistics(rank_list)
            # print "- Inverted (1-leakage)"
            # print_statistics(rank_list_inverted)

            # append_csv(csv_file, '{},{},{},{},{},{},{},{},{},{},\n'.format(var_name, var_number, time_point, coeff_array[time_point], get_average(rank_list), array_median(rank_list), corrected, coeff_array[corrected], rank, coeff_array[time_point]-coeff_array[corrected]))

    print template_dict

    # pickle.dump(template_dict, open(dict_file, 'wb'))
    # exit(1) #todo

    # # variable_list = ['k001', 'k002', 'p001', 'p002', 't001', 't002', 's001', 's002']
    # # variable_list = ['k001', 's001']
    # variable_list = ['k{}'.format(pad_string_zeros(i+1)) for i in range(16)]
    # # variable_list = ['k001', 'k002', 'k003']
    # # variable_list = ['k004']
    #
    # for var in variable_list:
    #     print "\n\n* Variable {}".format(var)
    #     var_name, var_number, _ = split_variable_name(var)
    #     time_point = np.load('{}{}.npy'.format(TIMEPOINTS_FOLDER, var_name))[var_number-1]
    #     print "Time Point {}".format(time_point)
    #     coeff_array = np.load('{}{}_{}.npy'.format(COEFFICIENT_FOLDER, var_name, var_number-1))
    #     TOP_N = 100
    #     top_n = coeff_array.argsort()[-TOP_N:][::-1]
    #     corrected = -1
    #     for c, index in enumerate(top_n):
    #         # print "Rank {}: Time Point {} ({})".format(c, index, coeff_array[index])
    #         if index < k_max and index > k_max - (tp_window):
    #             corrected = index
    #             break
    #     print "Old Time Point {}, Corrected: {}".format(time_point, corrected)
    #     if corrected != -1:
    #         k_max = corrected
    #     rank_list = rth.get_leakage_rank_list(var, traces=traces)
    #     print_statistics(rank_list)


    exit(1)

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
