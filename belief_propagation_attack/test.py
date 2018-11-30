import leakageSimulatorAESFurious as lSimF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis
from utility import *
import realTraceHandler as rTH


# variable_list = ['s{}'.format(pad_string_zeros(i)) for i in range(1,17)]
# traces = 10000
# lda = True
#
# rth = rTH.RealTraceHandler(no_print = False, use_nn = False, use_lda = lda, memory_mapped=True, nn_window = 700, lda_window = 200, debug=True)
#
# for var in variable_list:
#     print "\n\n* Variable {}".format(var)
#     print_statistics(rth.get_leakage_rank_list(var, traces=traces))

profile_traces = 200000
attack_traces = 100
variable = 's001'
window = 200

for normalise in [False, True]:

    var_name, var_number, _ = split_variable_name(variable)
    time_point = np.load('{}{}.npy'.format(TIMEPOINTS_FOLDER, var_name))[var_number-1]
    profile_trace_data = load_trace_data(filepath=TRACEDATA_FILEPATH, memory_mapped=True)[:profile_traces,time_point-(window/2):time_point+(window/2)]
    attack_trace_data = normalise_neural_traces(load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH, memory_mapped=True)[:attack_traces,time_point-(window/2):time_point+(window/2)])

    if normalise:
        profile_trace_data = normalise_neural_traces(profile_trace_data)
        attack_trace_data = normalise_neural_traces(attack_trace_data)

    profile_labels = np.load('{}{}.npy'.format(REALVALUES_FOLDER, var_name))[var_number-1][:profile_traces]
    attack_labels = np.load('{}extra_{}.npy'.format(REALVALUES_FOLDER, var_name))[var_number-1][:attack_traces]

    # print profile_trace_data.shape, attack_trace_data.shape, profile_labels.shape, attack_labels.shape
    lda = linDisAnalysis()

    lda.fit(profile_trace_data,profile_labels)

    predictions = lda.predict_proba(attack_trace_data)
    my_rank_list = list()

    for i in range(attack_traces):
        rank_list = get_rank_list_from_prob_dist(predictions[i])
        print "\nTrace {}. Correct label for {}: {}".format(i, variable, attack_labels[i])
        # print "Predicted:\n{}\n".format(predictions[0])
        # print "Rank List:\n{}\n".format(rank_list)
        print "Rank: {}, Probability: {}, Max: {}".format(rank_list[attack_labels[i]], predictions[i][attack_labels[i]], np.max(predictions[i]))
        my_rank_list.append(rank_list[attack_labels[i]])
    print "\nStatistics"
    print_statistics(my_rank_list)
