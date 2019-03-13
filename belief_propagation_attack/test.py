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

parser = argparse.ArgumentParser(description='Trains Neural Network Models')
parser.add_argument('-t', '-traces', '-test_traces', action="store", dest="TRACES",
                    help='Number of Traces to Test with (default 10000)',
                    type=int, default=10000)
args = parser.parse_args()
TRACES = args.TRACES

for var, length in variable_dict.iteritems():
    print "\n* Var {}".format(var)
    for j in range(length):
        timepoint = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var))[j]
        hw_timepoint = np.argmax(np.load("{}{}_{}_HW.npy".format(COEFFICIENT_FOLDER, var, j)))
        print "TP: {:6} HWTP: {:6} RANGE: {:6}".format(timepoint, hw_timepoint, np.abs(timepoint - hw_timepoint))

exit(1)

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

# # IAVG G2 100T 50I
# names = ['UNI', 'LDA', 'NN', 'IGB', 'BEST']
# # names = ['LDA']
# filepath = PATH_TO_TRACES + 'Results/'
#
# for name in names:
#     a = np.mean(np.load('{}{}.npy'.format(filepath, name)), axis=0)
#     # np.savetxt("{}{}.csv".format(filepath, name), a, delimiter=",")
#     plt.plot(a, label=name)
#
# plt.legend()
# plt.show()
#
# exit(1)
#
#
# best_templates = pickle.load(open(BEST_TEMPLATE_DICT,'ro'))
#
# csv_filepath = 'output/best_templates.csv'
#
# clear_csv(csv_filepath)
# out = 'Variable,Univariate,LDA,Neural Network\n'
# for variable, ranks in best_templates.iteritems():
#     out += '{},'.format(variable)
#     for method, rank in ranks.iteritems():
#         out += '{},'.format(rank)
#     out += '\n'
#
# append_csv(csv_filepath, out)
#
# # print type(best_templates)
# # print best_templates
#
#
# exit(1)
#
# for shifted in [2,10,50,100,500,1000]:
#     for extra in [True]:
#         realign_traces(extra=extra, shifted=shifted)
#
# print "All done!"
# exit(1)


my_graph = fG.FactorGraphAES()

print get_all_variables_that_match(my_graph.get_variables(), ['k001'])

exit(1)

my_length = 3
plot_start = 2000
plot_length = 1000

non_jitter = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH)[:my_length]
jitter = load_trace_data(filepath=get_shifted_tracedata_filepath(extra=True, shifted=1000))[:my_length]
realigned = [0] * my_length
for i in range(0,my_length):
    realigned[i] = realign_trace(non_jitter[0], jitter[i])
realigned = np.array(realigned)

print 'non_jitter:\n{}, {}'.format(non_jitter[-1][:10], non_jitter[-1][-10:])
print 'jitter:\n{}, {}'.format(jitter[-1][:10], jitter[-1][-10:])
print 'realigned:\n{}, {}'.format(realigned[-1][:10], realigned[-1][-10:])

plt.subplot(1, my_length, 1)
for i in range(my_length):
    plt.plot(non_jitter[i][plot_start:plot_start+plot_length])
plt.subplot(1, my_length, 2)
for i in range(my_length):
    plt.plot(jitter[i][plot_start:plot_start+plot_length])
plt.subplot(1, my_length, 3)
for i in range(my_length):
    plt.plot(realigned[i][plot_start:plot_start+plot_length])
plt.show()

exit(1)





# for shift_val in [2,10,50,100,500,1000]:
#     for shift_extra in [False, True]:
#         shift_traces(extra=shift_extra, shifted=shift_val)

a = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH)

for shift_val in [2,10,50,100,500,1000]:
    b = load_trace_data(filepath=get_shifted_tracedata_filepath(extra=True, shifted=shift_val))
    plt.subplot(1, 2, 1)
    for i in range(20):
        print 'A {}: {}'.format(i, a[i][1000:1010])
        plt.plot(a[i][1000:2000])
    plt.subplot(1, 2, 2)
    for i in range(20):
        print 'B {}: {}'.format(i, b[i][1000:1010])
        plt.plot(b[i][1000:2000])
    plt.title('Shift Val {}'.format(shift_val))
    plt.show()
exit(1)


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

    rth_nn = rTH.RealTraceHandler(no_print = False, use_nn = True, memory_mapped=True, tprange=700, debug=True)
    for var in ['k00{}'.format(i) for i in range(1,10)]:
        rth_nn.get_leakage(var, ignore_bad=True)
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
