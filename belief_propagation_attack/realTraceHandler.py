import numpy as np
import linecache
from utility import *
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis
import operator

KEY = [0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75]
TPRANGE_NN = 2000 #700
TPRANGE_LDA = 200
REALIGN_OVERHEAD = 500

class RealTraceHandler:

    def __init__(self, no_print = False, use_best = False, use_nn = False, use_lda = False, memory_mapped=True, tprange=200, debug=True, jitter = None, use_extra = True, auto_realign = False):
        self.no_print = no_print
        if not no_print:
            print "Preloading Matrix real_trace_data, may take a while..."

        self.use_extra = use_extra
        self.auto_realign = auto_realign


        if use_extra:
            self.real_trace_data = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH if jitter is None else get_shifted_tracedata_filepath(extra=True, shifted=jitter), memory_mapped=memory_mapped)
            self.plaintexts = np.load(PLAINTEXT_EXTRA_FILEPATH)
            if debug:
                self.realvalues = dict()
                for var in variable_dict:
                    self.realvalues[var] = np.load('{}extra_{}.npy'.format(REALVALUES_FOLDER, var))
        else:
            self.real_trace_data = load_trace_data(filepath=TRACEDATA_FILEPATH if jitter is None else get_shifted_tracedata_filepath(extra=False, shifted=jitter), memory_mapped=memory_mapped)
            self.plaintexts = np.load(PLAINTEXT_FILEPATH)
            if debug:
                self.realvalues = dict()
                for var in variable_dict:
                    self.realvalues[var] = np.load('{}{}.npy'.format(REALVALUES_FOLDER, var))

        self.real_trace_data_maxtraces, self.real_trace_data_len = self.real_trace_data.shape


        if not no_print:
            print "Preloading all timepoints, may take a while..."
        self.timepoints = dict()
        for var in variable_dict:
            self.timepoints[var] = np.load('{}{}.npy'.format(TIMEPOINTS_FOLDER, var))



        self.use_best = use_best
        self.use_lda = use_lda
        self.use_nn = use_nn
        self.tprange = tprange
        self.neural_network_dict = dict()
        self.lda_dict = dict()
        # TODO Fix Power Value File, it's totally wrong
        # if not no_print:
        #     print "Preloading all power values, may take a while..."
        # self.powervalues = dict()
        # for var in variable_dict:
        #     self.powervalues[var] = np.load('{}extra_{}.npy'.format(POWERVALUES_FOLDER, var))
        self.musigma_dict = pickle.load(open(MUSIGMA_FILEPATH, 'ro'))

        # TODO: CURRENTLY IN DEVELOPMENT
        self.best_templates = pickle.load(open(BEST_TEMPLATE_DICT,'ro'))

    def return_power_window(self, timepoint, trace, window=2000, nn_normalise=False):
        """ Return the window of power values for a given value """
        trace_data = normalise_neural_trace_single(self.real_trace_data[trace]) if nn_normalise else self.real_trace_data[trace]
        start_window, end_window = handle_window(timepoint, window, 0, trace_data.shape[0] - 1)
        return trace_data[start_window:end_window]

    def return_power_window_of_variable(self, variable, trace, window=2000, nn_normalise=False):
        var_name, var_number, _ = split_variable_name(variable)
        return self.return_power_window(self.timepoints[var_name][var_number-1], trace, window=window, nn_normalise=nn_normalise)


    def get_best_template(self, variable, rank=True):
        templates = self.best_templates[strip_off_trace(variable)]
        comp = 256 if rank else 0
        comp_type = 'uni'
        for type, template in templates.iteritems():
            if (rank and template[0] < comp) or (not rank and template[1] > comp):
                comp = template[0] if rank else template[1]
                comp_type = type
        return comp_type
        # return min(templates.iteritems(), key=operator.itemgetter(1))[0]

    def check_template(self, variable, method='nn', rank=True, rank_threshold=128, prob_threshold=0.001):
        template = self.best_templates[strip_off_trace(variable)][method]
        # print "+ Checking template for {}, median rank {}".format(variable, median_rank)
        return (rank and template[0] < rank_threshold) or (not rank and template[1] > prob_threshold)


    def get_leakage_value(self, variable, trace=0, average_power_values=False, averaged_traces=1, auto_realign=True):
        # myvarlist = ["k001-K", "k001"]
        # if variable in myvarlist:
            # print "Getting Leakage for {}, trace {}".format(variable, trace)
        tprange = self.tprange
        best = None
        if self.use_best:
            best = self.get_best_template(variable)
            if best == 'uni':
                tprange = 1
            elif best == 'lda':
                tprange = TPRANGE_LDA
            elif best == 'nn':
                tprange = TPRANGE_NN
            # if not self.no_print:
                # print 'Best Template for {}: {}'.format(variable, best)

        nn_normalise = True if (best == 'nn' or (best is None and self.use_nn)) else False

        if average_power_values:
            power_value = np.mean(np.array([self.return_power_window_of_variable(variable, i, nn_normalise=nn_normalise, window=tprange) for i in range(averaged_traces)]), axis=0)
        else:
            if self.auto_realign and trace > 0:
                # print "* auto realigning * tprange+realign = {}".format(tprange+REALIGN_OVERHEAD)
                target_window = self.return_power_window_of_variable(variable, trace, nn_normalise=nn_normalise, window=(tprange+(REALIGN_OVERHEAD*2)))
                base_window = self.return_power_window_of_variable(variable, 0, nn_normalise=nn_normalise, window=(tprange+(REALIGN_OVERHEAD*2)))
                realigned_window = realign_trace(base_window, target_window)
                sw, ew = REALIGN_OVERHEAD, tprange+REALIGN_OVERHEAD-1
                if sw == ew: ew += 1
                power_value = realigned_window[sw:ew]
            else:
                power_value = self.return_power_window_of_variable(variable, trace, nn_normalise=nn_normalise, window=tprange)
            # if trace > 0:
            #     print "> v {:10} t {:3}: {}".format(variable, trace, power_value)

        # if best == 'nn' or (best is None and self.use_nn):
        #     power_value = self.return_power_window_of_variable(variable, trace, nn_normalise=True, window=tprange)
        # elif best == 'lda' or (best is None and self.use_lda):
        #     # Get window of power values
        #     power_value = self.return_power_window_of_variable(variable, trace, nn_normalise=False, window=tprange)
        # else:
        #     power_value = self.return_power_window_of_variable(variable, trace, window=tprange)

        return power_value

    def get_leakage_distribution(self, variable, power_value, trace=0, normalise=True, ignore_bad=False, average_power_values=False, averaged_traces=1):
        # myvarlist = ["k001-K", "k001"]
        # if variable in myvarlist:
            # print "Getting Leakage for {}, trace {}".format(variable, trace)
        tprange = self.tprange
        best = None
        if self.use_best:
            best = self.get_best_template(variable)
            if best == 'uni':
                tprange = 1
            elif best == 'lda':
                tprange = TPRANGE_LDA
            elif best == 'nn':
                tprange = TPRANGE_NN

        var_notrace = strip_off_trace(variable)
        if best == 'nn' or (best is None and self.use_nn):

            if ignore_bad and not self.check_template(variable):
                out_distribution = get_no_knowledge_array()
                if not self.no_print:
                    print "> Ignoring NN for Variable {} as below threshold".format(variable)
            else:
                # Use neural network to predict value
                try:
                    neural_network = self.neural_network_dict[var_notrace]
                except KeyError:
                    # Add to dict!
                    if not self.no_print:
                        print "> Loading NN for Variable {}...".format(var_notrace)
                    # OLD: TODO FIX
                    # self.neural_network_dict[var_notrace] = load_sca_model('{}{}_mlp5_nodes200_window{}_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, var_notrace, tprange))
                    # NEW NEURAL NETWORKS 20/5/19
                    self.neural_network_dict[var_notrace] = load_sca_model('{}{}_mlp5_nodes100_window{}_epochs100_batchsize50_lr1e-05_sd100_traces190000_aug0_jitterNone_defaultloss_best.h5'.format(NEURAL_MODEL_FOLDER, var_notrace, tprange))
                    neural_network = self.neural_network_dict[var_notrace]
                new_input = np.resize(power_value, (1, power_value.size))
                out_distribution = neural_network.predict(new_input)[0]
        elif best == 'lda' or (best is None and self.use_lda):
            # Load LDA file
            try:
                lda = self.lda_dict[var_notrace]
            except KeyError:
                # Add to dict!
                # print "> Loading LDA for Variable {}...".format(var_notrace)
                try:
                    var_name, var_number, _ = split_variable_name(variable)
                    self.lda_dict[var_notrace] = pickle.load(open('{}{}_{}_{}.p'.format(LDA_FOLDER,
                        tprange, var_name, var_number-1),'ro'))
                except IOError:
                    if not self.no_print:
                        print "! No LDA for Variable {} Window {}, creating now...".format(var_notrace, tprange)
                lda = self.lda_dict[var_notrace]
            out_distribution = (lda.predict_proba([power_value])).astype(np.float32)[0]
        else:
            # Return Power Value having been matched!
            try:
                musigma_array = self.musigma_dict[strip_off_trace(variable)]
            except IndexError:
                if not self.no_print:
                    print "! No Mu Sigma Pair found for Variable {}".format(var_notrace)
                return get_no_knowledge_array()

            out_distribution = get_no_knowledge_array()

            for i in range(256):
                out_distribution[i] = gaussian_probability_density(power_value, musigma_array[i][0], musigma_array[i][1])

        # if variable in myvarlist:
        #     print 'Out Dist:\n{}'.format(out_distribution)
        #     exit(1)

        if normalise:
            return normalise_array(out_distribution)
        else:
            return out_distribution

    # Both together
    def get_leakage(self, variable, trace=0, normalise=True, ignore_bad=False, average_power_values=False, averaged_traces=1):
        return self.get_leakage_distribution(variable, self.get_leakage_value(variable, trace=trace, average_power_values=average_power_values, averaged_traces=averaged_traces), trace=trace, normalise=normalise, ignore_bad=ignore_bad)

    def get_plaintext_byte_distribution(self, variable, trace=0):
        # print 'For variable {}:\n{}\n\n'.format(variable, get_plaintext_array(self.plaintexts[trace][get_variable_number(variable) - 1]))
        return get_plaintext_array(self.plaintexts[trace][get_variable_number(variable) - 1])

    def get_real_value(self, variable, trace=0):
        var_name, var_number, _ = split_variable_name(variable)
        return self.realvalues[var_name][var_number-1][trace]

    def get_performance_of_handler(self, variable, traces=10000, use_extra=False):
        # Sanity check
        if use_extra != self.use_extra:
            print "! Can't get performance of Handler, we want extra {} but handler initialised as extra {}".format(use_extra, self.use_extra)
            raise
        # Returns median rank and median probability of correct value
        var_name, var_number, _ = split_variable_name(variable)
        rank_list = list()
        prob_list = list()
        # TESTS ON PROFILE TRACES (not extra!)
        for trace in range(traces):
            real_val = self.realvalues[var_name][var_number-1][self.real_trace_data_maxtraces-trace-1]
            leakage = self.get_leakage(variable, trace=self.real_trace_data_maxtraces-trace-1, normalise=False)
            rank = get_rank_from_prob_dist(leakage, real_val)
            prob = normalise_array(leakage)[real_val]
            rank_list.append(rank)
            prob_list.append(prob)
        # Return Rank List and Prob List
        return np.array(rank_list), np.array(prob_list)

    def get_leakage_rank_list(self, variable, traces=1, invert=False):
        var_name, var_number, _ = split_variable_name(variable)
        rank_list = list()
        for trace in range(traces):
            real_val = self.realvalues[var_name][var_number-1][trace]
            leakage = self.get_leakage(variable, trace=trace, normalise=False)
            if invert: leakage = normalise_array(1 - leakage)
            rank = get_rank_from_prob_dist(leakage, real_val)
            rank_list.append(rank)
        # Return Rank List
        return rank_list

    def get_leakage_rank_list_with_specific_model(self, model_file, traces=1, from_end=False):
        # Get variable of model
        model_name = model_file.replace(MODEL_FOLDER, '')
        variable = model_name.split('_')[0]
        if not self.no_print:
            print "\n* Checking model {} (variable {}) {}*\n".format(model_name, variable, 'WITH VALIDATION TRACES' if from_end else '')
        if not check_file_exists(model_file):
            if not self.no_print:
                print "!!! Doesn't exist!"
            return (None, None)
        else:

            multilabel = True if string_contains(model_file, '_multilabel_') else False
            hw = True if string_contains(model_file, 'hw_') else False
            cnn = True if string_contains(model_file, '_cnn') else False

            var_name, var_number, _ = split_variable_name(variable)
            window_size = get_window_size_from_model(model_file)

            if not self.no_print:
                print "Loading model..."
            model = load_sca_model(model_file)
            if not self.no_print:
                print "...loaded successfully!"



            # ### IF CNN, NEED TO CHANGE INPUT SHAPE
            #
            # # Get the input layer shape
            # input_layer_shape = model.get_layer(index=0).input_shape
            # # Adapt the data shape according our model input
            # if len(input_layer_shape) == 2:
            #     # This is a MLP
            #     input_data = dataset[min_trace_idx:max_trace_idx, :]
            # elif len(input_layer_shape) == 3:
            #     # This is a CNN: reshape the data
            #     input_data = dataset[min_trace_idx:max_trace_idx, :]
            #     input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
            # else:
            #     print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
            #     sys.exit(-1)



            rank_list = list()
            prob_list = list()
            predicted_values = list()
            for trace in range(traces):
                real_val = self.realvalues[var_name][var_number-1][(self.real_trace_data_maxtraces - trace - 1) if from_end else trace]

                # if trace < 3:
                #     print "Real Value {}: {}".format(trace, real_val)

                # leakage = self.get_leakage(variable, trace=trace)
                power_value = self.return_power_window_of_variable(variable, (self.real_trace_data_maxtraces - trace - 1) if from_end else trace, nn_normalise=True, window=window_size)
                new_input = np.resize(power_value, (1, power_value.size))

                ### IF CNN, NEED TO CHANGE INPUT SHAPE
                if cnn:
                    if trace == 0 and not self.no_print:
                        print "* CNN so reshaping: before {}...".format(new_input.shape)
                    new_input = new_input.reshape((new_input.shape[0], new_input.shape[1], 1))
                    if trace == 0 and not self.no_print:
                        print "** ...after {}".format(new_input.shape)

                leakage = model.predict(new_input)[0]

                if multilabel:
                    leakage = multilabel_probabilities_to_probability_distribution(leakage)
                elif hw:
                    leakage = hw_probabilities_to_probability_distribution(leakage)

                probability = leakage[real_val]

                rank = get_rank_from_prob_dist(leakage, real_val)
                # print 'Real value: {}, Prob: {}, Rank: {}, Best Value: {} (prob {})'.format(real_val, leakage[real_val], rank, np.argmax(leakage), leakage[np.argmax(leakage)])
                rank_list.append(rank)
                prob_list.append(probability)
                predicted_values.append(np.argmax(leakage))
            # Return Rank List
            return (rank_list, prob_list, predicted_values)
