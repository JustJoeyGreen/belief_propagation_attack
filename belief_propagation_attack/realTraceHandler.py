import numpy as np
import linecache
from utility import *
from sklearn.preprocessing import normalize

KEY = [0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75]

class RealTraceHandler:

    def __init__(self, no_print = False, use_nn = False, use_lda = False, memory_mapped=True, nn_window = 700):
        if not no_print:
            print "Preloading Matrix real_trace_data, may take a while..."
        self.real_trace_data = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH, memory_mapped=memory_mapped)
        self.real_trace_data_maxtraces, self.real_trace_data_len = self.real_trace_data.shape
        self.plaintexts = np.load(PLAINTEXT_FILEPATH)

        if not no_print:
            print "Preloading all timepoints, may take a while..."
        self.timepoints = dict()
        for var in variable_dict:
            self.timepoints[var] = np.load('{}{}.npy'.format(TIMEPOINTS_FOLDER, var))

        self.use_lda = use_lda
        self.use_nn = use_nn
        self.nn_window = nn_window
        if use_nn:
            # if not no_print:
            #     print "Preloading Neural Networks, may take a while..."
            self.neural_network_dict = dict()
            # for v, length in variable_dict.iteritems():
            #     for i in range(length):
            #         varname = '{}{}'.format(v, pad_string_zeros(i+1))
            #         self.neural_network_dict[varname] = load_sca_model('{}{}_mlp5_nodes200_window{}_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, varname, nn_window))
        elif use_lda:
            pass
        else: # neither
            if not no_print:
                print "Preloading all power values, may take a while..."
            self.powervalues = dict()
            for var in variable_dict:
                self.powervalues[var] = np.load('{}extra_{}.npy'.format(POWERVALUES_FOLDER, var))
            self.musigma_dict = pickle.load(open(MUSIGMA_FILEPATH, 'ro'))

    def return_power_window(self, timepoint, trace, window=700, nn_normalise=False):
        """ Return the window of power values for a given value """
        start_window = max(0, timepoint - (window/2))
        end_window = min(self.real_trace_data_len, timepoint + (window/2))
        trace_data = normalise_neural_trace_single(self.real_trace_data[trace]) if nn_normalise else self.real_trace_data[trace]
        return trace_data[start_window:end_window]

    def return_power_window_of_variable(self, var_name, var_number, trace, window=700, nn_normalise=False):
        return self.return_power_window(self.timepoints[var_name][var_number], trace, window=window, nn_normalise=nn_normalise)

    # The function to match real power value to probability distribution
    def real_value_match(var, power_value, normalise = True, use_lda = False,
        use_nn = False, trace_range = 200):
        """Matches the real power value to a probability distribution,
        using a specific method (LDA, NN, or standard Templates)"""
        # Check for power_value being None
        # print "Variable {}, Power Value {}".format(var, power_value)
        if np.any(np.isnan(power_value)):
            print "$ Variable {} has powervalue {}".format(var, power_value)
            return get_no_knowledge_array()
        # Get mu and sigma for var_name
        # Strip off trace
        var = strip_off_trace(var)
        if use_lda:
            # exit()
            var_name, var_number, _ = split_variable_name(var)
            # Load LDA file
            lda_file = pickle.load(open('{}{}_{}_{}.p'.format(LDA_FOLDER,
                trace_range, var_name, var_number-1),'ro'))
            probabilities = (lda_file.predict_proba([power_value]))\
                .astype(np.float32)[0]

            if (len(probabilities)) != 256:
                print "Length of Probability for var {} is {}, must be \
                    256".format(var, len(probabilities))
                raise IndexError
            # Predict Probabilities and Normalise
            return probabilities
        elif use_nn:
            var_name, var_number, _ = split_variable_name(var)
            # Load NN file
            # print "Using Neural Networks here! Variable {}".format(var)

            # TODO Major debugging!
            print "> Loading Neural Network for var {}...".format(var) #debug

            if PRELOAD_NEURAL_NETWORKS:
                # neural_network = neural_network_dict[var]
                probabilities = normalise_array(
                    neural_network_dict[var].predict(np.resize(
                        power_value, (1, power_value.size)))[0])
            else:
                # neural_network = load_sca_model('{}{}_mlp5_nodes200_window700_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, var))
                probabilities = normalise_array(
                    load_sca_model('{}{}_mlp5_nodes200_window700_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, var)).predict(np.resize(
                        power_value, (1, power_value.size)))[0])

            print ">> done! Already got probabilities!" #debug

            # probabilities = normalise_array(
            #     neural_network.predict(np.resize(
            #         power_value, (1, power_value.size)))[0])

            if (len(probabilities)) != 256:
                print "Length of Probability for var {} is {}, must be 256"\
                .format(var, len(probabilities))
                raise IndexError
            # Predict Probabilities and Normalise
            return probabilities
        else:
            musigma_dict = pickle.load(open(MUSIGMA_FILEPATH, 'ro'))
            try:
                musigma_array = musigma_dict[var]
            except IndexError:
                print "! No Mu Sigma Pair found for Variable {}".format(var)
                return get_no_knowledge_array()

            out_distribution = get_no_knowledge_array()

            for i in range(256):
                out_distribution[i] = gaussian_probability_density(power_value[0], musigma_array[i][0], musigma_array[i][1])

            if normalise:
                return normalise_array(out_distribution)
            else:
                return out_distribution

    def return_all_values(self, traces=1, offset=0, use_lda=True, trace_range=200, unprofiled=True,
                          use_random_traces=True, seed=0, no_print = True, memory_mapped=True, correlation_threshold = None, normalise_each_trace=False):


        # Input: Number of Repeats and Traces to use
        # Output: real_all_values[repeat][node][trace]

        no_print = False # TODO

        real_all_values = list()

        # Replace Plaintext Values with Plaintexts
        if unprofiled:
            plaintexts = np.load(PLAINTEXT_EXTRA_FILEPATH)
        else:
            plaintexts = np.load(PLAINTEXT_FILEPATH)

        if not use_random_traces and (traces) > len(plaintexts):
            print "!! Error: Attack requires {} Plaintexts, but only {} available (Unprofiled = " \
                  "{})".format(
                traces, len(plaintexts), unprofiled)
            raise ValueError

        if not no_print:
            print "Loading Matrix real_trace_data, may take a while..."
        # if unprofiled:
        #     # real_trace_data = np.transpose(np.load(TRACEDATA_EXTRA_FILEPATH, mmap_mode='r'))
        #     real_trace_data = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH, memory_mapped=memory_mapped)
        # else:
        #     # real_trace_data = np.transpose(np.load(TRACEDATA_FILEPATH, mmap_mode='r'))
        #     real_trace_data = load_trace_data(filepath=TRACEDATA_FILEPATH, memory_mapped=memory_mapped)
        # if normalise_each_trace:
        #     if not no_print:
        #         print "...now normalising..."
        #     real_trace_data = normalise_neural_traces(real_trace_data)
        if not no_print:
            print "...done!"
            print_new_line()

        if use_random_traces:
            potential_traces = real_trace_data.shape[0]
            np.random.seed(seed)
            random_traces = get_random_numbers(traces, 0, potential_traces - 1)

        real_all_values = dict()
        real_all_values['key'] = np.array(KEY)

        # For each variable, load TP file and pull through power values for each trace
        # THIS TAKES A SUPER LONG TIME!!!!
        for var, length in variable_dict.iteritems():
            print var,length


            real_all_values[var] = [0] * traces

            # Get Time Points for Variable var
            time_points = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var))

            # real_all_values[var] = [np.zeros((length, trace_range)) for trace in range(traces)]
            for trace in range(traces):
                real_all_values[var][trace] = np.zeros((length, trace_range))

            for node in range(length):
                tp = time_points[node]
                my_start = tp - (trace_range / 2)
                my_end = tp + (trace_range / 2)
                if my_end == my_start: my_end += 1
                trace_selection = real_trace_data[:, my_start:my_end]
                if normalise_each_trace:
                    # TEST
                    # print 'Before ({}):\n{}\n'.format(trace_selection.shape, trace_selection)
                    trace_selection = normalise_neural_traces(trace_selection) # THIS LINE IS BOTTLENECK
                    # trace_selection = trace_selection / trace_selection.max(axis=0)
                    # trace_selection = normalize(trace_selection, axis=0, norm='max')
                    # print 'After:\n{}'.format(trace_selection)
                    # pass
                    print "> done {} {}".format(var, node)
                for trace in range(traces):
                    my_trace = random_traces[trace] if use_random_traces else trace + offset
                    real_all_values[var][trace][node] = trace_selection[my_trace]


            # # if use_lda:
            # # Select Power Values from trace_range around the Time point (from real_trace_data)
            # for trace in range(traces):
            #
            #     real_all_values[var][trace] = np.zeros((len(time_points), trace_range))
            #     for node in range(len(time_points)):
            #         tp = time_points[node]
            #         my_trace = random_traces[trace] if use_random_traces else trace + offset
            #         my_start = tp - (trace_range / 2)
            #         my_end = tp + (trace_range / 2)
            #         if my_end == my_start: my_end += 1
            #         real_all_values[var][trace][node] = real_trace_data[my_trace, my_start:my_end]





        for trace in range(traces):
            print trace

            my_trace = random_traces[trace] if use_random_traces else trace + offset
            # if use_lda:
            for i in range(16):
                real_all_values['p'][trace][:16][i] = np.full(trace_range,
                                                                   plaintexts[my_trace][i])
            # else:
            #     real_all_values['p'][trace][:16] = np.array(plaintexts[my_trace])

        # print "In Real Trace Handler, Real All Values:\n{}\n".format(real_all_values)

        exit(1)

        return real_all_values



    def get_leakage(self, variable, trace=0, normalise=True):
        var_name, var_number, _ = split_variable_name(variable)
        var_notrace = strip_off_trace(variable)
        if self.use_nn:
            # Get window of power values
            power_value = self.return_power_window_of_variable(var_name, var_number-1, trace, nn_normalise=True)
            # Use neural network to predict value
            try:
                neural_network = self.neural_network_dict[var_notrace]
            except KeyError:
                # Add to dict!
                print "> Loading NN for Variable {}...".format(var_notrace)
                self.neural_network_dict[var_notrace] = load_sca_model('{}{}_mlp5_nodes200_window{}_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, var_notrace, self.nn_window))
                neural_network = self.neural_network_dict[var_notrace]

            new_input = np.resize(power_value, (1, power_value.size))
            out_distribution = neural_network.predict(new_input)[0]
        elif self.use_lda:
            print 'TODO in rTH LDA'
            raise
        else:
            # Return Power Value having been matched!
            try:
                musigma_array = self.musigma_dict[strip_off_trace(variable)]
            except IndexError:
                print "! No Mu Sigma Pair found for Variable {}".format(var_notrace)
                return get_no_knowledge_array()

            out_distribution = get_no_knowledge_array()
            power_val = self.powervalues[var_name][trace][var_number-1]
            for i in range(256):
                out_distribution[i] = gaussian_probability_density(power_val, musigma_array[i][0], musigma_array[i][1])

        if normalise:
            return normalise_array(out_distribution)
        else:
            return out_distribution

    def get_plaintext_byte_distribution(self, variable, trace=0):
        # print 'For variable {}:\n{}\n\n'.format(variable, get_plaintext_array(self.plaintexts[trace][get_variable_number(variable) - 1]))
        return get_plaintext_array(self.plaintexts[trace][get_variable_number(variable) - 1])
