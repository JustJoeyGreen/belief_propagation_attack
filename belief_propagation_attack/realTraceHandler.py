import numpy as np
import linecache
from utility import *
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis

KEY = [0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75]

class RealTraceHandler:

    def __init__(self, no_print = False, use_nn = False, use_lda = False, memory_mapped=True, nn_window = 700, lda_window = 200):
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
        self.lda_window = lda_window
        if use_nn:
            self.neural_network_dict = dict()
        elif use_lda:
            self.lda_dict = dict()
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
            # Get window of power values
            power_value = self.return_power_window_of_variable(var_name, var_number-1, trace, nn_normalise=False)
            # Load LDA file
            try:
                lda = self.lda_dict[var_notrace]
            except KeyError:
                # Add to dict!
                print "> Loading LDA for Variable {}...".format(var_notrace)
                try:
                    self.lda_dict[var_notrace] = pickle.load(open('{}{}_{}_{}.p'.format(LDA_FOLDER,
                        self.lda_window, var_name, var_number-1),'ro'))
                except IOError:
                    print "! No LDA for Variable {} Window {}, creating now...".format(var_notrace, self.lda_window)


                lda = self.lda_dict[var_notrace]
            out_distribution = (lda.predict_proba([power_value])).astype(np.float32)[0]
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
