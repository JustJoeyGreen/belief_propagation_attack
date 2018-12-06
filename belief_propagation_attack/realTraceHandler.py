import numpy as np
import linecache
from utility import *
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis

KEY = [0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75]
TPRANGE_NN = 700
TPRANGE_LDA = 200

class RealTraceHandler:

    def __init__(self, no_print = False, use_nn = False, use_lda = False, memory_mapped=True, tprange=200, debug=True):
        if not no_print:
            print "Preloading Matrix real_trace_data, may take a while..."
        self.real_trace_data = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH, memory_mapped=memory_mapped)
        self.real_trace_data_maxtraces, self.real_trace_data_len = self.real_trace_data.shape
        self.plaintexts = np.load(PLAINTEXT_EXTRA_FILEPATH)

        if not no_print:
            print "Preloading all timepoints, may take a while..."
        self.timepoints = dict()
        for var in variable_dict:
            self.timepoints[var] = np.load('{}{}.npy'.format(TIMEPOINTS_FOLDER, var))

        if debug:
            self.realvalues = dict()
            for var in variable_dict:
                self.realvalues[var] = np.load('{}extra_{}.npy'.format(REALVALUES_FOLDER, var))

        self.use_lda = use_lda
        self.use_nn = use_nn
        self.tprange = tprange
        if use_nn:
            self.neural_network_dict = dict()
        elif use_lda:
            self.lda_dict = dict()
        else: # neither
            # TODO Fix Power Value File, it's totally wrong
            # if not no_print:
            #     print "Preloading all power values, may take a while..."
            # self.powervalues = dict()
            # for var in variable_dict:
            #     self.powervalues[var] = np.load('{}extra_{}.npy'.format(POWERVALUES_FOLDER, var))
            self.musigma_dict = pickle.load(open(MUSIGMA_FILEPATH, 'ro'))

    def return_power_window(self, timepoint, trace, window=700, nn_normalise=False):
        """ Return the window of power values for a given value """
        start_window = max(0, timepoint - (window/2))
        end_window = min(self.real_trace_data_len, timepoint + (window/2))
        if start_window == end_window: end_window += 1 #TODO This isn't consistent!!
        trace_data = normalise_neural_trace_single(self.real_trace_data[trace]) if nn_normalise else self.real_trace_data[trace]
        return trace_data[start_window:end_window]

    def return_power_window_of_variable(self, variable, trace, window=700, nn_normalise=False):
        var_name, var_number, _ = split_variable_name(variable)
        return self.return_power_window(self.timepoints[var_name][var_number-1], trace, window=window, nn_normalise=nn_normalise)

    def get_leakage(self, variable, trace=0, normalise=True):
        # myvarlist = ["k001-K", "k001"]
        # if variable in myvarlist:
            # print "Getting Leakage for {}, trace {}".format(variable, trace)
        var_notrace = strip_off_trace(variable)
        if self.use_nn:
            # Get window of power values
            power_value = self.return_power_window_of_variable(variable, trace, nn_normalise=True, window=self.tprange)
            # Use neural network to predict value
            try:
                neural_network = self.neural_network_dict[var_notrace]
            except KeyError:
                # Add to dict!
                print "> Loading NN for Variable {}...".format(var_notrace)
                self.neural_network_dict[var_notrace] = load_sca_model('{}{}_mlp5_nodes200_window{}_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, var_notrace, self.tprange))
                neural_network = self.neural_network_dict[var_notrace]
            new_input = np.resize(power_value, (1, power_value.size))
            out_distribution = neural_network.predict(new_input)[0]
        elif self.use_lda:
            # Get window of power values
            power_value = self.return_power_window_of_variable(variable, trace, nn_normalise=False, window=self.tprange)
            # Load LDA file
            try:
                lda = self.lda_dict[var_notrace]
            except KeyError:
                # Add to dict!
                # print "> Loading LDA for Variable {}...".format(var_notrace)
                try:
                    var_name, var_number, _ = split_variable_name(variable)
                    self.lda_dict[var_notrace] = pickle.load(open('{}{}_{}_{}.p'.format(LDA_FOLDER,
                        self.tprange, var_name, var_number-1),'ro'))
                except IOError:
                    print "! No LDA for Variable {} Window {}, creating now...".format(var_notrace, self.tprange)


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

            power_val = self.return_power_window_of_variable(variable, trace, window=1)

            for i in range(256):
                out_distribution[i] = gaussian_probability_density(power_val, musigma_array[i][0], musigma_array[i][1])

        # if variable in myvarlist:
        #     print 'Out Dist:\n{}'.format(out_distribution)
        #     exit(1)

        if normalise:
            return normalise_array(out_distribution)
        else:
            return out_distribution

    def get_plaintext_byte_distribution(self, variable, trace=0):
        # print 'For variable {}:\n{}\n\n'.format(variable, get_plaintext_array(self.plaintexts[trace][get_variable_number(variable) - 1]))
        return get_plaintext_array(self.plaintexts[trace][get_variable_number(variable) - 1])

    def get_real_value(self, variable, trace=0):
        var_name, var_number, _ = split_variable_name(variable)
        return self.realvalues[var_name][var_number-1][trace]

    def get_leakage_rank_list(self, variable, traces=1):
        var_name, var_number, _ = split_variable_name(variable)
        rank_list = list()
        for trace in range(traces):
            real_val = self.realvalues[var_name][var_number-1][trace]
            leakage = self.get_leakage(variable, trace=trace, normalise=False)
            rank = get_rank_from_prob_dist(leakage, real_val)
            rank_list.append(rank)
        # Return Rank List
        return rank_list

    def get_leakage_rank_list_with_specific_model(self, model_file, traces=1):
        # Get variable of model
        model_name = model_file.replace(MODEL_FOLDER, '')
        variable = model_name.split('_')[0]
        print "\n* Checking model {} (variable {}) *\n".format(model_name, variable)
        if not check_file_exists(model_file):
            print "!!! Doesn't exist!"
            return False
        else:
            var_name, var_number, _ = split_variable_name(variable)
            window_size = get_window_size_from_model(model_file)
            model = load_sca_model(model_file)
            rank_list = list()
            for trace in range(traces):
                real_val = self.realvalues[var_name][var_number-1][trace]
                # leakage = self.get_leakage(variable, trace=trace)
                power_value = self.return_power_window_of_variable(variable, trace, nn_normalise=True, window=window_size)
                new_input = np.resize(power_value, (1, power_value.size))
                leakage = model.predict(new_input)[0]
                rank = get_rank_from_prob_dist(leakage, real_val)
                rank_list.append(rank)
            # Return Rank List
            return rank_list
