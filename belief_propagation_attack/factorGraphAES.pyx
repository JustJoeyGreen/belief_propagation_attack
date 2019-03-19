import gexf_graphCreator
import leakageSimulatorAESFurious as lSimF
import realTraceHandler as rTraceH
import networkx as nx
import time
import math
import numpy as np
cimport numpy as np
from utility import *
import sys
# from joblib import Parallel, delayed
# import dill
# import multiprocessing as mp

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

FORCE_GRAPH_CREATION                = False
CONVERGENCE_LENGTH                  = 500
LAZY_THRESHOLD                      = 30
INFORMATION_EXHAUSTED_THRESHOLD     = 40
INFORMATION_EXHAUSTED_S             = 10
MARTIN_THRESHOLD                    = 3125
CHECK_EMPTY                         = False
STOP_EMPTY_PROPAGATION              = True
# failure_threshold                   = 1e-6 #0.0035 #3e-3 #1e-3 #0.00001
FAILURE_THRESHOLD_DICT              = { 5 : 1e-8,    4 : 1e-8,  3 : 1e-8,   2 : 1e-7, 1 : 1e-6,
                                        0 : 1e-5,   -1 : 1e-4, -2 : 7.5e-3,-3 : 2e-3,-5 : 1e-5}
FAILURE_THRESHOLD_AT_NODE           = 4e-5 #2e-5 #4e-5
FAILURE_NUMBER_OF_PLAINTEXTS        = 1 #4
FAILURE_MIN_ROUNDS                  = 8
FIX_WITH_HAMMING_WEIGHT_ARRAY         = False
FIX_WITH_ELMO_ARRAY                  = False
CPF_INCLUDE_K                       = False

PRINT_EXHAUSTION                    = False

rcon = list([
    1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108
])

shared_variables = ['k', 'sk', 'xk', 'rc']

class FactorGraphAES:

    """ Container for Factor Graph """
    def __init__(self, no_print = False, int traces = 1, removed_nodes = None, left_out_nodes = None, key_scheduling = False, furious = True, rounds_of_aes = 10, remove_cycle = False, my_print = False, real_traces = False, use_nn = False, use_lda = False, use_best = False, tprange=200, jitter = None, badly_leaking_snr = -7, badly_leaking_nodes = None, badly_leaking_traces = None, no_noise_nodes = None):

        self.no_print = no_print

        if removed_nodes is None:
            removed_nodes = []
        if left_out_nodes is None:
            left_out_nodes = []

        furious_string = ""
        round_string = str(rounds_of_aes)
        if my_print:
            print "Rounds of AES: {}".format(rounds_of_aes)
        remove_cycle_string = ""
        if furious:
            furious_string = "Furious"
        if remove_cycle:
            remove_cycle_string = "RemovedCycle"

        if my_print:
            print "Remove Cycle: {}".format(remove_cycle)

        if FORCE_GRAPH_CREATION:
            # TEST: Force Create Graph
            if my_print:
                print "Creating {} Round AES {} Factor Graph {}\n{} Traces, removed_nodes {}, key scheduling {}...".format(round_string, furious_string, remove_cycle_string, traces, removed_nodes, key_scheduling)
            if furious:
                gexf_graphCreator.create_factor_graph_full_aes_furious(traces, removed_nodes, key_scheduling, rounds_of_aes, remove_cycle)
            else:
                gexf_graphCreator.create_factor_graph_full_aes(traces, removed_nodes, key_scheduling, rounds_of_aes)
            if my_print:
                print "...finished creating!"
                print_new_line()
        else:
            try:
                self.G = nx.read_gexf(
                    'graphs/{}_trace_fullAES{}{}{}_removednodes-{}_keysched-{}.graph'.format(traces, furious_string,
                                                                                         rounds_of_aes,
                                                                                         remove_cycle_string,
                                                                                         list_to_file_string(
                                                                                             removed_nodes),
                                                                                         key_scheduling))
                if my_print:
                    print "Loaded AES {} Factor Graph {}{}{} with {} Traces, removed_nodes {}, key scheduling {}".format(furious_string, one_round_string + ' ', remove_cycle_string + ' ', two_rounds_string + ' ', traces, removed_nodes, key_scheduling)
            except IOError:
                # TEST: Force Create Graph
                if my_print:
                    print "- Attempted load, could not find available graph.\nCreating AES {} Factor Graph {}{}{} with {} Traces, removed_nodes {}, key scheduling {}...".format(furious_string, one_round_string, two_rounds_string, remove_cycle_string, traces, removed_nodes, key_scheduling)
                if furious:
                    gexf_graphCreator.create_factor_graph_full_aes_furious(traces, removed_nodes, key_scheduling, rounds_of_aes, remove_cycle)
                else:
                    gexf_graphCreator.create_factor_graph_full_aes(traces, removed_nodes, key_scheduling, rounds_of_aes)
                if my_print:
                    print "...finished creating!"
                    print_new_line()

        self.G = nx.read_gexf(
            'graphs/{}_trace_fullAES{}{}{}_removednodes-{}_keysched-{}.graph'.format(traces, furious_string,
                                                                                 rounds_of_aes,
                                                                                 remove_cycle_string,
                                                                                 list_to_file_string(removed_nodes),
                                                                                 key_scheduling))

        variables       = list()
        factors         = list()
        key_nodes       = list()
        plaintext_nodes = list()

        for node in self.G:
            if string_starts_with(node, '_'):
                factors.append(node)
            else:
                variables.append(node)
                if string_ends_with(node, '-K'):
                    key_nodes.append(node)
                elif string_starts_with(node, 'p') and (get_variable_number(node) <= 16):
                    plaintext_nodes.append(node)

        self.rounds_of_aes          = rounds_of_aes
        self.variables              = sorted(variables)
        self.factors                = sorted(factors)
        self.edges                  = sorted(list(self.G.edges()))
        self.key_nodes              = sorted(key_nodes)
        self.plaintext_nodes        = sorted(plaintext_nodes)
        self.initial_distribution   = {}
        self.key                    = np.zeros(16)
        self.traces                 = traces
        self.SNR_exp                = 5

        self.left_out_nodes = left_out_nodes
        self.dead_factor_nodes = []

        for var in self.variables:
            if string_contains_any(var, self.left_out_nodes):
                # Found variable to be left out
                dead_factors = self.get_neighbours(var)
                for fac in dead_factors:
                    self.dead_factor_nodes.append(fac)

        self.dead_factor_nodes = set(self.dead_factor_nodes)

        self.leaf_nodes = []
        for var in self.variables:
            if (len(self.get_neighbours(var)) == 1) and not string_ends_with(var, '-K') and not string_starts_with(var, 'p'):
                self.leaf_nodes.append(var)

        self.steps_from_key_length = LAZY_THRESHOLD
        self.steps_from_key_variables, self.steps_from_key_factors = self.get_steps_from_key_lists()

        self.initialise_edges()

        if real_traces:
            self.handler = rTraceH.RealTraceHandler(no_print=self.no_print, use_nn=use_nn, use_lda=use_lda, use_best=use_best, tprange=tprange, jitter=jitter)

        self.averaged_key_values    = None

        self.badly_leaking_snr      = badly_leaking_snr
        self.badly_leaking_traces   = badly_leaking_traces

        self.badly_leaking_nodes    = badly_leaking_nodes
        self.no_noise_nodes         = no_noise_nodes

        # self.badly_leaking_nodes = None if badly_leaking_nodes is None else get_all_variables_that_match(self.variables, self.badly_leaking_nodes)
        # self.no_noise_nodes = None if no_noise_nodes is None else get_all_variables_that_match(self.variables, self.no_noise_nodes)

        # print 'Badly Leaking Nodes: {}'.format(self.badly_leaking_nodes)

    def set_key(self, val):
        self.key = val

    def set_snr_exp(self, val):
        self.SNR_exp = val

    def get_snr_exp(self):
        return self.SNR_exp

    def get_number_of_traces(self):
        return self.traces

    def get_leaf_nodes(self):
        return self.leaf_nodes

    def get_variables(self):
        return self.variables

    def get_factors(self):
        return self.factors

    def get_edges(self):
        return self.edges

    def get_key_nodes(self):
        return self.key_nodes

    def set_initial_distribution(self, node, np.ndarray [DTYPE_t, ndim=1] value):
        # print "Now setting initial dist for variable {}".format(node)
        # print "> Setting Initial Dist for {}:\n{}\n".format(node, value)
        self.initial_distribution[node] = value
        # print "done: {}".format(value)

    def get_initial_distribution(self, node):
        try:
            return self.initial_distribution[node]
        except KeyError:
            if node in self.variables:
                return get_no_knowledge_array()
            else:
                print "Error: No node named {} in variable list".format(node)
                raise

    # ****************************************************** ALL FUNCTIONS ******************************************************

    def get_matching_variables(self, target_name):
        return get_variables_that_match(self.variables, target_name)

    def get_all_key_initial_distributions(self):
        # Container
        key_distributions = np.zeros((len(self.key_nodes), 256))
        key_names = self.key_nodes
        for i in range(len(key_names)):
            key_distributions[i] = self.get_initial_distribution(key_names[i])
        return key_distributions

    def get_all_key_incoming_messages(self):
        # Container
        key_distributions = np.zeros((len(self.key_nodes), 256))
        key_names = self.key_nodes
        for i in range(len(key_names)):
            key_distributions[i] = self.get_all_incoming_messages(key_names[i])
        return key_distributions

    def get_plaintext_values(self):
        plaintexts = np.zeros(len(self.plaintext_nodes))
        for i, p in enumerate(self.plaintext_nodes):
            plaintexts[i] = find_value_in_list(self.get_initial_distribution(p), 1.0)
        return plaintexts

    def check_leakage_details(self, worst_case = True):
        # Get plaintext
        plaintext = self.get_plaintext_values()
        # Simulate the real values
        sim = lSimF.LeakageSimulatorAESFurious()
        sim.fix_key(self.key)
        sim.fix_plaintext(plaintext)
        sim.simulate(read_plaintexts=0, print_all=0, random_plaintexts=0, affect_with_noise=False,
             hw_leakage_model=False, real_values=True)
        real_values = sim.get_leakage_dictionary()
        # Loop through all variables
        rank_list = [ [] for i in range (256) ]
        all_ranks = list()

        # chosen_variables = self.variables
        # chosen_variables = ['k{}-0'.format(pad_string_zeros(i)) for i in range(17, 33)]
        # chosen_variables += ['sk{}-0'.format(pad_string_zeros(i)) for i in range(1, 5)]
        # chosen_variables.append('xk001-0')
        chosen_variables = ['{}{}-0'.format(v, pad_string_zeros(i)) for v in ['t','p','s','mc','cm','xt','h'] for i in range(1,17)]
        for i in range(1,17):
            chosen_variables.append('k{}-K'.format(pad_string_zeros(i)))

        for variable in chosen_variables:

            var_name, var_number, _ = split_variable_name(variable)
            try:
                real_val = int(real_values[var_name][0][var_number-1])
            except IndexError:
                real_val = int(real_values[var_name][var_number-1])
            try:
                initial_dist = self.initial_distribution[variable]
                rank = get_rank_from_prob_dist(self.initial_distribution[variable], real_val, worst_case = worst_case)

                if rank > 128:
                    print "\n* Variable {} is rank {}, details:\n".format(variable, rank)
                    print "Real Value: {}\nProbability: {}\nInitial Distribution:\n{}\n".format(real_val, initial_dist[real_val], initial_dist)

                # print "Variable {:8}: Rank {:3}".format(variable, rank)
                rank_list[rank-1].append(variable)
                if var_name == 'p' and var_number <= 16:
                    pass
                    # print "- Ignoring {}".format(variable)
                else:
                    all_ranks.append(rank)

            except KeyError:
                pass

        print "*** Checking Leakage Details from Initial Distribution (Worst Case {}) ***\n".format(worst_case)
        for i in range (256):
            if len(rank_list[i]) >= 1:
                print "Rank {:3} Nodes:\n{}\n".format(i+1, rank_list[i])
        print "*** Stats ***"
        print_statistics(all_ranks)

    def compute_averaged_key_values(self, averaged_traces = 1, specific_trace = None, no_leak = None,
                                      fixed_value = None, elmo_pow_model = False, real_traces = False,
                                      seed=0, no_noise=False, offset=0, ignore_bad=False):
        snr = 2 ** self.SNR_exp
        # Handle no leaks and fixed values
        if no_leak is None:
            no_leak = []
        else:
            no_leak = get_all_variables_that_match(self.variables, no_leak)
        if fixed_value is None:
            matched_fixed_values = []
        else:
            matched_fixed_values = get_variables_that_match(self.variables, fixed_value[0])
            if not self.no_print:
                print "::: Fixing var {}, matched: {}".format(fixed_value[0], matched_fixed_values)

        averaged_power_values = [0] * len(self.key_nodes)

        if not real_traces:
            # Simulate traces - if LFG then do all traces at same time
            leakage_simulator = lSimF.LeakageSimulatorAESFurious(seed=seed+offset)
            leakage_simulator.fix_key(self.key)

            # leakage = leakage_simulator.simulate(snr = snr, traces = self.traces, offset = 0, read_plaintexts = 0, random_plaintexts = 1)
            print "* In Factor Graph, badly leaking traces {}".format(self.badly_leaking_traces)
            leakage_simulator.simulate(snr = snr, traces = averaged_traces, offset = offset, read_plaintexts = 0, random_plaintexts = 1, badly_leaking_nodes = self.badly_leaking_nodes, badly_leaking_traces = self.badly_leaking_traces, badly_leaking_snr = self.badly_leaking_snr, no_noise_nodes = self.no_noise_nodes, threshold = None, local_leakage = 0, print_all = 0, affect_with_noise = not no_noise, hw_leakage_model = False, real_values = False, rounds_of_aes = self.rounds_of_aes)

            leakage = leakage_simulator.get_leakage_dictionary()

            hw_sigma = get_sigma(snr)

            for i, var in enumerate(self.key_nodes):
                # Split name
                var_name, var_number, var_trace = split_variable_name(var)
                try:
                    var_leakage = leakage[var_name]
                    powervalue = var_leakage[:, var_number-1]
                    # Need to handle multiple power values!!!
                    # For now, average them I guess
                    # print "In COMPUTE_AVERAGE BEFORE AVERAGE: Var {}, PowerValue {}".format(var, powervalue)
                    powervalue = np.mean(powervalue) #debug
                    averaged_power_values[i] = powervalue
                    # print "In COMPUTE_AVERAGE: Var {}, PowerValue {}".format(var, powervalue)
                except KeyError:
                    print "! Key Error for Variable {}".format(var)
                    print leakage[var_name][var_trace][var_number-1]
                    raise
        else:

            for i, var in enumerate(self.key_nodes):
                # Split name
                var_name, var_number, var_trace = split_variable_name(var)
                try:
                    powervalue = self.handler.get_leakage_value(var, average_power_values=True, averaged_traces=averaged_traces)
                    # print "In Computer Average Keys, power value {}, type {}".format(powervalue, type(powervalue))
                    averaged_power_values[i] = powervalue
                    # print "In COMPUTE_AVERAGE: Var {}, PowerValue {}".format(var, powervalue)
                except KeyError:
                    print "! Key Error for Variable {}".format(var)
                    raise

        self.averaged_key_values = averaged_power_values


    def set_all_initial_distributions(self, specific_trace = None, no_leak = None,
                                      fixed_value = None, elmo_pow_model = False, real_traces = False,
                                      seed=0, no_noise=False, offset=0, ignore_bad=False, trace_id = None):

        snr = 2 ** self.SNR_exp

        # Handle no leaks and fixed values
        if no_leak is None:
            no_leak = []
        else:
            no_leak = get_all_variables_that_match(self.variables, no_leak)
        if fixed_value is None:
            matched_fixed_values = []
        else:
            matched_fixed_values = get_variables_that_match(self.variables, fixed_value[0])
            if not self.no_print:
                print "::: Fixing var {}, matched: {}".format(fixed_value[0], matched_fixed_values)

        # SIMULATED DATA
        if not real_traces:
            # Simulate traces - if LFG then do all traces at same time
            leakage_simulator = lSimF.LeakageSimulatorAESFurious(seed=seed+offset)
            leakage_simulator.fix_key(self.key)

            # leakage = leakage_simulator.simulate(snr = snr, traces = self.traces, offset = 0, read_plaintexts = 0, random_plaintexts = 1)
            leakage_simulator.simulate(snr = snr, traces = self.traces, offset = offset, read_plaintexts = 0, random_plaintexts = 1, badly_leaking_nodes = self.badly_leaking_nodes, badly_leaking_traces = self.badly_leaking_traces, badly_leaking_snr = self.badly_leaking_snr, no_noise_nodes = self.no_noise_nodes, threshold = None, local_leakage = 0, print_all = 0, affect_with_noise = not no_noise, hw_leakage_model = False, real_values = False, rounds_of_aes = self.rounds_of_aes, trace_id=trace_id)

            leakage = leakage_simulator.get_leakage_dictionary()

            hw_sigma = get_sigma(snr)

            # Go through and put in
            for var in self.variables:
                # Split name
                var_name, var_number, var_trace = split_variable_name(var)
                # If chosen not to leak, initialise with all ones
                if var in no_leak or var_name in no_leak or strip_off_trace(var) in no_leak:
                    # print "NOT LEAKING ON {}".format(var_name)
                    distribution = get_no_knowledge_array()
                # If shared variable, apply straight to variable (independent of trace)
                elif var_name in shared_variables and self.traces > 1:
                    # Check if in dictionary
                    try:
                        var_leakage = leakage[var_name]
                        powervalue = var_leakage[:, var_number-1]
                        # Need to handle multiple power values!!!
                        # For now, average them I guess


                        powervalue = np.mean(powervalue)
                        # powervalue = powervalue[0] #debug

                        # Check if p1 - p16 or rc
                        if var_name == 'rc':
                            # Add as plaintext array
                            distribution = get_plaintext_array(powervalue)
                        else:
                            if elmo_pow_model:
                                self.set_initial_distribution(var, template_match(var, powervalue, snr))
                                # # DEBUG
                                # if var_name == 'k' and var_number == 1:
                                #     print "Setting Dist of Var {} from Power Value {}:\n{}\n".format(var, powervalue, template_match(var, powervalue, snr))
                            else:
                                self.set_initial_distribution(var, get_hamming_weight_array(powervalue, hw_sigma))
                    except KeyError:
                        print "! Key Error for Variable {}".format(var)
                        print leakage[var_name][var_trace][var_number-1]
                        raise
                else:
                    # Check if in dictionary
                    try:
                        # print "Getting value for var {}...".format(var)
                        # Otherwise, get right trace number
                        try:
                            if var_name == 'k' and var_number <= 16 and self.averaged_key_values is not None:
                                powervalue = self.averaged_key_values[var_number-1]
                                # print "In SET_ALL_INIT: Var {}, PowerValue {}".format(var, powervalue)
                            elif specific_trace is not None:
                                powervalue = leakage[var_name][specific_trace][var_number-1]
                            else:
                                powervalue = leakage[var_name][var_trace][var_number-1]
                            # print "...done! Value = {}".format(powervalue)
                        except IndexError:
                            print "! ERROR: Power Value not Found! leakage[{}][{}][{}]\n".format(var_name, var_trace, var_number-1)
                            raise
                        # print "In FactorGraphAES, Power Value for {}{} in trace {}: {}".format(var_name, var_number, var_trace, powervalue)
                        # Check if p1 - p16 or rc
                        if var_name == 'p' and var_number <= 16:
                            self.set_initial_distribution(var, get_plaintext_array(powervalue))
                        else:
                            # Add to Initial Distribution
                            if elmo_pow_model:
                                self.set_initial_distribution(var, template_match(var, powervalue, snr))
                                # # DEBUG
                                # if var_name == 'k' and var_number == 1:
                                #     print "Setting Dist of Var {} from Power Value {}:\n{}\n".format(var, powervalue, template_match(var, powervalue, snr))
                            else:
                                self.set_initial_distribution(var, get_hamming_weight_array(powervalue, hw_sigma))
                    except KeyError:
                        print "KeyError: No record for {}[{}] in leakage ({})".format(var_name, var_number-1, var)
                        raise

        else:

            # Already stored, do one trace at a time!
            for trace in range(self.traces):
                # Each var in turn
                cheat = 0
                for var in self.variables:
                    # Split name
                    var_name, var_number, var_trace = split_variable_name(var)
                    if var in no_leak or var_name in no_leak or strip_off_trace(var) in no_leak:
                        self.set_initial_distribution(var, get_no_knowledge_array())
                    elif var_name == 'p' and var_number <= 16:
                        self.set_initial_distribution(var, self.handler.get_plaintext_byte_distribution(var, trace=offset+trace))
                    elif var_name == 'k' and var_number <= 16 and self.averaged_key_values is not None:
                        self.set_initial_distribution(var, self.handler.get_leakage_distribution(var, self.averaged_key_values[var_number-1], ignore_bad=ignore_bad))
                        # print "Initial Dist for var {}:\n{}\n".format(var, self.handler.get_leakage_distribution(var, self.averaged_key_values[var_number-1], ignore_bad=ignore_bad)) #debug
                    else:
                        if cheat == 0:
                            self.set_initial_distribution(var, self.handler.get_leakage(var, trace=offset+trace, ignore_bad=ignore_bad))
                            # cheat = 1
                        else:
                            pass

    def set_key_distributions(self, key_distributions):
        # For each key byte, set the distribution from the input
        cdef int i
        for i in range(len(self.key_nodes)):
            k = 'k{}-K'.format(pad_string_zeros(str(i + 1)))
            self.set_initial_distribution(k, key_distributions[i].astype(DTYPE))

    def fabricate_key_scheduling_leakage(self):
        if 'k017-K' in self.variables:
            print "Fabricating now!"
        else:
            print "No need to fabricate"
        sys.exit(1)

    def get_neighbours(self, name):
        return list(self.G.neighbors(name))

    def get_other_neighbours(self, name, neighbour):
        neighbours = list(self.get_neighbours(name))
        try:
            neighbours.remove(neighbour)
        except (KeyError, ValueError):
            pass
        return neighbours

    def initialise_edges(self):
        for node in self.G.nodes():
            for neighbour in self.get_neighbours(node):
                self.G.edge[node][neighbour] = get_no_knowledge_array()

    def get_incoming_message(self, node, neighbour):
        return self.G.edge[neighbour][node]

    def get_outgoing_message(self, node, neighbour):
        return self.G.edge[node][neighbour]

    def check_factor_nodes(self, print_all = False, simple_xor = True):
        print_length = 20
        print "*** Checking Factor Nodes ***"
        print_new_line()
        factor_counter = dict()
        edge_counter = 0
        for f in self.factors:
            # Count name
            f_name = f.split('_')[1]
            if f_name in factor_counter:
                factor_counter[f_name] += 1
            else:
                factor_counter[f_name] = 1
            if f_name == 'Xor':
                edge_counter += 3
            else:
                edge_counter += 2
            neighbours = self.get_neighbours(f)
            # If XOR nodes, needs 3+ neighbours (exactly 3 if simple_xor)
            # Else (Sbox or Xtimes), needs exactly 2
            if (string_starts_with(f, '_Xor_') and ((simple_xor and len(neighbours) == 3) or (not simple_xor and len(neighbours) >= 3))) or (not string_starts_with(f, '_Xor_') and len(neighbours) == 2):
                if print_all:
                    print_length_append("{}:".format(f), "ok", print_length)
                else:
                    pass
            else:
                print_length_append("{}:".format(f), "*** neighbours: {}".format(neighbours), print_length)

        print "{:20s}: {:17}".format("Total Number of Factors", len(self.factors))
        for key, val in factor_counter.iteritems():
            print "{:20s}: {:20}".format(key, val)
        print "Counted Edges: {}".format(edge_counter)
        print "Edges found from G.edges: {}".format(len(self.edges))
        print_new_line()

    def check_variable_nodes(self, print_all = False):

        unique_leaves = list()
        for leaf in self.leaf_nodes:
            leaf_name, leaf_number, _ = split_variable_name(leaf)
            if (leaf_name is not 'k') and not (leaf_name is 'p' and (leaf_number <= 16 or leaf_number > 144)):
                unique_leaves.append(leaf)

        # Just check size etc
        print "*** Checking Variable Nodes ***"
        print_new_line()
        print "Total Number of Variables: {:10}".format(len(self.variables))
        print "Leaf Nodes: {:25}".format(len(self.leaf_nodes))
        print "Unique Leaves: {:22} ({})".format(len(unique_leaves), unique_leaves)


    def update_edge(self, src, dst, np.ndarray val):
        self.G.edge[src][dst] = val

    def update_edge_list_inner(self, l):
        for a, b, c in l:
            self.update_edge(a, b, c)

    def update_edge_list(self, l):
        # Parallel(n_jobs=2)(delayed(self.update_edge_list_inner)(inner_l) for inner_l in l)
        for inner_l in l:
            self.update_edge_list_inner(inner_l)

    def bp_variable_pass(self, int rnd, int total_rounds, debug = False):
        # Variables send the product of their initial
        # distributions with all other incoming messages
        # cdef int rnds_left
        # rnds_left = total_rounds - rnd
        # for variable in self.variables:
        #     if debug:
        #         print "Variable Pass: {}".format(variable)
        #     # if variable not in self.left_out_nodes and (rnds_left > self.steps_from_key_length or self.can_reach_key_within_steps(variable, rnds_left)):
        #     #     self.bp_variable_handle(variable)
        #     self.bp_variable_handle(variable)

        # self.update_edge_list([self.bp_variable_handle(variable) for variable in self.variables])
        # pass
        # pool = mp.Pool()
        # for variable in self.variables:
        #     pool.apply_async(self.bp_variable_handle, args=(variable, ))
        # pool.close()
        # pool.join()

        self.update_edge_list(parmap(self.bp_variable_handle,self.variables))


    def bp_factor_pass(self, int rnd, int total_rounds, debug = False):
        # Factors compute their function on incoming messages
        # and pass it to target neighbours
        # cdef int rnds_left
        # rnds_left = total_rounds - rnd
        # for factor in self.factors:
        #     if debug:
        #         print "Factor Pass: {}".format(factor)
        #     # if factor not in self.dead_factor_nodes and (rnds_left > self.steps_from_key_length or self.can_reach_key_within_steps(factor, rnds_left)):
        #     #             #     self.bp_factor_handle(factor)
        #     self.bp_factor_handle(factor)

        # pass
        # pool = mp.Pool()
        # for factor in self.factors:
        #     pool.apply_async(self.bp_factor_handle, args=(factor, ))
        # pool.close()
        # pool.join()

        # self.update_edge_list([self.bp_factor_handle(factor) for factor in self.factors])
        self.update_edge_list(parmap(self.bp_factor_handle,self.factors))


    def bp_variable_handle(self, variable, print_updated_edge = False):

        # print_out = True if string_starts_with(variable, 't') else False
        print_out = False

        cdef np.ndarray v_product

        out_list = list()

        if variable not in self.left_out_nodes:

            for neighbour in self.get_neighbours(variable):

                # Get initial distribution
                initial_distribution = self.get_initial_distribution(variable)

                v_product = np.copy(initial_distribution[:])

                other_neighbours = self.get_other_neighbours(variable, neighbour)

                if print_out:
                    print ">>> Handling message from Variable {} to Factor {} <<<\n".format(variable, neighbour)

                for other_neighbour in other_neighbours:
                    if other_neighbour != neighbour:
                        # Take the product with the incoming message
                        incoming = self.get_incoming_message(variable, other_neighbour)
                        v_product = array_multiply(v_product, incoming)
                        # Quick Check Here
                        if CHECK_EMPTY and is_zeros_array(v_product):
                            print "EMPTY ARRAY FOUND, Variable", variable, "sending to neighbour", neighbour
                            print "Specifcally, incoming message from {}".format(other_neighbour)
                            print_new_line()
                            raise ValueError

                # Check product
                if CHECK_EMPTY and is_zeros_array(v_product):
                    print "EMPTY ARRAY FOUND, Variable", variable, "sending to neighbour", neighbour
                    raise ValueError

                if STOP_EMPTY_PROPAGATION and is_zeros_array(v_product):
                    v_product = get_no_knowledge_array()

                if print_updated_edge or print_out:
                    print "Updating Edge from Variable {} to Factor {}:\n{}\n".format(variable, neighbour, v_product[:4])

                # Set as new outgoing message
                # self.update_edge(variable, neighbour, v_product)
                out_list.append((variable, neighbour, v_product))

                # if print_out:
                #     print "Edge Now should be:\n{}\n".format(v_product)
                #     print "Now, edge:\n{}\n\n".format(self.get_outgoing_message(variable, neighbour))

        return out_list

    def bp_factor_handle(self, factor):

        # print_out = True if string_starts_with(factor, '_Xor_t001') else False
        print_out = False

        out_list = list()

        if factor not in self.dead_factor_nodes:

            for neighbour in self.get_neighbours(factor):

                # make sure neighbour isn't leaf node
                if neighbour not in self.leaf_nodes:

                    if print_out:
                        print ">>> Handling message from Factor {} to Variable {} <<<\n".format(factor, neighbour)

                    message = get_no_knowledge_array()
                    # Either XOR or SBOX: Handle separately
                    other_neighbours = self.get_other_neighbours(factor, neighbour)

                    if len(other_neighbours) == 0:

                        print "--- Factor {} sending to neighbour {}, doesn't have any other neighbours".format(factor, neighbour)
                        pass

                    # Handle each type of operation - we can generalise if we have too many here

                    elif is_xor_node(factor):
                        # XOR
                        message = self.get_incoming_message(factor, other_neighbours[0])
                        if print_out:
                            print "--->>> Message from neighbour {} ({}):\n{}\n".format(other_neighbours[0], message.dtype, message)
                        for i in range(1, len(other_neighbours)):
                            incoming_message = self.get_incoming_message(factor, other_neighbours[i])
                            if print_out:
                                print "--->>> Xoring message from neighbour {} ({}):".format(other_neighbours[i], incoming_message.dtype)
                            message = message_xor(message, incoming_message)
                            if print_out:
                                print "{}\n".format(message)

                    elif is_sbox_node(factor):
                        # SBOX - handle direction (not invertible)
                        if string_contains(neighbour, 's'):
                            message = message_sbox(self.get_incoming_message(factor, other_neighbours[0]))
                        else:
                            message = message_inv_sbox(self.get_incoming_message(factor, other_neighbours[0]))
                    elif is_xtimes_node(factor):
                        # XTIMES - handle direction (not invertible)
                        if string_contains(neighbour, 'xt'):
                            message = message_xtimes(self.get_incoming_message(factor, other_neighbours[0]))
                        else:
                            message = message_inv_xtimes(self.get_incoming_message(factor, other_neighbours[0]))

                    elif is_xor_constant_node(factor):
                        # XOR Constant: Same both directions - handle value
                        value = self.get_rc_from_factor(factor)
                        message = message_xor_constant(self.get_incoming_message(factor, other_neighbours[0]), value)

                    elif is_xor_xtimes_node(factor):
                        # Need to handle direction!
                        if string_contains(neighbour, 'xt'):

                            # XOR then XTIMES
                            message = message_xor(self.get_incoming_message(factor, other_neighbours[0]), self.get_incoming_message(factor, other_neighbours[1]))
                            message = message_xtimes(message)

                        elif string_contains(other_neighbours[0], 'xt'):

                            # invXtimes message from other_neighbours[0], then XOR
                            message = message_inv_xtimes(self.get_incoming_message(factor, other_neighbours[0]))
                            message = message_xor(message, self.get_incoming_message(factor, other_neighbours[1]))

                        else:

                            # invXtimes message from other_neighbours[1], then XOR
                            message = message_inv_xtimes(self.get_incoming_message(factor, other_neighbours[1]))
                            message = message_xor(message, self.get_incoming_message(factor, other_neighbours[0]))


                    else:
                        print "Factor node has no id: {}".format(factor)
                        # break
                        raise ValueError

                    # Quick Check Here
                    if CHECK_EMPTY and is_zeros_array(message):
                        print "EMPTY ARRAY FOUND, Factor", factor, "sending to neighbour", neighbour
                        raise ValueError

                    # Set as new outgoing message
                    # self.update_edge(factor, neighbour, message)
                    out_list.append((factor, neighbour, message))
        return out_list


    def bp_initial_pass(self):
        # Variables send their initial distributions
        for v in self.variables:
            self.send_initial_distribution(v)

    def send_initial_distribution(self, variable):
        for neighbour in self.get_neighbours(variable):
            self.update_edge(variable, neighbour, self.get_initial_distribution(variable))

    def bp_one_round(self, rnd, total_rounds):
        # Round consists of Variable Pass and a Factor Pass
        self.bp_variable_pass(rnd, total_rounds)
        self.bp_factor_pass(rnd, total_rounds)

    def bp_run(self, int rounds, print_all_messages = False, print_all_marginal_distributions = False,
               print_all_key_ranks = False, print_possible_values = False, print_marginal_distance = False,
               rank_order = False, break_when_found = True, break_when_information_exhausted_pattern = False,
               float epsilon = 0, int epsilon_s = INFORMATION_EXHAUSTED_S, break_if_failed = True,
               round_csv = False, snrexp = "UNKNOWN", update_key_initial_distributions = False, debug_mode = False):

        cdef int ranking_start, i, epsilon_succession

        # Variable to store when ranks will be saved
        ranking_start = max(0, rounds - CONVERGENCE_LENGTH)

        # Variable to store when converged
        round_converged = None
        final_state = None

        # Round Rank Storage
        if round_csv or break_when_information_exhausted_pattern:
            k_rank_order = []
            for i in range(len(self.key_nodes)):
                k_rank_order.append([0] * (rounds + 1))

        previous_marginal_distributions = None
        epsilon_succession = 0
        failed = False

        round_found = None

        for i in range(rounds+1):

            # Run
            if i > 0:
                self.bp_one_round(i, rounds+1)

            if (print_all_messages or print_all_marginal_distributions or print_all_key_ranks or print_marginal_distance or print_possible_values) and not self.no_print:
                print "----------- Running Round {} -----------".format(i)
                print_new_line()

            if print_all_messages:
                self.print_all_messages()
                print_new_line()

            if print_all_marginal_distributions:
                self.print_all_marginal_distributions()
                print_new_line()

            if print_possible_values:
                self.print_all_possible_values()
                print_new_line()

            if print_all_key_ranks:
                self.print_key_rank()
                print_new_line()

            if round_csv or break_when_information_exhausted_pattern:
                for j, node in enumerate(self.key_nodes):
                    key_rank = self.get_key_rank(j+1)
                    if round_csv:
                        # Store subkey rank in csv
                        append_csv('Output/SubkeyRoundRank.csv',
                                   '{}, {}, {}, {}\n'.format(snrexp, node, i, key_rank[0] + key_rank[1]))
                    k_rank_order[j][i] = key_rank[0] + key_rank[1]

            if print_marginal_distance and i > 0:
                self.print_marginal_distance(previous_marginal_distributions)
                print_new_line()

            # Break if found
            if break_when_found and self.found_key():
                if not self.no_print:
                    print '+++++++ Found the Key at Round {} +++++++'.format(i)
                final_state = "foundkey"
                break

            # Break if failed
            if break_if_failed and (i > FAILURE_MIN_ROUNDS) and (self.check_plaintext_failure() or self.check_failure_on_specific_byte('t')):
                if not self.no_print:
                    print '!!!!!!!!!! FAILED at Round {} !!!!!!!!!!'.format(i)
                final_state = "failed"
                break

            # Break if all key bytes converged or repeating a pattern
            if break_when_information_exhausted_pattern and self.information_exhausted_pattern(k_rank_order, i):
                if PRINT_EXHAUSTION and not self.no_print:
                    print '+++ Information Exhausted at Round {} (Pattern Matched) +++'.format(i)
                final_state = "patternexhaust"
                break

            if (epsilon > 0) and (i > 0):
                if self.information_exhausted_epsilon(previous_marginal_distributions, epsilon):
                    epsilon_succession += 1
                else:
                    epsilon_succession = 0
                if epsilon_succession > epsilon_s:
                    if PRINT_EXHAUSTION and not self.no_print:
                        print '+++ Information Exhausted at Round {} (Below Epsilon Threshold after {} Successions) +++'.format(i, INFORMATION_EXHAUSTED_S)
                    # round_converged = i - epsilon_s + 1
                    # round_found = i - epsilon_s + 1
                    round_converged = i
                    round_found = i
                    final_state = "epsilonexhaust"
                    break

            previous_marginal_distributions = self.get_marginal_distributions_of_key_bytes()

            # Update Key Inital Distributions if set
            if update_key_initial_distributions:
                self.set_key_distributions(self.get_marginal_distributions_of_key_bytes())

            # if i > 0:
            #     print "Ending After First Round!"
            #     sys.exit(1)

        else:
            round_converged = rounds
            final_state = "maxediterations"

        if self.found_key():
            if round_found is None:
                # Maxed Iterations
                round_found = rounds
        else:
            round_found = None

        if rank_order:

            output_string = "Rank Order for {} Traces, {} Rounds, SNR = 2**{}\n\n".format(self.get_number_of_traces(), rounds, self.get_snr_exp())

            for i in range(len(self.key_nodes)):
                smaller_rank_order = k_rank_order[i][ranking_start:]
                mode = get_mode(smaller_rank_order)

                string = "Key Byte {}: {}".format(i+1, smaller_rank_order)
                if not self.no_print:
                    print string
                output_string += string + "\n"

                string = "Max Rank: {}".format(max(smaller_rank_order))
                if not self.no_print:
                    print string
                output_string += string + "\n"

                string = "Min Rank: {}".format(min(smaller_rank_order))
                if not self.no_print:
                    print string
                output_string += string + "\n"

                string = "Average Rank: {}".format(get_average(smaller_rank_order))
                if not self.no_print:
                    print string
                output_string += string + "\n"

                string = "Mode Rank: {}".format(mode)
                if not self.no_print:
                    print string
                output_string += string + "\n"

                string = "Final Rank: {}".format(smaller_rank_order[-1])
                if not self.no_print:
                    print string
                output_string += string + "\n"

                if not self.no_print:
                    print_new_line()
                    print_new_line()

            # Write output_string to file
            f = open("Output/RankOrder_{}Traces_{}Rounds_{}SNRExp.txt".format(self.get_number_of_traces(), rounds, self.get_snr_exp()), "w")
            f.write(output_string)
            f.close()


        # Return
        return final_state, round_converged, round_found

    def get_possible_values(self, node, supplied_dist = None):
        if supplied_dist is not None:
            # Take from input
            marginal_dist = supplied_dist
        else:
            # Get Marginal Distribution of node
            marginal_dist = self.get_marginal_distribution(node)
        # Loop through, find possible values
        possible_values = list()
        current_probability = 0
        for i in range (len(marginal_dist)):
            if marginal_dist[i] > current_probability:
                # Update
                current_probability = marginal_dist[i]
                possible_values = [i]
            elif marginal_dist[i] == current_probability:
                # Append
                possible_values.append(i)

        if len(possible_values) == 256:
            return []
        return possible_values

    def get_key_probability(self, k_number, supplied_dist = None):
        # Get Marginal Dist
        k = self.key_nodes[k_number-1]
        if supplied_dist is not None:
            marginal_dist = supplied_dist[k_number-1]
        else:
            marginal_dist = self.get_marginal_distribution(k)
        # Find Rank
        probability = marginal_dist[self.key[k_number-1]]
        return probability

    def get_final_key_probability(self, supplied_dist = None):
        key_names = self.key_nodes
        # Container
        rank_product = 1
        for i in range(len(key_names)):
            rank_product *= self.get_key_probability(i, supplied_dist)
        return rank_product

    def get_key_rank(self, k_number, supplied_dist = None):
        # Get Marginal Dist
        # k = 'k{}-K'.format(pad_string_zeros(str(k_number)))
        k = self.key_nodes[k_number-1]
        if supplied_dist is not None:
            marginal_dist = supplied_dist[k_number-1]
        else:
            marginal_dist = self.get_marginal_distribution(k)
        # Find Rank
        # print "marginal_dist of {}:\n{}".format(k, marginal_dist)

        probability = marginal_dist[self.key[k_number-1]]

        rank = 1
        duplicate = 0

        for i in range (len(marginal_dist)):

            if marginal_dist[i] > probability:
                rank += 1
                # print "Comparing marginal_dist[{}] = {} to probability = {}: marginal_dist greater!".format(i, marginal_dist[i], probability)
            elif marginal_dist[i] == probability and i != self.key[k_number-1]:
                duplicate += 1
                # print "Comparing marginal_dist[{}] = {} to probability = {}: duplicate found!".format(i, marginal_dist[i], probability)

        return rank, duplicate, probability

    def check_failure_on_specific_byte(self, var_name, debug_mode = False):
        # TODO - Deprecated
        return False
        # failure_threshold = (FAILURE_THRESHOLD_AT_NODE * (2**self.get_snr_exp()))
        # if debug_mode:
        #     print "Checking Failure for Var {}\n-> Fail if smaller than {} * (2**{}) = {}".format(var_name, FAILURE_THRESHOLD_AT_NODE, self.get_snr_exp(), failure_threshold)
        # # Get all variables that match between 1 and 16
        # target_list = ['{}{}-0'.format(var_name, pad_string_zeros(i)) for i in range(1,17)]
        # # For every variable, get initial distribution, and also the product of all incoming messages
        # distance_list = [0] * 16
        # for i, target_var in enumerate(target_list):
        #     target_initial_dist = self.get_initial_distribution(target_var)
        #     target_incoming_messages = self.get_all_incoming_messages(target_var)
        #     product = array_multiply(target_initial_dist, target_incoming_messages)
        #     distance = euclidian_distance(target_initial_dist, target_incoming_messages)
        #     distance_list[i] = distance
        #     if debug_mode:
        #         print "* Target Var {} *\n".format(target_var)
        #         print "Top Values of Initial Distribution: {}".format(get_top_values(target_initial_dist))
        #         print "Top Values of Incoming Messages:    {}".format(get_top_values(target_incoming_messages))
        #         print "Top Values of Product (Marginal):   {}".format(get_top_values(product))
        #         print "Distance Between the Two Distributions: {}".format(distance)
        #         print_new_line()
        #     if distance < failure_threshold:
        #         return True
        # if debug_mode:
        #     print_statistics(distance_list)
        # # print "Failure Threshold: {}".format(FAILURE_THRESHOLD_AT_NODE * self.get_snr_exp())
        # return False
        # # sys.exit(1)

    def check_plaintext_failure(self, snr = None, debug_mode = False):
        if debug_mode:
            print "&&& Checking Plaintext Failure &&&"
        cdef int i
        cdef int failed_plaintexts
        failed_plaintexts = 0
        cdef np.ndarray probability_list = np.zeros(len(self.plaintext_nodes))
        cdef np.ndarray incoming

        if snr is None:
            snr = 2 ** self.get_snr_exp()

        try:
            failure_threshold = FAILURE_THRESHOLD_DICT[snr]
        except KeyError:
            failure_threshold = 1e-5

        for i, p in enumerate(self.plaintext_nodes):
            correct_val = find_value_in_list(self.get_initial_distribution(p), 1.0)
            if correct_val < 0:
                # Can't tell if plaintext fail, if not leaking on plaintext...
                return False



            # if CPF_INCLUDE_K:
            #     # Get names for adjacent t, k, and XOR
            #     k_val = 84; t_val = k_val ^ correct_val
            #     #
            #     xor_node = self.get_neighbours(p)[0]
            #     neighbours = (self.get_other_neighbours(xor_node, p))
            #     neighbours.sort()
            #     k_node, t_node = neighbours
            #     t_to_xor = self.get_incoming_message(xor_node, t_node)
            #     xor_to_k = self.get_incoming_message(k_node, xor_node)
            #     incoming = arrayXOR(t_to_xor, xor_to_k)
            #     print "Coming From t to XOR:\n\n{}\n".format(t_to_xor)
            #     print "Correct Value for t: {}, Top Values: {}".format(t_val, get_top_values(t_to_xor))
            #     print "Coming From XOR to k:\n\n{}\n".format(xor_to_k)
            #     print "Correct Value for k: {}, Top Values: {}".format(k_val, get_top_values(xor_to_k))
            #     print "XORd together:\n\n{}\n".format(incoming)
            #     print "Correct Value for p: {}, Top Values: {}".format(correct_val, get_top_values(incoming))
            #     sys.exit(1)
            # else:
            incoming = self.get_all_incoming_messages(p)


            if debug_mode:
                print "For Plaintext Byte {}, correct value = {:3}, rank {:3}, probability found = {}".format(p,
                                                                                                              correct_val,
                                                                                                              get_rank_from_prob_dist(
                                                                                                                  incoming,
                                                                                                                  correct_val),
                                                                                                              incoming[
                                                                                                                  correct_val])

            if incoming[correct_val] == 0.0:
                return True

            if incoming[correct_val] < failure_threshold:
                if debug_mode:
                    print "OH NO! incoming message stats:\nMax:       {}\nMin:          {}\nCorrect Val: {}\nSum:          {}".format(
                        array_max(incoming), array_min(incoming), incoming[correct_val], sum(incoming))
                # return True
                failed_plaintexts += 1

            probability_list[i] = incoming[correct_val]

        if debug_mode:
            print_statistics(probability_list)

        if failed_plaintexts >= FAILURE_NUMBER_OF_PLAINTEXTS:
            return True
        else:
            return False

    def get_ground_truths(self):
        # Returns Euclidian Distances between plaintext initial distributions (known value) and incoming messages
        cdef float [16] distances
        cdef int i
        for i, p in enumerate(self.plaintext_nodes):
            # print "Plaintext Node {}:\nInitial: {}\nIncoming: {}\n".format(p, self.get_initial_distribution(p), self.get_all_incoming_messages(p))
            distances[i] = euclidian_distance(self.get_initial_distribution(p), self.get_all_incoming_messages(p))
        return distances

    def found_key(self, supplied_dist = None):
        cdef int i, rank, duplicate
        cdef float prob
        if self.check_plaintext_failure():
            return False
        # Returns true if all sub-keys are ranked 1st with no duplicates
        for i in range(1,len(self.key_nodes)+1):
            rank, duplicate, prob = self.get_key_rank(i, supplied_dist)
            if (rank != 1) or (duplicate != 0): return False
        else:
            return True

    def information_exhausted_pattern(self, k_rank_order, int i):
        # Returns true if all information has been exhausted
        # e.g. If all key bytes have either converged or are repeating a set pattern (within the last 40 rounds)
        if i < 50:
            return False

        # print "* Checking if Information is Exhausted *"

        for key_byte in range(len(self.key_nodes)):
            # print "Checking Key Byte {}, Rank Order:\n{}".format(key_byte, k_rank_order[key_byte][i-INFORMATION_EXHAUSTED_THRESHOLD:i])
            if get_repeating_pattern(k_rank_order[key_byte][i - INFORMATION_EXHAUSTED_THRESHOLD:i]) is None:
                # print "Key Byte {} has NOT been exhausted!".format(key_byte)
                return False
            else:
                pass
                # print "Key Byte {} has been exhausted!".format(key_byte)

        # print "* All Information has been Exhausted! *"
        return True

    def information_exhausted_epsilon(self, previous_marginal_distributions, epsilon = 0.0015):
        # Returns true if all information has changed less than epsilon -> negligible impact on key distribution
        current_distributions = self.get_marginal_distributions_of_key_bytes()
        all_pass = True
        # Get difference between the two
        for i in range(len(self.key_nodes)):
        # for i in range(1):
            # print "* Key Byte {}:".format(i)
            difference = all_positive(array_subtract(current_distributions[i], previous_marginal_distributions[i]))
            # Comment on it
            # print "Difference: {}".format(difference)
            if max(difference) >= epsilon:
                return False
        return True

    def get_longest_cycle(self):
        longest_path = []
        # depth_first_search
        for node in self.variables:
            path = self.get_longest_cycle_from_node(node)
            if len(path) > len(longest_path):
                longest_path = path
        return longest_path

    def get_longest_cycle_from_node(self, node):
        # Sequential depth_first_search
        # For each node, get neighbours, and add to edge
        return self.depth_first_search(node, node, [node])

    def get_nodes_reachable_from_key(self, steps = 1):

        reachable_variable_nodes = set(self.key_nodes)
        reachable_factor_nodes = set()

        for i in range(steps):

            # Get all factor node neighbours from variable set
            for var in reachable_variable_nodes:
                neighbours = self.get_neighbours(var)
                for neighbour in neighbours:
                    reachable_factor_nodes.add(neighbour)

            # Get all variable node neighbours from factor set
            for fac in reachable_factor_nodes:
                neighbours = self.get_neighbours(fac)
                for neighbour in neighbours:
                    reachable_variable_nodes.add(neighbour)

        return set(sorted(reachable_variable_nodes)), set(sorted(reachable_factor_nodes))

    def get_nodes_reachable_from_variable_node(self, start, steps = 1):

        reachable_variable_nodes = {start}
        reachable_factor_nodes = set()

        for i in range(steps):

            # Get all factor node neighbours from variable set
            for var in reachable_variable_nodes:
                neighbours = self.get_neighbours(var)
                for neighbour in neighbours:
                    reachable_factor_nodes.add(neighbour)

            # Get all variable node neighbours from factor set
            for fac in reachable_factor_nodes:
                neighbours = self.get_neighbours(fac)
                for neighbour in neighbours:
                    reachable_variable_nodes.add(neighbour)

        return set(sorted(reachable_variable_nodes))

    def get_nodes_reachable_from_factor_node(self, start, steps = 1):

        reachable_variable_nodes = set()
        reachable_factor_nodes = {start}

        for i in range(steps):

            # Get all variable node neighbours from factor set
            for fac in reachable_factor_nodes:
                neighbours = self.get_neighbours(fac)
                for neighbour in neighbours:
                    reachable_variable_nodes.add(neighbour)

            # Get all factor node neighbours from variable set
            for var in reachable_variable_nodes:
                neighbours = self.get_neighbours(var)
                for neighbour in neighbours:
                    reachable_factor_nodes.add(neighbour)

        return set(sorted(reachable_variable_nodes))

    def get_steps_from_key_lists(self, variable = True):
        v_lst = [0] * self.steps_from_key_length
        f_lst = [0] * self.steps_from_key_length

        for i in range(self.steps_from_key_length):
            v_lst[i], f_lst[i] = self.get_nodes_reachable_from_key(steps = i)

        return v_lst, f_lst


    def can_reach_key_within_steps(self, node, steps):
        if steps >= self.steps_from_key_length:
            return True
        if (string_starts_with(node, '_') and node in self.steps_from_key_factors[steps]) or (not string_starts_with(node, '_') and node in self.steps_from_key_variables[steps]):
                return True
        return False

    def get_length_between_nodes(self, node1, node2):
        my_max = 42
        for i in range(my_max):
            reachable_nodes = self.get_nodes_reachable_from_variable_node_after_steps(node1, i)
            if node2 in reachable_nodes:
                return i
        else:
            print "Couldn't find {} from {} within {} steps".format(node2, node1, my_max)
            return -1

    def print_nodes_reachable_from_variable_node_after_series_of_steps(self, start, steps = 5):
        print "*** Nodes Reachable from Variable {} ***".format(start)
        for i in range(steps+1):
            print "* Step {} *".format(i)
            print self.get_nodes_reachable_from_variable_node_after_steps(start, i)

    def get_nodes_reachable_from_variable_node_after_steps(self, start, steps = 1):

        if steps <= 0:
            return {start}

        a = self.get_nodes_reachable_from_variable_node(start, steps)
        b = self.get_nodes_reachable_from_variable_node(start, steps-1)

        return sorted(a - b)

    def get_minimum_rounds_needed(self, check_all_key_bytes = False):

        # Compute the minimum rounds required to pass all information to key bytes.
        minimum_rounds = 0

        if check_all_key_bytes:
            for key_byte in self.key_nodes:
                key_min = self.get_minimum_rounds_needed_for_variable(key_byte)
                if key_min > minimum_rounds:
                    minimum_rounds = key_min
        else:
            minimum_rounds = self.get_minimum_rounds_needed_for_variable(self.key_nodes[0])

        return minimum_rounds

    def get_minimum_rounds_needed_for_variable(self, node):
        minimum_rounds = 0
        all_variables = sorted(self.variables)
        # print "* Key Byte: {}".format(key_byte)
        for count in range(len(self.variables) + 1):
            reachable_nodes = sorted(self.get_nodes_reachable_from_variable_node(node, count))
            if reachable_nodes == all_variables:
                # print "Minimum Rounds needed: {}".format(count)
                if count > minimum_rounds:
                    minimum_rounds = count
                break
        return minimum_rounds


    def depth_first_search(self, start_node, current_node, visited_nodes):

        # print "* depth_first_search Depth {}: current_node = {}, visited_nodes = {}".format(len(visited_nodes), current_node, visited_nodes)

        # First, get all neighbours of current_node
        neighbours = self.get_other_neighbours(current_node, visited_nodes[-1])

        # Check to see if start node is current_node (but has to have traversed somewhere!)
        if start_node == current_node and len(visited_nodes) > 1:
            # Check if not cheated and gone back to start node with only one hop
            if len(visited_nodes) == 3: # start, neighbour, start
                return []
            return visited_nodes

        # Handle only one place to go
        if len(neighbours) == 1:
            # if already been to this node, then dead end!
            if neighbours[0] in visited_nodes:
                return []
            # Otherwise, go straight to that node
            return self.depth_first_search(start_node, neighbours[0], visited_nodes + [neighbours[0]])

        # Set longest path variable
        longest_path = []

        # Go through neighbours
        for neighbour in neighbours:
            # Check if not in visited_nodes
            if neighbour not in visited_nodes or neighbour == start_node:
                # Add to visited_nodes and traverse, whilst it returns list of edges
                dfs_result = self.depth_first_search(start_node, neighbour, visited_nodes + [neighbour])
                if len(dfs_result) > len(longest_path):
                    longest_path = dfs_result

        # Return longest_path
        return longest_path

    def get_all_incoming_messages(self, node):
        # Product of initial distribution with all other messages
        cdef np.ndarray incoming_messages = get_no_knowledge_array()
        for neighbour in self.get_neighbours(node):
            # Take the product with the incoming message
            incoming_messages = array_multiply(incoming_messages, self.get_incoming_message(node, neighbour))
        return incoming_messages

    def get_marginal_distribution(self, node, debug_mode = False):
        # cdef float [256] marginal_dist
        # cdef float [256] incoming
        cdef np.ndarray marginal_dist = get_no_knowledge_array()
        cdef np.ndarray incoming = get_no_knowledge_array()

        # Product of initial distribution with all other messages

        try:
            marginal_dist = np.copy(self.get_initial_distribution(node)[:])
        except TypeError:
            print "!!! ERROR: TypeError Encountered in FactorGraphAES, slicing initial distribution of {} but self.get_initial_distribution(node) = {}".format(node, self.get_initial_distribution(node))
            print_new_line()
            raise

        if node == "k005-K" and debug_mode:
            print "\n-=-=- Getting Marginal Distribution for {} -=-=-".format(node)
            print "115:{}".format(marginal_dist[115])

        for neighbour in self.get_neighbours(node):
            # Take the product with the incoming message
            # print "Marginal Dist for node {}, multiplying by neighbour {}: {}".format(node, neighbour, self.get_incoming_message(node, neighbour))
            marginal_dist = array_multiply(marginal_dist, self.get_incoming_message(node, neighbour))
            if (node == "k005-K") and debug_mode:
                incoming = np.copy(self.get_incoming_message(node, neighbour))
                print "-> Incoming Message from {}: \n----->Most likely Val {} with probability {}\n----->115 has probability {}, difference = {}".format(
                    neighbour, max_index(incoming), max(incoming), incoming[115], max(incoming) - incoming[115])
                print "--> New Marginal Dist (after mult):\n----->Most likely Val {} with probability {}\n------>115 has probability {}, difference = {}".format(
                    max_index(marginal_dist), max(marginal_dist), marginal_dist[115],
                    max(marginal_dist) - marginal_dist[115])

        return marginal_dist

    def get_marginal_distributions_of_key_bytes(self):
        # Save all marginal distributions of key bytes as list
        cdef int i
        cdef np.ndarray key_distributions = np.zeros((len(self.key_nodes), 256), dtype=DTYPE)
        # cdef np.ndarray key_distributions = np.zeros((len(self.key_nodes), 256))

        for i in range (len(self.key_nodes)):
            k = 'k{}-K'.format(pad_string_zeros(str(i + 1)))
            key_distributions[i] = self.get_marginal_distribution(k)

        return key_distributions

    def get_marginal_distributions_of_plaintext_bytes(self, trace = 0):
        # Save all marginal distributions of key bytes as list
        cdef int i
        cdef np.ndarray p_distributions = np.zeros((len(self.key_nodes), 256))

        for i in range (len(self.key_nodes)):
            p = "p{}-{}".format(pad_string_zeros(str(i + 1)), trace)
            p_distributions[i] = self.get_all_incoming_messages(p)

        return p_distributions

    def get_rc_from_factor(self, factor):
        # Split name to get number of factor
        key_target = factor.split('_')[2]
        _, key_number, _ = split_variable_name(key_target)
        return rcon[((key_number - 1) / 16) - 1]


    # ************************************ PRINTING FUNCTIONS ************************************

    def print_all_messages(self):
        print "*** Printing All Messages ***"
        print_new_line()
        for node in self.G.nodes():
            print "From", node
            print_new_line()
            neighbours = self.get_neighbours(node)
            for neighbour in neighbours:
                print "-> to", neighbour, ":", self.get_outgoing_message(node, neighbour)
                print_new_line()
        print_new_line()

    def print_all_variables(self):

        print "*** Printing Variables and their Neighbours ***"
        print_new_line()

        numbers = ['001','017','144','145','160','161']

        for v in self.variables:

            if string_contains_any(v, numbers):
                print v
                neighbours = sorted(self.get_neighbours(v))
                for n in neighbours:
                    print "---->", n
                print_new_line()

        print "***********************************************"

    def print_all_initial_distributions(self):
        print "*** Printing All Initial Distributions ***"
        print_new_line()
        for v in self.variables:
            print v, ":", self.get_initial_distribution(v)
            print_new_line()

    def print_all_marginal_distributions(self):
        print "*** Printing All Marginal Distributions ***"
        print_new_line()
        for v in self.variables:
            if string_ends_with(v, '-K') and string_contains(v, '005'):
                print v, ":", self.get_marginal_distribution(v, debug_mode = True)
                print_new_line()
                # print "Prob:", convert_to_probability_distribution(self.get_marginal_distribution(v))
                # print_new_line()

    def print_marginal_distance(self, previous_marginal_distributions, distance_threshold = 0.0015):
        print "*** Printing All Marginal Distances ***"
        print_new_line()
        current_distributions = self.get_marginal_distributions_of_key_bytes()

        for i, dist in enumerate(previous_marginal_distributions):
            previous_marginal_distributions[i] = convert_to_probability_distribution(dist)
        for i, dist in enumerate(current_distributions):
            current_distributions[i] = convert_to_probability_distribution(dist)

        all_pass = True

        total_max = 0
        total_max_i = -1
        total_min = 1
        total_min_i = -1
        total_avg = 0


        # Get difference between the two
        for i in range(len(self.key_nodes)):
        # for i in range(1):

            difference = all_positive(array_subtract(current_distributions[i], previous_marginal_distributions[i]))
            # Comment on it

            my_max = max(difference)
            my_min = min(difference)
            my_avg = get_average(difference)

            if my_max > total_max:
                total_max = my_max
                total_max_i = i
            if my_min < total_min:
                total_min = my_min
                total_min_i = i
            total_avg += my_avg

            passed = max < distance_threshold

            if i == 4:
                print "* Key Byte {} ({}):".format(i+1, self.key_nodes[i])
                # print "Previous: {}".format(previous_marginal_distributions[i])
                # print "Current: {}".format(current_distributions[i])
                # print "Difference: {}".format(difference)
                print "Max: {}\nMin: {}\nAvg: {}\nBelow Threshold: {}".format(my_max, my_min, my_avg, passed)
                print "Correct key 115, value {}".format(current_distributions[i][115])
                print "Maximum probability {} (Index {})".format(max(current_distributions[i]),
                                                                 max_index(current_distributions[i]))
                print_new_line()

            if not passed:
                all_pass = False


        print "* Over all Key Bytes:"
        print "Max: {} ({})\nMin: {} ({})\nAvg: {}\n".format(total_max, total_max_i, total_min, total_min_i, total_avg / len(self.key_nodes))

        if all_pass:
            print "*** All Values Below Threshold, Can Conclude Exhaustion ***"
        else:
            print "--- Not all Values below Threshold ---"

    def print_all_possible_values(self):
        print "*** Printing All Possible Values ***"
        print_new_line()
        # Print Possible values
        for var in self.variables:
            var_name, var_number, var_trace = split_variable_name(var)
            if var_number < 3:
                print "{}: {}".format(var, self.get_possible_values(var))
        print_new_line()

    def print_key_rank(self, supplied_dist = None, print_line = False, martin=False):
        if print_line:
            print "*** Printing Key Ranks ***"
            print_new_line()

        rank_product = 1
        for i in range(1,len(self.key_nodes) + 1):
            index = pad_string_zeros(i, 3)
            rank, duplicate, value = self.get_key_rank(i, supplied_dist)
            rank_product *= (rank+duplicate)
            l = 15
            if value == 0:
                # print "k{} FAILED - Value 0 (Ranked {} with {} duplicates)".format(index, rank, duplicate)
                print "k%3s %8s (Ranked %3s with %3s duplicates) [%s]" % (index, 'FAILED', rank, duplicate, value)
            else:
                # print "k{} Rank {} (Ranked {} with {} duplicates, value = {})".format(index, rank+duplicate, rank, duplicate, value)
                print "k%3s Rank %3s (Ranked %3s with %3s duplicates) [%s]" % (index, rank+duplicate, rank, duplicate, value)

        if martin:
            # Different
            rank_product = self.get_final_key_rank(martin)
        print "Estimated Whole Key Rank: {} (~2^{})".format(rank_product, bit_length(rank_product))
        print_new_line()

    def get_final_key_rank(self, martin=False, supplied_dist = None):
        if martin:
            all_dists = self.get_marginal_distributions_of_key_bytes()
            print "...computing Martin Key Rank, please wait..."
            return martin_rank(all_dists)
        else:
            # Container
            rank_product = 1

            for i in range(1,len(self.key_nodes) + 1):
                index = pad_string_zeros(i, 3)
                rank, duplicate, value = self.get_key_rank(i, supplied_dist)
                rank_product *= (rank+duplicate)
                # print 'Rank {}, Duplicate {}, Value {}, New Rank Product: {}'.format(rank, duplicate, value, rank_product)

            return rank_product


if __name__ == "__main__":

    G = FactorGraphAES(key_scheduling = False)

    print G

    TEST_KEY_SCHEDULING = False
    TEST_REMOVING_NODES = False

    # Test Key Scheduling
    if TEST_KEY_SCHEDULING:

        print "*** Without Key Scheduling ***"
        print_new_line()
        G_without = FactorGraphAES(key_scheduling = False)

        print "*** With Key Scheduling ***"
        print_new_line()
        G_with = FactorGraphAES(key_scheduling = True)
        print_new_line()

        G_with.check_factor_nodes()

    # Test Removing Nodes
    if TEST_REMOVING_NODES:

        print "*** No Removal (Excluding Key Scheduling) ***"
        print_new_line()
        G = FactorGraphAES()

        print "*** Removing cm node ***"
        print_new_line()
        G_no_cm = FactorGraphAES(removed_nodes = ['cm'])

        print "*** Removing xa node ***"
        print_new_line()
        G_no_xa = FactorGraphAES(removed_nodes = ['xa'])

        print "*** Removing xb node ***"
        print_new_line()
        G_no_xb = FactorGraphAES(removed_nodes = ['xb'])

        print "*** Removing cm, xa, xb nodes ***"
        print_new_line()
        G_no_all = FactorGraphAES(removed_nodes = ['cm','xa','xb'])
