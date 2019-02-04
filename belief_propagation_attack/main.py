import sys
import os
import os.path
import argparse
import factorGraphAES as fG
from utility import *
import cPickle as Pickle
from datetime import datetime
import timing
import matplotlib.pyplot as plt

CHOSEN_KEY = [0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75]

def run_belief_propagation_attack(margdist=None):
    global key_distributions
    if margdist is not None:
        fixed_node_tuple = (margdist[0], margdist[1])
    else:
        fixed_node_tuple = None

    if not LEAKAGE_ON_THE_FLY:

        trace_files = 0

        if LOCAL_LEAKAGE:
            directory = 'Leakage/'
            trace_files -= 3  # 3 other files, not traces
            if ELMO_POWER_MODEL:
                directory += 'ELMOPowerModel/'
            else:
                directory += 'HWModel/'
        else:
            directory = '$HOME/Desktop/ELMO/output/traces/'

        if not NO_PRINT:
            print "Leakage Directory: {}".format(directory)

        trace_files += len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

        trace_files_required = REPEAT * TRACES

        if trace_files_required > trace_files:
            print "^^^ ERROR: {} Trace Files required ({} Traces * {} Repeats), " \
                  "but only found {}! Aborting. ^^^".format(
                trace_files_required, TRACES, REPEAT, trace_files)
            exit(1)

        if not NO_PRINT:
            print "> {} Trace Files Found, Using {}\n".format(trace_files, trace_files_required)

    # else:
    #
    #     if not NO_PRINT and not REAL_TRACES:
    #         print "> Simulating {} Leakage Trace(s) in advance ({} Traces * {} Repeats)".format(TRACES * REPEAT, TRACES,
    #                                                                                             REPEAT)

    ################################################################################

    # key = []
    # key = [0, 1, 128, 3, 192, 7, 224, 15, 240, 31, 248, 63, 252, 127, 254, 255]
    if ELMO_POWER_MODEL and LOCAL_LEAKAGE and READ_PLAINTEXTS:
        key = np.array([0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46,
                        0x75])  # CORRECT KEY
    elif TEST_KEY:
        key = np.zeros(16)
    elif RANDOM_KEY:
        # Randomise Key with respect to the seed
        key = get_random_bytes(seed=SEED)
    else:
        # Can play around here
        key = np.array(CHOSEN_KEY)  # CORRECT KEY
        # key = [0x00, 0xA3, 0xFC, 0x47, 0x37, 0x02, 0xD6, 0x97, 0x02, 0xB4, 0x57, 0xE6, 0x76, 0x02, 0x63, 0x57]

    # Set up Signal to Noise Ratio
    snr = float(2 ** SNR_exp)
    sigma = get_sigma(snr)

    badly_leaking_snr = 2 ** BADLY_LEAKING_SNR_exp

    ################################################################################

    ################################################################################

    print_fill_1 = ".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-."
    print_fill_2 = "+-+-+-+-+-+-+-+-+"
    graph_string = 'G{}{}{}'.format(ROUNDS_OF_AES, 'A' if REMOVE_CYCLE else '', 'KS' if INCLUDE_KEY_SCHEDULING else '')

    if not NO_PRINT:
        print "\nRunning BP on {}: {} Trace(s) and {} Round(s), Averaging over {} Pass(es)".format(graph_string, TRACES,
                                                                                                   ROUNDS, REPEAT)
        print "Key Scheduling: {}, Removed Nodes: {}, Not Leaking Nodes: {}, Badly Leaking Nodes: {} (Traces {}, SNR 2^{}), " \
              "Left Out Nodes: {}, Not Noisy Nodes: {}".format(
            INCLUDE_KEY_SCHEDULING, REMOVED_NODES, NOT_LEAKING_NODES, BADLY_LEAKING_NODES, BADLY_LEAKING_TRACES,
            BADLY_LEAKING_SNR_exp,
            LEFT_OUT_NODES, NO_NOISE_NODES)
        if fixed_node_tuple is None:
            print "No Fixed Node(s)."
        else:
            print "Fixed Node: {}, Fixed Value: {}".format(FIXED_VALUE_NODE, fixed_node_tuple[1])
        print "snr = 2^{} = {} (sigma = {}), Threshold: {}".format(SNR_exp, snr, sigma, THRESHOLD)
        print "Epsilon = {}, Epsilon Successions: {}".format(EPSILON, EPSILON_S)
        print "Using REAL TRACES: {} (Jitter {})".format(REAL_TRACES, JITTER)
        print "Using LDA: {}, Using Neural Networks: {} (Window Size {}), Using Best Template: {}".format(USE_LDA, USE_NN, TPRANGE, USE_BEST)
        if REAL_TRACES:
            print "Correlation Threshold: {}".format(CORRELATION_THRESHOLD)
        print "If Simulating Data, Using ELMO Power Model: {}, Leakage on the Fly: {}, Reading Plaintexts: {}".format(ELMO_POWER_MODEL,
                                                                                                  LEAKAGE_ON_THE_FLY,
                                                                                                  READ_PLAINTEXTS)
        print "Seed: {}".format(SEED)
        print_new_line()
        print "Key:\n{}\n".format(key)
        print "Key Hex String:\n{}\n".format(get_list_as_hex_string(key))
        print "No Noise Power Modelled Key Values:\n{}\n".format(
            get_power_modelled_key_values(key, elmo=ELMO_POWER_MODEL))
        # if REAL_TRACES:
        #     print "Observed Power Modelled Key Values: NONE, cannot get power values from Real Traces!\n"
        # else:
        #     print "Observed Power Modelled Key Values:\n{}\n".format(all_values[0]['k'][:16])

    ################################################################################

    # ===============================================================================

    connecting_methods = []
    if INDEPENDENT_GRAPHS:
        connecting_methods.append('IND')
    if SEQUENTIAL_GRAPHS:
        connecting_methods.append('SEQ')
    if LARGE_FACTOR_GRAPH:
        connecting_methods.append('LFG')

    connecting_dict = {"LFG": "Large Factor Graph", "SEQ": "Sequential Graphs", "IND": "Independent Graphs"}

    if len(connecting_methods) == 0:
        print "!!! No connecting methods selecting, aborting"
        exit(1)
    for method in connecting_methods:

        if not NO_PRINT:
            print print_fill_1, "Starting {}{}".format(connecting_dict[method], ', Key Averaging {}'.format(KEY_POWER_VALUE_AVERAGE) if (method == "IND" or method == "SEQ") else ''), print_fill_1
            print_new_line()

        final_key_ranks = []
        final_key_ranks_averaged = []
        final_key_rank_martin = 0
        traces_required = []
        total_failures = 0
        total_maxed_iterations = 0
        total_epsilon_exhaustion = 0
        convergence_holder = []
        found_holder = []
        timer_holder = []

        if TRACE_NPY:
            rank_after_each_trace = np.zeros((REPEAT, TRACES), dtype=np.uint8)

        # Set up graph
        traces_to_use = TRACES
        if method == "SEQ" or method == "IND":
            traces_to_use = 1

        my_graph = fG.FactorGraphAES(no_print=NO_PRINT, traces=traces_to_use, removed_nodes=REMOVED_NODES, left_out_nodes=LEFT_OUT_NODES,
                                     key_scheduling=INCLUDE_KEY_SCHEDULING, furious=not USE_ARM_AES,
                                     rounds_of_aes=ROUNDS_OF_AES,
                                     remove_cycle=REMOVE_CYCLE, real_traces=REAL_TRACES,
                                     use_lda=USE_LDA, use_nn=USE_NN, use_best=USE_BEST, tprange=TPRANGE, jitter=JITTER)

        for rep in range(REPEAT):

            if PRINT_ALL_KEY_RANKS:
                print ("-_-_-_-_-_-_- Repeat {} -_-_-_-_-_-_-".format(rep))

            # Set Key and snr
            my_graph.set_key(key)
            my_graph.set_snr_exp(SNR_exp)

            # Container to hold distributions of key bytes
            key_distributions = np.array([get_no_knowledge_array() for i in range(16)])  # 16 Key Bytes
            key_distributions_sum = np.array([get_no_knowledge_array() for i in range(16)])
            if KEY_POWER_VALUE_AVERAGE:
                my_graph.compute_averaged_key_values(averaged_traces = TRACES, no_leak=NOT_LEAKING_NODES, fixed_value=fixed_node_tuple,
                    elmo_pow_model=ELMO_POWER_MODEL, real_traces=REAL_TRACES,
                    no_noise=NO_NOISE, offset=(rep*TRACES), ignore_bad=IGNORE_BAD_TEMPLATES)
                key_initial_distributions = np.array([get_zeros_array() for i in range(16)])  # 16 Key Bytes
                current_incoming_key_messages = np.array([get_no_knowledge_array() for i in range(16)])  # 16 Key Bytes

            # Bool to handle early break
            found_before_end = False

            # Failed Trace Counter
            failed_traces = 0

            max_rounds_taken = 0

            incoming_messages_list = [0] * 16
            for i in range(16):
                incoming_messages_list[i] = [0] * 256
                for j in range(256):
                    incoming_messages_list[i][j] = [0] * TRACES

            for trace in range(TRACES): # but only done once if LFG, as it breaks early

                # not_allowed_traces = [7, 8, 11, 14, 19, 24, 26, 28, 29, 30, 38, 39, 42, 44, 47, 54, 55, 57, 60, 64, 65,
                #                       66, 67, 68, 69, 74, 77, 81, 82, 88, 89, 97, 99, 101, 102, 103, 106, 112, 118,
                #                       120, 124, 126, 131, 132, 134, 137, 140, 146, 147, 150, 151, 153, 155, 160, 161,
                #                       164, 175, 177, 182, 185, 188, 190, 197, 198, 199]
                #
                # # if TEST2 and (trace not in allowed_traces):
                # #     continue
                # if TEST2 and (trace in not_allowed_traces):
                #     continue

                # Clear edges
                my_graph.initialise_edges()

                # Set distributions
                specific_trace = None
                if method == "SEQ" or method == "IND":
                    specific_trace = trace

                my_graph.set_all_initial_distributions( #specific_trace=specific_trace,
                                                       no_leak=NOT_LEAKING_NODES, fixed_value=fixed_node_tuple,
                                                       elmo_pow_model=ELMO_POWER_MODEL, real_traces=REAL_TRACES,
                                                       no_noise=NO_NOISE, offset=trace+(rep*TRACES), ignore_bad=IGNORE_BAD_TEMPLATES)

                if PRINT_ALL_KEY_RANKS:
                    print "-~-~-~-~-~-~- Trace {} -~-~-~-~-~-~-".format(trace)
                    # first_plaintext_bytes = all_values['p'][trace][:16].astype(int)
                    first_plaintext_bytes = my_graph.get_plaintext_values()
                    print "Plaintext: {}".format(first_plaintext_bytes)
                    print "Plaintext Hex String: {}".format(get_list_as_hex_string(first_plaintext_bytes))

                # Check
                if CHECK_LEAKAGE:
                    my_graph.check_leakage_details()
                    exit(1)

                # Handle SEQ
                if (method == "SEQ") and (trace > 0) and not KEY_POWER_VALUE_AVERAGE:
                    key_distributions = array_2d_multiply(key_distributions,
                                                          my_graph.get_all_key_initial_distributions())
                    # print key_distributions.shape, key_distributions
                    my_graph.set_key_distributions(key_distributions)
                elif (method == "SEQ") and (trace == 0) and KEY_POWER_VALUE_AVERAGE:
                    key_distributions = my_graph.get_all_key_initial_distributions()
                elif (method == "IND") and not KEY_POWER_VALUE_AVERAGE:
                    # Multiply incoming messages to key_distributions
                    key_distributions = array_2d_multiply(key_distributions,
                                                          my_graph.get_all_key_initial_distributions())
                    key_distributions_sum = array_2d_add(key_distributions_sum,
                                                         my_graph.get_all_key_initial_distributions())
                elif (method == "IND") and KEY_POWER_VALUE_AVERAGE and (trace == 0):
                    key_initial_distributions = my_graph.get_all_key_initial_distributions()

                # Start Timer
                start_time = datetime.now()

                # Run
                final_state, round_converged, round_found = my_graph.bp_run(ROUNDS,
                                                                            print_all_key_ranks=PRINT_ALL_KEY_RANKS,
                                                                            break_when_found=BREAK_WHEN_FOUND,
                                                                            print_all_marginal_distributions=PRINT_ALL_MARGINAL_DISTRIBUTIONS,
                                                                            print_possible_values=PRINT_POSSIBLE_VALUES,
                                                                            rank_order=PRINT_RANK_ORDER,
                                                                            break_when_information_exhausted_pattern=
                                                                            BREAK_WHEN_PATTERN_MATCHED, epsilon=EPSILON,
                                                                            epsilon_s=EPSILON_S,
                                                                            print_marginal_distance=PRINT_MARGINAL_DISTANCE,
                                                                            break_if_failed=BREAK_IF_FAILED,
                                                                            round_csv=ROUND_CSV, snrexp=SNR_exp,
                                                                            update_key_initial_distributions=UPDATE_KEY)

                # TODO
                # print "Ending!"
                # exit(1)

                # print "Has it failed? {}".format(my_graph.check_plaintext_failure())

                # End Timer
                timer_holder.append((datetime.now() - start_time).seconds)

                # Use Final State and Round Converged (etc)
                if final_state == "maxediterations":
                    total_maxed_iterations += 1
                elif final_state == "epsilonexhaust":
                    total_epsilon_exhaustion += 1

                # Number of rounds taken (averaged at end)
                max_rounds_taken = max(max_rounds_taken, round_converged)

                # TEST
                # ground_truths = my_graph.get_ground_truths()

                # print "+++ Repeat {} Trace {} Converged at Round {}".format(rep, trace, round_converged)
                convergence_holder.append(round_converged)
                # print "convergence_holder: {}".format(convergence_holder)


                ################# PRINT FINAL KEY RANKS FOR CURRENT TRACE #################
                if (method == "IND" or method == "SEQ") and PRINT_EVERY_TRACE and not NO_PRINT:
                    print print_fill_2, "Final Key Ranks (Repeat {} Trace {})".format(rep, trace), print_fill_2
                    print_new_line()
                    my_graph.print_key_rank()
                elif method == "LFG":
                    # Only need to do once, so break here
                    break

                # Check for failure
                if IGNORE_GROUND_TRUTHS or (
                        not my_graph.check_plaintext_failure(debug_mode=True) and
                        not my_graph.check_failure_on_specific_byte('t', debug_mode=False)):

                    # Update Key Distribution Depending
                    if method == "SEQ" and KEY_POWER_VALUE_AVERAGE:
                        key_distributions = array_2d_multiply(key_distributions, my_graph.get_all_key_incoming_messages())
                    elif method == "SEQ":
                        key_distributions = my_graph.get_marginal_distributions_of_key_bytes()
                    elif method == "IND" and not KEY_POWER_VALUE_AVERAGE:
                        # Multiply incoming messages to key_distributions
                        key_distributions = array_2d_multiply(key_distributions,
                                                              my_graph.get_all_key_incoming_messages())

                        key_distributions_sum = array_2d_add(key_distributions_sum,
                                                             my_graph.get_all_key_incoming_messages())

                        # Store key distributions
                        # incoming_messages_for_trace.append(my_graph.get_all_key_incoming_messages())
                        incoming_messages = my_graph.get_all_key_incoming_messages()
                        for i_ in range(16):
                            for j_ in range(256):
                                incoming_messages_list[i_][j_][trace] = incoming_messages[i_][j_]
                    elif method == "IND" and KEY_POWER_VALUE_AVERAGE:
                        # Multiply the incoming messages
                        current_incoming_key_messages = array_2d_multiply(current_incoming_key_messages,
                                                              my_graph.get_all_key_incoming_messages())
                        key_distributions = array_2d_multiply(key_initial_distributions, current_incoming_key_messages)

                    if BREAK_WHEN_FOUND and (method == "SEQ" or method == "IND") and my_graph.found_key(
                            supplied_dist=key_distributions):
                        if not NO_PRINT:
                            print "+++++++ Found Key at the end of Trace {} +++++++".format(trace + 1)
                        traces_required.append(trace + 1)
                        found_before_end = True
                        break
                else:
                    # Print discounted
                    if not ONLY_END and not NO_PRINT:
                        print "!+!+!+!+! Discounting Trace to avoid Error Propagation !+!+!+!+!"
                        print_new_line()
                    failed_traces += 1
                    total_failures += 1

                if not ONLY_END and PRINT_EVERY_TRACE and not NO_PRINT and (method == "IND" or method == "SEQ"):
                    print print_fill_2, "Current Computed Key Ranks (Repeat {}, after Trace {})".format(rep,
                                                                                                        trace), print_fill_2
                    print_new_line()
                    my_graph.print_key_rank(supplied_dist=key_distributions)

                if TRACE_NPY:
                    rank_after_each_trace[rep][trace] = bit_length(
                        my_graph.get_final_key_rank(supplied_dist=key_distributions))

                ################## End of Trace ##################

                # TODO
                # exit(1)

            ########## After all Traces done:

            # If found correct key, add it to found_holder
            if (((method == 'IND' or method == 'SEQ') and (my_graph.found_key(supplied_dist=key_distributions))) or (
                    (method == 'LFG') and my_graph.found_key())):
                found_holder.append(max_rounds_taken)
            else:
                found_holder.append(None)

            # Add Final Key Rank Only if not failed
            if failed_traces == TRACES:
                # All traces failed!
                if not ONLY_END and not NO_PRINT:
                    print "!!!+!+!!! All Traces Failed in this Repeat, Not Adding to Statistic !!!+!+!!!"
            else:
                if method == "IND" or method == "SEQ":
                    final_key_ranks.append(my_graph.get_final_key_rank(supplied_dist=key_distributions))
                    for i in range(len(key_distributions_sum)):
                        key_distributions_sum[i] = normalise_array(key_distributions_sum[i].astype(np.float32))
                    final_key_ranks_averaged.append(my_graph.get_final_key_rank(supplied_dist=key_distributions_sum))
                    if MARTIN_RANK:
                        final_key_rank_martin += my_graph.get_final_key_rank(martin=MARTIN_RANK,
                                                                             supplied_dist=key_distributions)
                else:
                    final_key_ranks.append(my_graph.get_final_key_rank())
                    # print "\n\n\n! TEST KEY RANK HERE (APPEND FINAL): {}\n\n\n".format(my_graph.get_key_rank(1)) #debug
                    if MARTIN_RANK:
                        final_key_rank_martin += my_graph.get_final_key_rank(martin=MARTIN_RANK, factor=1000000)

                # Add traces required
                if not found_before_end:
                    traces_required.append(TRACES)

            # Print Final Key Rank
            if PRINT_FINAL_KEY_RANK or REPEAT_ONLY:
                print print_fill_2, "Final Key Ranks (Repeat {})".format(rep), print_fill_2
                print_new_line()
                if method == "LFG":
                    my_graph.print_key_rank()
                else:
                    my_graph.print_key_rank(supplied_dist=key_distributions)

            if PRINT_FINAL_KEY_DISTRIBUTION:
                print print_fill_2, "Final Key Distribution (Repeat {})".format(rep), print_fill_2
                print key_distributions

            if REPEAT_CSV:
                # Write csv after each repeat
                csv_string = ""
                # TODO: JUST FIRST KEY BYTE
                csv_string += remove_brackets(key_distributions[0]) + '\n'
                # for i in range(len(key_distributions)):
                # csv_string += remove_brackets(key_distributions[i]) + '\n'

                furious_string = "Furious" if not USE_ARM_AES else ""
                one_round_string = "{}Rounds".format(ROUNDS_OF_AES)
                remove_cycle_string = "RemovedCycle" if REMOVE_CYCLE else ""

                f = open(
                    'output/keydist_AES{}{}{}_Traces{}_SNRexp{}_Repeat{}.csv'.format(furious_string, one_round_string,
                                                                                     remove_cycle_string, TRACES,
                                                                                     SNR_exp, rep), 'w+')
                f.write("{}\n".format(remove_brackets(csv_string)))
                f.close()

        # After all Repeats:

        # Convergence Statistics
        convergence_statistics = [i for i in convergence_holder if i is not None]
        not_converged = len([i for i in convergence_holder if i is None])
        successful_attacks = final_key_ranks.count(1)
        round_found_statistics = [i for i in found_holder if i is not None]

        # Final Key Ranks
        if not NO_PRINT:
            OUTOF = REPEAT if method == "LFG" else TRACES * REPEAT
            print "+++++++++ Key Rank Statistics +++++++++"
            print_statistics(final_key_ranks, log=True)

            if PRINT_AVERAGES:
                print "+++++++++ Key Rank Statistics (Averaging Incoming Messages) +++++++++"
                print_statistics(final_key_ranks_averaged, log=True)

            # Convergence Statistics
            print "+++++++ Stopping Criteria Statistics +++++++"
            print_statistics(convergence_statistics)
            print "Total Successes:  {:4}  ({:7}%)".format(successful_attacks,
                                                           (successful_attacks / (REPEAT + 0.0) * 100))
            print "Total Attacks:    {:4}".format(OUTOF)
            print "Failures:         {:4}  ({:7}%)".format(total_failures, (total_failures / (OUTOF + 0.0) * 100))
            print "Maxed Iterations: {:4}  ({:7}%)".format(total_maxed_iterations,
                                                           (total_maxed_iterations / (OUTOF + 0.0) * 100))
            print "Epsilon Exhaust:  {:4}  ({:7}%)".format(total_epsilon_exhaustion,
                                                           (total_epsilon_exhaustion / (OUTOF + 0.0) * 100))

            print "+++++++ Key Finding Statistics +++++++"
            # print round_found_statistics
            print_statistics(round_found_statistics)
            print_new_line()

            print "+++++++ Timing Statistics (per trace, in seconds) +++++++"
            print_statistics(timer_holder)
            print_new_line()

        if RANK_CSV:
            OUTOF = REPEAT if method == "LFG" else TRACES * REPEAT
            # Store as csv:
            best_case = bit_length(min(final_key_ranks))
            worst_case = bit_length(max(final_key_ranks))
            avg_case = bit_length(get_average(final_key_ranks))
            variance = bit_length(array_variance(final_key_ranks))
            failed_percentage = (total_failures / (OUTOF + 0.0) * 100)

            csv_string = "{}, {}, {}, {}, {}, {}, {}".format(SNR_exp, TRACES, best_case, worst_case, avg_case, variance,
                                                             failed_percentage)

            no_cycle_string = "NOCYCLE" if REMOVE_CYCLE else ""
            no_mc = "NOMC" if REMOVED_NODES else ""
            f = open('output/rank_dump{}{}.csv'.format(no_cycle_string, no_mc), 'a+')
            f.write("{}\n".format(csv_string))
            f.close()

        if FULL_ROUND_CSV:
            OUTOF = REPEAT if method == "LFG" else TRACES * REPEAT
            # Store as csv:
            best_case = bit_length(min(final_key_ranks))
            worst_case = bit_length(max(final_key_ranks))
            avg_case = bit_length(get_average(final_key_ranks))
            variance = bit_length(array_variance(final_key_ranks))
            failed_percentage = (total_failures / (OUTOF + 0.0) * 100)
            csv_string = "{}, {}, {}, {}, {}, {}, {}, {}, {}".format(SNR_exp, ROUNDS, successful_attacks,
                                                                     (successful_attacks / (REPEAT + 0.0) * 100),
                                                                     best_case, worst_case, avg_case, variance,
                                                                     failed_percentage)

            first_round_string = "{}ROUNDS".format(ROUNDS_OF_AES)
            no_cycle_string = "NOCYCLE" if REMOVE_CYCLE else ""
            no_mc = "NOMC" if REMOVED_NODES else ""
            f = open('output/FullRoundCsv{}{}{}.csv'.format(first_round_string, no_cycle_string, no_mc), 'a+')
            f.write("{}\n".format(csv_string))
            f.close()

        if CONVERGENCE_CSV:
            OUTOF = REPEAT if method == "LFG" else TRACES * REPEAT
            # Store as csv:
            if len(convergence_statistics) == 0:
                best_case = -1
                worst_case = -1
                avg_case = -1
                variance = 0
            else:
                best_case = (min(convergence_statistics))
                worst_case = (max(convergence_statistics))
                avg_case = (get_average(convergence_statistics))
                variance = (array_variance(convergence_statistics))

            failed_percentage = (not_converged / (OUTOF + 0.0) * 100)

            # first_round_string = "True" if FIRST_ROUND else "False"
            round_string = str(ROUNDS_OF_AES)
            no_cycle_string = "True" if REMOVE_CYCLE else "False"
            no_mc = "True" if REMOVED_NODES else "False"

            csv_string = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(method, TRACES, SNR_exp,
                                                                                         round_string,
                                                                                         no_cycle_string, no_mc,
                                                                                         best_case, worst_case,
                                                                                         avg_case, variance,
                                                                                         failed_percentage,
                                                                                         successful_attacks, REPEAT, (
                                                                                                 successful_attacks / (
                                                                                                 REPEAT + 0.0) * 100))

            f = open('output/ConvergenceCsv.csv'.format(no_cycle_string, no_mc), 'a+')
            f.write("{}\n".format(csv_string))
            f.close()

        if ALL_CSV:

            OUTOF = REPEAT if method == "LFG" else TRACES * REPEAT

            round_string = str(ROUNDS_OF_AES)
            no_cycle_string = "True" if REMOVE_CYCLE else "False"
            no_mc = "True" if REMOVED_NODES else "False"

            best_case = bit_length(min(final_key_ranks))
            worst_case = bit_length(max(final_key_ranks))
            median_case = array_median(np.array(get_log_list(final_key_ranks)))

            if len(round_found_statistics) == 0:
                best_found = None
                worst_found = None
                average_found = None
            else:
                best_found = min(round_found_statistics)
                worst_found = max(round_found_statistics)
                average_found = get_average(round_found_statistics)

            timer_holder_min = min(timer_holder)
            timer_holder_max = max(timer_holder)
            timer_holder_avg = get_average(timer_holder)

            # csv_string = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
            #     TEST_NAME, TRACES, REPEAT, ROUNDS, method,
            #     SNR_exp, round_string, no_cycle_string, no_mc,
            #     OUTOF, successful_attacks, ((successful_attacks / (REPEAT + 0.0)) * 100), best_case,
            #     worst_case, median_case, total_epsilon_exhaustion, total_maxed_iterations,
            #     total_failures, best_found, worst_found, average_found,
            #     timer_holder_min,timer_holder_max,timer_holder_avg)

            csv_string = "{};".format(TEST_NAME)
            for parakey, paravalue in args.__dict__.iteritems():
                csv_string += str(paravalue) + ";"
            for value_to_store in [successful_attacks, ((successful_attacks / (REPEAT + 0.0)) * 100),
                best_case, worst_case, median_case, total_epsilon_exhaustion, total_maxed_iterations,
                total_failures, best_found, worst_found, average_found,
                timer_holder_min,timer_holder_max,timer_holder_avg]:
                csv_string += str(value_to_store) + ";"

            dump_file = 'output/results_dump.csv'

            if not check_file_exists(dump_file):
                header = "TestName;"
                for parakey, paravalue in args.__dict__.iteritems():
                    header += str(parakey) + ";"
                header += "SuccessfulAttacks;PercentageSuccess;BestCase;WorstCase;MedianCase;TotalEpsilonExhaustion;TotalTMax;TotalFailures;BestRoundFound;WorstRoundFound;AverageRoundFound;MinTraceTime;MaxTraceTime;AvgTraceTime;"
                f = open(dump_file, 'a+')
                f.write("{}\n".format(header))
                f.close()

            f = open(dump_file, 'a+')
            f.write("{}\n".format(csv_string))
            f.close()

        # Martin Rank
        if MARTIN_RANK:
            avg_key_rank_martin = final_key_rank_martin / REPEAT
            if not NO_PRINT:
                print "[[[ Martin Average Key Rank: {} (~2^{}) ]]]".format(avg_key_rank_martin,
                                                                           bit_length(avg_key_rank_martin))

        # Average Traces Needed
        if (method == "SEQ" or method == "IND") and (BREAK_WHEN_FOUND or BREAK_WHEN_PATTERN_MATCHED):
            if not NO_PRINT:
                print "+++++++++ Trace Statistics +++++++++"
                print_statistics(traces_required)
            if TRACE_CSV:
                # Store as csv:
                f = open('output/traces_r{}_rep{}_snrexp{}.csv'.format(ROUNDS, REPEAT, SNR_exp), 'w+')
                f.write("{}\n".format(remove_brackets(str(traces_required))))
                f.close()

        # Dump the Result
        if DUMP_RESULT:
            dump_result(test=TEST_NAME, connecting_method=connecting_dict[method], traces=TRACES, rounds=ROUNDS,
                        repeats=REPEAT, snrexp=SNR_exp, noleak=NOT_LEAKING_NODES, badly=BADLY_LEAKING_NODES,
                        removed=REMOVED_NODES, threshold=THRESHOLD, epsilon=EPSILON, epsilon_s=EPSILON_S,
                        rank=key_rank_string, traces_needed=avg_traces_required)

        if SAVE_FIRST_DISTS:
            k_node = "k001-K"
            p_node = "p001-0"
            # k_node_dist = my_graph.get_marginal_distribution(k_node)
            k_node_dist = my_graph.get_all_incoming_messages(k_node)
            p_node_dist = my_graph.get_all_incoming_messages(p_node)

            # print "{}:\n\n{}\n{}:\n\n{}\n".format(k_node, k_node_dist, p_node, p_node_dist)
            # print  "Check: k {} p {}".format(sum(k_node_dist), sum(p_node_dist))

            # np.save(SFD_NPY_K_FILE, k_node_dist)
            # np.save(SFD_NPY_P_FILE, p_node_dist)

            Pickle.dump(k_node_dist, open(SFD_NPY_K_FILE, "wb"))
            Pickle.dump(p_node_dist, open(SFD_NPY_P_FILE, "wb"))

            # Also plot, why not
            plot_two_distributions(k_node_dist, p_node_dist)

        if TRACE_NPY:
            g_string = "G{}".format(ROUNDS_OF_AES)
            if REMOVE_CYCLE:
                if 'mc' in REMOVED_NODES:
                    g_string += "Aprime"
                else:
                    g_string += "A"
            connection_string = "{}_".format(method)
            lda_string = "LDA{}_".format(TPRANGE) if USE_LDA else ""
            nn_string = "NN{}_".format(TPRANGE) if USE_NN else ""
            best_string = "BEST_" if USE_BEST else ""
            ignore_string = "IGNOREBAD_" if IGNORE_BAD_TEMPLATES else ""
            ks_string = "KS_" if INCLUDE_KEY_SCHEDULING else ""
            # np.save("{}_{}_{}.npy".format(OUTPUT_FILE_PREFIX, g_string, TRACES), rank_after_each_trace)
            snr_string = SNR_exp if not REAL_TRACES else 'REAL{}'.format(CORRELATION_THRESHOLD)
            kavg_string = "KEYAVG_" if KEY_POWER_VALUE_AVERAGE else ""
            Pickle.dump(rank_after_each_trace,
                        open("{}_{}{}{}{}{}{}_{}{}{}T_{}I_{}.npy".format(OUTPUT_FILE_PREFIX, connection_string, lda_string, nn_string, best_string, ignore_string, g_string, kavg_string,
                                                            ks_string, TRACES, ROUNDS, snr_string), "wb"))
            if PLOT:

                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                x = np.arange(1, TRACES + 1)

                for repeat_number, repeat_rank in enumerate(rank_after_each_trace):
                    ax1.plot(x, repeat_rank, label='Repeat {}'.format(repeat_number))

                plt.xlabel('Number of Traces')
                plt.ylabel('Final Key Rank ' + r'$(\log_2)$')
                plt.title(
                    'Graph {} Plot, {} Traces, SNR '.format(graph_string, TRACES) + r'$2^{%d}$' % (int(float(SNR_exp))))
                legend = ax1.legend(loc='best')
                plt.show()
                print "Finished Plotting!"

        # If Marginal Distance, return
        if margdist is not None:
            if method == "LFG":
                key_distributions = my_graph.get_marginal_distributions_of_key_bytes()
            return key_distributions

    # Finally, return key_distributions
    if len(connecting_methods) > 0:
        return key_distributions

    # ===============================================================================


# except KeyboardInterrupt:
#     print "\n\n+=+= Interrupted by User =+=+\n"

if __name__ == "__main__":

    # try:

    ################################################################################

    PRINT_ALL_KEY_RANKS = True
    PRINT_ALL_MARGINAL_DISTRIBUTIONS = False
    PRINT_FINAL_KEY_RANK = False
    PRINT_POSSIBLE_VALUES = False
    PRINT_RANK_ORDER = False
    PRINT_MARGINAL_DISTANCE = False
    PRINT_AVERAGES = False
    BREAK_WHEN_FOUND = False

    ############################# Argument Parsing ######################################

    parser = argparse.ArgumentParser(description='Run the Belief Propagation Algorithm.')

    parser.add_argument('-k', action="store", dest="MY_KEY", help='Chosen Key as Hex String (default: FIXED inside)',
                        default=None)
    parser.add_argument('-t', action="store", dest="TRACES", help='Number of Traces (default: 1)', type=int, default=1)
    parser.add_argument('-r', action="store", dest="ROUNDS", help='Number of Iterations in BP (default: 5)', type=int,
                        default=5)
    parser.add_argument('-snrexp', action="store", dest="SNR_exp",
                        help='snr Exponent, s.t. snr = 2**SNR_exp (default: 5)', type=float, default=5)
    parser.add_argument('-rep', action="store", dest="REPEAT", help='Number of Repetitions to Average (default: 1)',
                        type=int, default=1)
    parser.add_argument('-thresh', action="store", dest="THRESHOLD",
                        help='Threshold for refusing bad leakage (default: None)', type=float, default=None)
    parser.add_argument('-epsilon', action="store", dest="EPSILON",
                        help='Threshold for breaking early (default: 0.0001)', type=float, default=0.0001)
    parser.add_argument('-epsilon_s', action="store", dest="EPSILON_S",
                        help='How many successive epsilon round to break early (default: 10)', type=int, default=10)
    parser.add_argument('-seed', action="store", dest="SEED", help='Seed for extra randomisation (default: 0)',
                        type=int, default=0)
    parser.add_argument('-tp', '-trace_range', action="store", dest="TPRANGE",
                        help='Window of Power Values over Timepoint (default: 1)',
                        type=int, default=1)
    parser.add_argument('-j', '-jitter', action="store", dest="JITTER",
                        help='Clock Jitter to use on real traces (default: None)',
                        type=int, default=None)

    parser.add_argument('-ct', '-cthresh', action="store", dest="CORRELATION_THRESHOLD",
                        help='Threshold for refusing bad point of interest detected nodes (default: None)', type=float, default=None)

    parser.add_argument('-raes', '-rounds_of_aes', action="store", dest="ROUNDS_OF_AES",
                        help='Number of Rounds of AES (default: 10)', type=int, default=10)

    parser.add_argument('--LFG', action="store_true", dest="LARGE_FACTOR_GRAPH",
                        help='Toggles Large Factor Graph On (default: False)', default=False)
    parser.add_argument('--SQG', '--SEQ', '--SFG', action="store_true", dest="SEQUENTIAL_GRAPHS",
                        help='Toggles Sequential Graph On (default: False)', default=False)
    parser.add_argument('--ING', '--IND', '--IFG', action="store_false", dest="INDEPENDENT_GRAPHS",
                        help='Toggles Independent Graph Off (default: True)', default=True)
    parser.add_argument('--KEYAVG', '--KAVG', '--KPAVG', action="store_false", dest="KEY_POWER_VALUE_AVERAGE",
                        help='Toggles Key Power Value Averaging Off (default: True)', default=True)
    parser.add_argument('--KS', action="store_true", dest="KEY_SCHEDULING",
                        help='Toggles Key Scheduling On (default: False)', default=False)
    parser.add_argument('--ELMO', action="store_true", dest="ELMO_POWER_MODEL",
                        help='Toggles ELMO Power Model On (default: True)', default=False)
    parser.add_argument('--HW', action="store_true", dest="HW_POWER_MODEL",
                        help='Toggles HW Power Model On (default: False)', default=False)

    parser.add_argument('--BPAT', action="store_true", dest="BREAK_WHEN_PATTERN_MATCHED",
                        help='Break when patterns matched (default: False)', default=False)
    parser.add_argument('--BFND', action="store_true", dest="BREAK_WHEN_FOUND",
                        help='Break when correct value found (default: False)', default=False)
    parser.add_argument('--BERR', '--BIF', action="store_true", dest="BREAK_IF_FAILED",
                        help='Break if failed (check through Plaintext) (default: False)', default=False)

    parser.add_argument('--IGT', '--IGNORE_GROUND_TRUTHS', action="store_false", dest="IGNORE_GROUND_TRUTHS",
                        help='Toggles Ignore Ground Truths (default: True)', default=True)

    parser.add_argument('--IGB', '--IGNORE_BAD_TEMPLATES', action="store_true", dest="IGNORE_BAD_TEMPLATES",
                        help='Toggles Ignore Bad Templates (default: False)', default=False)

    parser.add_argument('--LEAKAGE_ON_THE_FLY', '--LOTF', action="store_false", dest="LEAKAGE_ON_THE_FLY",
                        help='Toggles Leakage on the Fly off (default: True)', default=True)

    parser.add_argument('--LOCAL_LEAKAGE', action="store_false", dest="LOCAL_LEAKAGE",
                        help='Toggles Local Leakage Off, if cannot compute on the fly (default: True)', default=True)

    parser.add_argument('--READ_PLAINTEXTS', action="store_true", dest="READ_PLAINTEXTS",
                        help='Reads Plaintexts from File (default: False)', default=False)

    parser.add_argument('--CONVERGENCE_TEST', action="store_true", dest="CONVERGENCE_TEST",
                        help='Prints out Convergence Statistics (default: False)', default=False)
    parser.add_argument('--CURRENT_TEST', action="store_true", dest="CURRENT_TEST",
                        help='Alters values to match current test statistic (default: False)', default=False)
    parser.add_argument('--TRACE_NPY', action="store_true", dest="TRACE_NPY",
                        help='Store Trace Values as npy (default: False)', default=False)
    parser.add_argument('--PLOT', action="store_true", dest="PLOT",
                        help='Plot Final Key Ranks (default: False)', default=False)
    parser.add_argument('--DUMP_RESULT', '--DATA_DUMP', action="store_true", dest="DUMP_RESULT",
                        help='Dumps Result in output/data_dump.txt (default: False)', default=False)
    parser.add_argument('--TRACE_CSV', action="store_true", dest="TRACE_CSV",
                        help='Store Trace Values as csv (default: False)', default=False)
    parser.add_argument('--WRITE_CSV', '--CSV', action="store_true", dest="WRITE_CSV",
                        help='Writes key distirbution to csv file (default: False)', default=False)
    parser.add_argument('--WRITE_NPY', '--NPY', action="store_true", dest="WRITE_NPY",
                        help='Writes key distirbution to npy file (default: False)', default=False)
    parser.add_argument('--RANK_CSV', action="store_true", dest="RANK_CSV",
                        help='Writes key rank data to csv file (default: False)', default=False)
    parser.add_argument('--ROUND_CSV', action="store_true", dest="ROUND_CSV",
                        help='Writes key rank round data to csv file (default: False)', default=False)
    parser.add_argument('--FULL_ROUND_CSV', action="store_true", dest="FULL_ROUND_CSV",
                        help='Writes full key rank round data to csv file (default: False)', default=False)
    parser.add_argument('--REPEAT_CSV', action="store_true", dest="REPEAT_CSV",
                        help='Writes key distribution data to csv file after each repeat (default: False)',
                        default=False)
    parser.add_argument('--CONVERGENCE_CSV', action="store_true", dest="CONVERGENCE_CSV",
                        help='Writes convergence data to csv file after each trace and repeat (default: False)',
                        default=False)
    parser.add_argument('--ALL_CSV', action="store_true", dest="ALL_CSV",
                        help='Writes all data to csv file after each repeat (default: False)', default=False)
    parser.add_argument('--FKD', action="store_true", dest="PRINT_FINAL_KEY_DISTRIBUTION",
                        help='Prints Final Key Distribution (default: False)', default=False)

    parser.add_argument('--TEST_NAME', action="store", dest="TEST_NAME",
                        help='Name for Test (if testing) (default: "Standard")', type=str, default="Standard")

    parser.add_argument('--ONLY_FINAL', '--FINAL_ONLY', action="store_true", dest="ONLY_FINAL",
                        help='Only prints out Final Rank (default: False)', default=False)
    parser.add_argument('--ONLY_RESULT', '--RESULT_ONLY', '--RESULTS_ONLY', action="store_true", dest="ONLY_RESULT",
                        help='Only prints out Result (default: False)', default=False)
    parser.add_argument('--ONLY_END', '--END_ONLY', action="store_true", dest="ONLY_END",
                        help='Only prints out End Result (default: False)', default=False)
    parser.add_argument('--ONLY_REP', '--REPEAT_ONLY', action="store_true", dest="REPEAT_ONLY",
                        help='Only prints out End of each Repeat Result (default: False)', default=False)
    parser.add_argument('--EVERY_TRACE', action="store_true", dest="PRINT_EVERY_TRACE",
                        help='Prints out Key Rank after each trace (default: False)', default=False)
    parser.add_argument('--NO_PRINT', action="store_true", dest="NO_PRINT",
                        help='Only prints out Average Rank (default: False)', default=False)
    parser.add_argument('--NO_NOISE', action="store_true", dest="NO_NOISE",
                        help='No Noise in Simulation (default: False)', default=False)
    parser.add_argument('--PRINT_DICT', action="store_true", dest="PRINT_DICTIONARY",
                        help='Prints Dictionary of Values from Leakage Simulation (default: False)', default=False)
    parser.add_argument('--UPDATE_KEY', '--UKID', action="store_true", dest="UPDATE_KEY",
                        help='Updates Key Initial Distributions (default: False)', default=False)
    parser.add_argument('--UNPROFILED', '--NEW', action="store_false", dest="UNPROFILED",
                        help='If Real Traces, only attack unprofiled traces (default: True)', default=True)

    # parser.add_argument('--ONE_ROUND', '--FIRST_ROUND', action="store_true", dest="FIRST_ROUND",
    #                     help='Only models first round of AES (default: False)', default=False)
    # parser.add_argument('--TWO_ROUNDS', '--SECOND_ROUND', action="store_true", dest="SECOND_ROUND",
    #                     help='Only models first two rounds of AES (default: False)', default=False)

    parser.add_argument('--REAL_TRACES', '--REAL', action="store_true", dest="REAL_TRACES",
                        help='Attacks a Real Trace (default: False)', default=False)

    parser.add_argument('--REMOVE_CYCLE', '--RM_C', '--ACYCLIC', action="store_true", dest="REMOVE_CYCLE",
                        help='Removes cycle in MixColumns step (default: False)', default=False)

    parser.add_argument('--MARTIN', '--MARTIN_RANK', action="store_true", dest="MARTIN_RANK",
                        help='Adds Martin Rank to final Rank (default: False)', default=False)
    parser.add_argument('--ARM', '--ARM_AES', action="store_true", dest="USE_ARM_AES",
                        help='Use ARM AES Implementation instead of AES Furious (default: False)', default=False)

    parser.add_argument('--RANDOM_REAL', action="store_false", dest="RANDOM_REAL",
                        help='Uses Random Trace Subsets for Real Trace Experiments (default: True)', default=True)

    parser.add_argument('--TEST_KEY', '--NEW_KEY', action="store_true", dest="TEST_KEY",
                        help='Uses Different Key (Test Purposes Only) (default: False)', default=False)
    parser.add_argument('--RANDOM_KEY', action="store_true", dest="RANDOM_KEY",
                        help='Uses Random Key (Test Purposes Only) (default: False)', default=False)
    parser.add_argument('--USE_LDA', '--LDA', action="store_true", dest="USE_LDA",
                        help='Uses LDA for Real Traces (default: False)', default=False)
    parser.add_argument('--USE_NN', '--NN', action="store_true", dest="USE_NN",
                        help='Uses Neural Network for Real Traces (default: False)', default=False)
    parser.add_argument('--USE_BEST', '--B', '--BEST', action="store_true", dest="USE_BEST",
                        help='Uses Best Template for Real Traces, out of Univariate, LDA, and NN (default: False)', default=False)

    parser.add_argument('--SAVE_FIRST_DISTS', '--SFD', action="store_true", dest="SAVE_FIRST_DISTS",
                        help='Saves the Distributions of the first Key and Plaintexts bytes (Test Purposes Only) (default: False)',
                        default=False)

    parser.add_argument('-rm', action="append", dest="REMOVED_NODES", help="Removes target variable (default: [])",
                        default=[])
    parser.add_argument('-lo', action="append", dest="LEFT_OUT_NODES",
                        help="For Distance between nodes, leave out node (default: [])", default=[])
    parser.add_argument('-nl', action="append", dest="NOT_LEAKING_NODES",
                        help="Doesn't leak on target variable (default: [])", default=[])
    parser.add_argument('-nn', action="append", dest="NO_NOISE_NODES", help="No noise on target variable (default: [])",
                        default=[])
    parser.add_argument('-bl', action="append", dest="BADLY_LEAKING_NODES",
                        help="Badly leaks on target variable (default: [])", default=[])
    parser.add_argument('-blsnrexp', action="store", dest="BADLY_LEAKING_SNR_exp",
                        help='snr Exponent for the bad leakage, s.t. snr = 2**SNR_exp (default: -7)', type=float,
                        default=-7)
    parser.add_argument('-blt', action="append", dest="BADLY_LEAKING_TRACES",
                        help="Badly leaks on target traces (default: [])", default=[])
    parser.add_argument('-fix', action="store", dest="FIXED_VALUE_NODE",
                        help='Fix Variable node to get Marginal Distance to Key Bytes (default: None)', default=None)

    parser.add_argument('--CL', '--CHECK_LEAKAGE', action="store_true", dest="CHECK_LEAKAGE",
                        help='Checks Initial Leakage (default: False)', default=False)
    parser.add_argument('--NLKS', '--NO_LEAK_KEY_SCHEDULE', action="store_true", dest="NO_LEAK_KEY_SCHEDULE",
                        help="Doesn't leak on Key Schedule (default: False)", default=False)

    args = parser.parse_args()
    # print args
    # exit(1)

    MY_KEY = args.MY_KEY
    TRACES = args.TRACES
    ROUNDS = args.ROUNDS
    SNR_exp = args.SNR_exp
    BADLY_LEAKING_SNR_exp = args.BADLY_LEAKING_SNR_exp
    REPEAT = args.REPEAT
    THRESHOLD = args.THRESHOLD
    EPSILON = args.EPSILON
    EPSILON_S = args.EPSILON_S
    SEED = args.SEED
    LARGE_FACTOR_GRAPH = args.LARGE_FACTOR_GRAPH
    SEQUENTIAL_GRAPHS = args.SEQUENTIAL_GRAPHS
    INDEPENDENT_GRAPHS = args.INDEPENDENT_GRAPHS
    INCLUDE_KEY_SCHEDULING = args.KEY_SCHEDULING
    LEFT_OUT_NODES = args.LEFT_OUT_NODES
    REMOVED_NODES = args.REMOVED_NODES
    NOT_LEAKING_NODES = args.NOT_LEAKING_NODES
    BADLY_LEAKING_NODES = args.BADLY_LEAKING_NODES
    BADLY_LEAKING_TRACES = args.BADLY_LEAKING_TRACES
    NO_NOISE_NODES = args.NO_NOISE_NODES
    BREAK_WHEN_PATTERN_MATCHED = args.BREAK_WHEN_PATTERN_MATCHED
    BREAK_WHEN_FOUND = args.BREAK_WHEN_FOUND
    BREAK_IF_FAILED = args.BREAK_IF_FAILED
    LOCAL_LEAKAGE = args.LOCAL_LEAKAGE
    ELMO_POWER_MODEL = args.ELMO_POWER_MODEL
    TEST_NAME = args.TEST_NAME
    DUMP_RESULT = args.DUMP_RESULT
    MARTIN_RANK = args.MARTIN_RANK
    LEAKAGE_ON_THE_FLY = args.LEAKAGE_ON_THE_FLY
    # FIXED_VALUE_NODES       = dict(args.FIXED_VALUE_NODES)
    FIXED_VALUE_NODE = args.FIXED_VALUE_NODE
    USE_ARM_AES = args.USE_ARM_AES
    READ_PLAINTEXTS = args.READ_PLAINTEXTS
    TRACE_CSV = args.TRACE_CSV
    # FIRST_ROUND = args.FIRST_ROUND
    # SECOND_ROUND = args.SECOND_ROUND
    CHECK_LEAKAGE = args.CHECK_LEAKAGE
    ROUNDS_OF_AES = args.ROUNDS_OF_AES
    REMOVE_CYCLE = args.REMOVE_CYCLE
    WRITE_CSV = args.WRITE_CSV
    WRITE_NPY = args.WRITE_NPY
    RANK_CSV = args.RANK_CSV
    ROUND_CSV = args.ROUND_CSV
    FULL_ROUND_CSV = args.FULL_ROUND_CSV
    NO_PRINT = args.NO_PRINT
    NO_NOISE = args.NO_NOISE
    REPEAT_ONLY = args.REPEAT_ONLY
    PRINT_DICTIONARY = args.PRINT_DICTIONARY
    REPEAT_CSV = args.REPEAT_CSV
    CONVERGENCE_CSV = args.CONVERGENCE_CSV
    ALL_CSV = args.ALL_CSV
    PRINT_EVERY_TRACE = args.PRINT_EVERY_TRACE
    TEST_KEY = args.TEST_KEY
    RANDOM_KEY = args.RANDOM_KEY
    REAL_TRACES = args.REAL_TRACES
    TRACE_NPY = args.TRACE_NPY
    IGNORE_GROUND_TRUTHS = args.IGNORE_GROUND_TRUTHS
    SAVE_FIRST_DISTS = args.SAVE_FIRST_DISTS
    UPDATE_KEY = args.UPDATE_KEY
    ONLY_END = args.ONLY_END
    HW_POWER_MODEL = args.HW_POWER_MODEL
    USE_LDA = args.USE_LDA
    USE_NN = args.USE_NN
    TPRANGE = args.TPRANGE
    UNPROFILED = args.UNPROFILED
    PRINT_FINAL_KEY_DISTRIBUTION = args.PRINT_FINAL_KEY_DISTRIBUTION
    RANDOM_REAL = args.RANDOM_REAL
    PLOT = args.PLOT
    NO_LEAK_KEY_SCHEDULE = args.NO_LEAK_KEY_SCHEDULE
    CORRELATION_THRESHOLD = args.CORRELATION_THRESHOLD
    IGNORE_BAD_TEMPLATES = args.IGNORE_BAD_TEMPLATES
    USE_BEST = args.USE_BEST
    KEY_POWER_VALUE_AVERAGE = args.KEY_POWER_VALUE_AVERAGE
    JITTER = args.JITTER

    if MY_KEY is not None:
        CHOSEN_KEY = hex_string_to_int_array(MY_KEY)

    BADLY_LEAKING_TRACES = [int(i) for i in BADLY_LEAKING_TRACES]

    ################################################################################

    if args.CONVERGENCE_TEST:
        PRINT_ALL_KEY_RANKS = False
        PRINT_FINAL_KEY_RANK = True
        PRINT_RANK_ORDER = True
        BREAK_WHEN_FOUND = False
    elif args.ONLY_END or args.REPEAT_ONLY:
        PRINT_ALL_KEY_RANKS = False
        PRINT_FINAL_KEY_RANK = False
        ONLY_END = True
    elif args.ONLY_FINAL:
        PRINT_ALL_KEY_RANKS = False
        PRINT_FINAL_KEY_RANK = True
    elif args.ONLY_RESULT:
        PRINT_ALL_KEY_RANKS = False
        PRINT_FINAL_KEY_RANK = True
        PRINT_EVERY_TRACE = False
    elif NO_PRINT:
        PRINT_ALL_KEY_RANKS = False
    elif args.CURRENT_TEST:
        # Stick everything in here
        # BREAK_WHEN_FOUND                  = False
        # PRINT_MARGINAL_DISTANCE           = True
        PRINT_ALL_MARGINAL_DISTRIBUTIONS = True
        # BREAK_WHEN_FOUND        = True
        # PRINT_FINAL_KEY_RANK    = True
        # PRINT_ALL_KEY_RANKS     = False

    if USE_LDA and not REAL_TRACES:
        # if not NO_PRINT:
        print "! Can't currently use LDA on simulated data."
        raise ValueError

    if USE_NN and not REAL_TRACES:
        # if not NO_PRINT:
        print "! Can't currently use Neural Networks on simulated data."
        raise ValueError

    if USE_BEST and not REAL_TRACES:
        # if not NO_PRINT:
        print "! Can't currently use Best Templates on simulated data."
        raise ValueError

    if (USE_NN and USE_LDA) or (USE_NN and USE_BEST) or (USE_LDA and USE_BEST):
        if not NO_PRINT:
            print "!! Can only use one (NN, LDA, or BEST)! Please specify which. Aborting..."
        exit(1)

    if TPRANGE > 1 and (not USE_LDA and not USE_NN):
        if not NO_PRINT:
            print "|| If not using LDA or NN, range must be 1 - setting TPRANGE to 1"
        TPRANGE = 1

    if USE_NN and TPRANGE != 700:
        if not NO_PRINT:
            print "|| Neural Networks only uses window size 700 - setting TPRANGE to 700"
        TPRANGE = 700

    if USE_LDA and TPRANGE == 1:
        if not NO_PRINT:
            print "|| LDA doesn't like being set to 1 - setting TPRANGE to 200"
        TPRANGE = 200

    if REAL_TRACES:
        if not NO_PRINT:
            print "|| Real Traces modelled around snr = 2**-7 - setting SNR_exp to -7"
        SNR_exp = -7

    # Handle ELMO Default
    if not HW_POWER_MODEL and not ELMO_POWER_MODEL:
        if not NO_PRINT:
            print "|| Default Power Model: ELMO"
        ELMO_POWER_MODEL = True

    if REAL_TRACES and not TRACE_NPY:
        if not NO_PRINT:
            print "|| Real Traces usually need to be stored - setting TRACE_NPY to True"
        TRACE_NPY = True

    if PLOT and not TRACE_NPY:
        if not NO_PRINT:
            print "|| Need to store traces to plot them - setting TRACE_NPY to True"
        TRACE_NPY = True

    if REAL_TRACES and ROUNDS_OF_AES > r_of_aes:
        if not NO_PRINT:
            print "|| Selected Trace File can only handle up to {} AES Rounds - setting ROUNDS_OF_AES to {}".format(r_of_aes, r_of_aes)
        ROUNDS_OF_AES = r_of_aes

    # Handle optional extra rounds for first round tree
    G0_ROUNDS = 2
    if ROUNDS_OF_AES == 0 and (ROUNDS > G0_ROUNDS):
        if not NO_PRINT:
            print "|| Don't need {} Rounds for tree graph - setting to {}".format(ROUNDS, G0_ROUNDS)
        ROUNDS = G0_ROUNDS
    elif not LARGE_FACTOR_GRAPH and REMOVE_CYCLE and ROUNDS_OF_AES <= 2 and (ROUNDS > 16):
        new_rounds = ROUNDS_OF_AES * 8
        if not NO_PRINT:
            print "|| Don't need {} Rounds for tree graph - setting to {} ({} Rounds of AES * 8)".format(ROUNDS,
                                                                                                         new_rounds,
                                                                                                         ROUNDS_OF_AES)
        ROUNDS = new_rounds

    if 'p' in NOT_LEAKING_NODES and not IGNORE_GROUND_TRUTHS:
        if not NO_PRINT:
            print "|| Can't use Ground Truths if not leaking on plaintext bytes: setting IGNORE_GROUND_TRUTHS to True"
        IGNORE_GROUND_TRUTHS = True

    if NO_LEAK_KEY_SCHEDULE and not INCLUDE_KEY_SCHEDULING:
        if not NO_PRINT:
            print "|| Must include key schedule to not leak on it: setting INCLUDE_KEY_SCHEDULING to True"
        INCLUDE_KEY_SCHEDULING

    if NO_LEAK_KEY_SCHEDULE:
        NOT_LEAKING_NODES += ['k{}-0'.format(pad_string_zeros(count)) for count in range(17, 33)]
        NOT_LEAKING_NODES += ['sk{}-0'.format(pad_string_zeros(count)) for count in range(1, 5)]
        NOT_LEAKING_NODES += ['xk001-0']

    # if FIRST_ROUND and SECOND_ROUND:
    #     if not NO_PRINT:
    #         print "|| Can't have both first round and second round flags - opting to second round"
    #     FIRST_ROUND = False

    # Run BPA!
    if FIXED_VALUE_NODE is None:

        my_key_distributions = run_belief_propagation_attack()

        # exit(1)

        if WRITE_CSV:
            csv_string = ""
            for i in range(len(my_key_distributions)):
                csv_string += remove_brackets(my_key_distributions[i]) + '\n'
            my_furious_string = "Furious" if not USE_ARM_AES else ""
            my_round_string = "{}Rounds".format(ROUNDS_OF_AES)
            my_remove_cycle_string = "RemovedCycle" if REMOVE_CYCLE else ""
            f = open(
                'output/keydist_AES{}{}{}_Traces{}_SNRexp{}_Seed{}.csv'.format(my_furious_string, my_round_string,
                                                                               my_remove_cycle_string, TRACES,
                                                                               SNR_exp, SEED), 'w+')
            f.write("{}\n".format(remove_brackets(csv_string)))
            f.close()
        if WRITE_NPY:
            round_string = str(ROUNDS_OF_AES)
            remove_cycle_string = "A" if REMOVE_CYCLE else ""
            Pickle.dump(my_key_distributions, open(
                'output/keydist_G{}{}_Traces{}_SNRexp{}_Seed{}.npy'.format(round_string, remove_cycle_string,
                                                                           TRACES, SNR_exp, SEED), "wb"))
    else:
        # Store as numpy file, not csv!
        numpyfile_alldists = np.zeros((16, 256, 256))
        # For all Values, Fix each one
        for fix_val in range(256):
            my_key_distributions = run_belief_propagation_attack(margdist=(FIXED_VALUE_NODE, fix_val))
            numpyfile_alldists[:, fix_val, :] = my_key_distributions
        # Save to Numpy File
        # np.save('output/marginaldist_{}_{}_{}.npy'.format(FIXED_VALUE_NODE, SNR_exp, SEED), numpyfile_alldists)
        Pickle.dump(numpyfile_alldists,
                    open('output/marginaldist_{}_{}_{}.npy'.format(FIXED_VALUE_NODE, SNR_exp, SEED), "wb"))

    ########################## CHECK ENOUGH TRACES ##########################
