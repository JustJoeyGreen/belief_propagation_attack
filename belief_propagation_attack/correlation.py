#!/usr/bin/env python3
from utility import *
import leakageSimulatorAESFurious as lSimF
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linDisAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as quadDisAnalysis
from sklearn.neural_network import MLPClassifier
import argparse
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pandas as pd
import timing

import time #DEBUG

KEY = [0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75]
ATTACK_TRACES = 10000
POI_CAP = 10000 # Because this is computationally expensive, don't need 200000 traces to find PoI, just use 10000
USED_TRACES = None
PRINT = False
save = True
# variable_dict = {'p': 32, 't': 16, 's': 16, 'mc': 16, 'xt': 16, 'cm': 16, 'h': 12}

MEMORY_MAPPED = True

################################################################################


def read_to_list(start, length, number_of_bytes=1, signedint=False, float_coding=False):
    if float_coding:
        data = np.empty(length, dtype=np.float)
    else:
        data = np.empty(length, dtype=np.int16)
    with open(TRACE_FILE, "rb") as binary_file:
        binary_file.seek(start)  # Go to beginning
        for i in range(length):
            # Seek position and read N bytes
            byte = bytearray(binary_file.read(number_of_bytes))
            val = big_endian_to_int(byte, number_of_bytes, signedint, float_coding)
            try:
                data[i] = val
            except OverflowError:
                print "!! Overflow Error: \ni = {}\nNumber of Bytes: {}\nByte: {}\nint_val = {}".format(i,
                                                                                                        number_of_bytes,
                                                                                                        byte, val)
                exit(0)
    return data


def raise_trigger(name, expected, got):
    print "{} Trigger not correct, expected {}, got {}".format(name, expected, got)
    raise IndexError


def parse_header():
    # Parse Header and return important data

    # Read into list
    data = read_to_list(0, 100)

    # print data

    # First, Number of Traces
    if data[0] != 0x41:
        raise_trigger('Trace', 0x41, data[0])

    length = int(data[1])
    offset = 2

    # print data
    # print "Length: {}".format(length)
    # print "Traces List: {}".format(data[offset:offset+length])
    # exit()

    traces_lst = data[offset:offset + length]
    traces = byte_list_to_int(traces_lst)

    # Next, Samples Per Trace
    offset = offset + length

    if data[offset] != 0x42:
        raise_trigger('Samples', 0x42, data[offset])

    length = data[offset + 1]
    offset += 2

    samples_lst = data[offset:offset + length]
    samples = byte_list_to_int(samples_lst)

    # Next, SampleSpace
    offset = offset + length

    if data[offset] != 0x43:
        raise_trigger('SampleSpace', 0x43, data[offset])

    length = data[offset + 1]
    offset += 2

    samplespace_lst = data[offset:offset + length]
    samplespace_val = byte_list_to_int(samplespace_lst)
    samplespace = samplespace_val & 15
    float_coding = samplespace_val & 16
    check_sampleval = samplespace_val & 224

    if check_sampleval > 0:
        print "!! Error: Samplespace Value must have first three bits 0. Found to be {}".format(samplespace_val)
        exit(1)

    # Make sure samplespace is 1, 2, or 4!
    if samplespace != 1 and samplespace != 2 and samplespace != 4:
        print "!! Error: Samplespace can only be 1, 2, or 4. Found to be {}".format(samplespace)
        exit(1)

    # Next, SampleSpace
    offset = offset + length

    if data[offset] != 0x44:
        raise_trigger('DataSpace', 0x44, data[offset])

    length = data[offset + 1]
    offset += 2

    data_space_list = data[offset:offset + length]
    data_space = byte_list_to_int(data_space_list)

    # Finally, find start offset
    start_offset = get_index_of_sublist(data, [0x5F, 0x00]) + 2

    return traces, samples, samplespace, float_coding, data_space, start_offset


def get_and_save_meta():

    print "+ Trace File: {}\n+ Size: {} bytes\n".format(TRACE_FILE, os.path.getsize(TRACE_FILE))

    traces, samples, samplespace, float_coding, data_space, start_offset = parse_header()

    if USED_TRACES is not None:
        traces = min(USED_TRACES, traces)

    attack_traces = ATTACK_TRACES
    profile_traces = max(traces - attack_traces, 0)

    # Save metadata
    save_meta((profile_traces, attack_traces, samples, np.float32 if float_coding else np.int16))


def get_trace_data_and_plaintexts(just_keys_and_plaintexts=False):

    print "+ Trace File: {}\n+ Size: {} bytes\n".format(TRACE_FILE, os.path.getsize(TRACE_FILE))

    traces, samples, samplespace, float_coding, data_space, start_offset = parse_header()

    profile_traces, attack_traces, _, _ = load_meta()

    bytes_in_trace = (samples * samplespace) + data_space
    print "Traces: {}\nSamples: {}\nSample Space: {}\nData Space: {}\nFloat Coding: {}\nStart Offset: {}\n".format(traces, samples, samplespace, data_space, float_coding, start_offset)
    offset = start_offset

    coding = np.float32 if float_coding else np.int16

    if not just_keys_and_plaintexts:
        if MEMORY_MAPPED:
            all_data    = np.memmap(TRACEDATA_FILEPATH, shape=(profile_traces, samples), mode='w+', dtype=coding)
            if profile_traces < traces:
                extra_data  = np.memmap(TRACEDATA_EXTRA_FILEPATH, shape=(attack_traces, samples), mode='w+', dtype=coding)
        else:
            all_data    = np.empty([profile_traces, samples], dtype=coding)
            if profile_traces < traces:
                extra_data  = np.empty([attack_traces, samples], dtype=coding)

    all_plaintexts = np.empty([profile_traces, 16], dtype=np.int16)
    extra_plaintexts = np.empty([attack_traces, 16], dtype=np.int16)
    all_keys = np.empty([profile_traces, 16], dtype=np.int16)
    extra_keys = np.empty([attack_traces, 16], dtype=np.int16)

    percent = traces / 100

    for t in range(traces):

        if PRINT:
            print "*** Trace {} ***".format(t)

        if PRINT:
            print "Length of File: {}".format(os.path.getsize(TRACE_FILE))
            print "Offset: {}".format(offset)
            final_byte = offset + data_space + (samples * samplespace)
            print "Final Byte: {}".format(final_byte)
            print "Is this ok: {}".format(final_byte <= os.path.getsize(TRACE_FILE))
        title_data = read_to_list(offset, data_space)

        if not just_keys_and_plaintexts:
            trace_data = read_to_list(offset + data_space, samples, number_of_bytes=samplespace, signedint=True,
                                      float_coding=float_coding)

        if PRINT:
            print "First 100 values of trace data:\n{}\n".format(list(trace_data[:100]))

        if data_space == 32:
            plaintext = title_data[:16]
            ciphertext = title_data[16:32]
            key = KEY
        elif data_space == 48:
            # plaintext = title_data[:16]
            # key = title_data[16:32]
            # ciphertext = title_data[32:48]
            key = title_data[:16]
            plaintext = title_data[16:32]
            ciphertext = title_data[32:48]

        if PRINT:
            print "Key:        {}".format(key)
            print "Plaintext:  {}".format(plaintext)
            print "Ciphertext: {}".format(ciphertext)
            print_new_line()

        # Simulate
        sim = lSimF.LeakageSimulatorAESFurious()
        sim.fix_key(key)
        sim.fix_plaintext(plaintext)
        sim.simulate(read_plaintexts=0, print_all=0, random_plaintexts=0, affect_with_noise=False,
                     hw_leakage_model=False, real_values=True)
        leakage_dict = sim.get_leakage_dictionary()

        simulated_ciphertext = leakage_dict['p'][0][-16:]
        simulated_end_of_round_one = leakage_dict['p'][0][16:32]
        simulated_end_of_g2 = leakage_dict['t'][0][32:48]

        if PRINT:
            print "* SIMULATED *"
            print "Key:        {}".format(leakage_dict['k'][:16])
            print "Plaintext:  {}".format(leakage_dict['p'][0][:16])
            print "Ciphertext: {}".format(simulated_ciphertext)
            print "Eof Round1: {}".format(simulated_end_of_round_one)
            print "End of G2:  {}".format(simulated_end_of_g2)

            print_new_line()

        # Check for correctness
        if CHECK_CORRECTNESS and not ((ciphertext == simulated_ciphertext).all()
                                      or (ciphertext == simulated_end_of_round_one).all()
                                      or (ciphertext == simulated_end_of_g2).all()):
            print "*** Error in Trace {}: Did not Match!".format(t)
            raise ValueError
        elif PRINT:
            print "+ Checked: Correct!"

        # Add Trace Data to all_data

        if t < profile_traces:
            if not just_keys_and_plaintexts:
                all_data[t] = np.array(trace_data)
            all_plaintexts[t] = np.array(plaintext)
            all_keys[t] = np.array(key)
        else:
            if not just_keys_and_plaintexts:
                extra_data[t - profile_traces] = np.array(trace_data)
            extra_plaintexts[t - profile_traces] = np.array(plaintext)
            extra_keys[t - profile_traces] = np.array(key)

        if (t % percent) == 0:
            print "{}% Complete".format(t / percent)

        if PRINT:
            print "This is what we stored:\n{}\n".format(all_data[t])
            print_new_line()

        # exit(1)

        # Increment offset
        offset = offset + bytes_in_trace

        # # Just first
        # if t > 3:
        #     exit(1)

    if not just_keys_and_plaintexts:
        if MEMORY_MAPPED:
            del all_data
            if profile_traces < traces:
                del extra_data
        else:
            # Save the tranpose as a file!
            np.save(TRACEDATA_FILEPATH, np.transpose(all_data))
            # Save the tranpose as a file!
            np.save(TRACEDATA_EXTRA_FILEPATH, np.transpose(extra_data))

    # Save plaintexts as file
    np.save(PLAINTEXT_FILEPATH, all_plaintexts)
    # Save plaintexts as file
    np.save(PLAINTEXT_EXTRA_FILEPATH, extra_plaintexts)
    # Save keys as file
    np.save(KEY_FILEPATH, all_keys)
    # Save keys as file
    np.save(KEY_EXTRA_FILEPATH, extra_keys)



    print "Saved and Completed!"
    print_new_line()


def simulate_data_from_plaintexts():

    extra = [0, 1]

    # for plaintext_count, plaintext_filepath in enumerate([PLAINTEXT_FILEPATH, PLAINTEXT_EXTRA_FILEPATH]):
    for use_extra_data in extra:

        plaintext_filepath = PLAINTEXT_EXTRA_FILEPATH if use_extra_data else PLAINTEXT_FILEPATH
        key_filepath = KEY_EXTRA_FILEPATH if use_extra_data else KEY_FILEPATH
        # Show plaintexts!
        plaintexts = np.load(plaintext_filepath, mmap_mode='r')
        keys = np.load(key_filepath, mmap_mode='r')
        traces = plaintexts.shape[0]

        # k = np.empty([16, traces], dtype=np.uint8)
        # p = np.empty([32, traces], dtype=np.uint8)
        # t = np.empty([32, traces], dtype=np.uint8)
        # s = np.empty([32, traces], dtype=np.uint8)
        # mc = np.empty([16, traces], dtype=np.uint8)
        # xt = np.empty([16, traces], dtype=np.uint8)
        # cm = np.empty([16, traces], dtype=np.uint8)
        # h = np.empty([12, traces], dtype=np.uint8)

        k = np.empty([48, traces], dtype=np.uint8)
        p = np.empty([48, traces], dtype=np.uint8)
        t = np.empty([48, traces], dtype=np.uint8)
        s = np.empty([48, traces], dtype=np.uint8)
        mc = np.empty([32, traces], dtype=np.uint8)
        xt = np.empty([32, traces], dtype=np.uint8)
        cm = np.empty([32, traces], dtype=np.uint8)
        h = np.empty([24, traces], dtype=np.uint8)
        sk = np.empty([8, traces], dtype=np.uint8)
        xk = np.empty([2, traces], dtype=np.uint8)

        for i, (plaintext, key) in enumerate(zip(plaintexts, keys)):

            if PRINT:
                print "Trace {}\nPlaintext: {}\nKey: {}".format(i, plaintext, key)

            sim = lSimF.LeakageSimulatorAESFurious()
            sim.fix_key(key)
            sim.fix_plaintext(plaintext)
            sim.simulate(read_plaintexts=0, print_all=0, random_plaintexts=0, affect_with_noise=False,
                         hw_leakage_model=False, real_values=True)
            leakage_dict = sim.get_leakage_dictionary()

            for j in range(48):
                # p
                p[j][i] = leakage_dict['p'][0][j]
                # t
                t[j][i] = leakage_dict['t'][0][j]
                # s
                s[j][i] = leakage_dict['s'][0][j]

                # k
                k[j][i] = leakage_dict['k'][j]

                if j < 32:
                    # mc
                    mc[j][i] = leakage_dict['mc'][0][j]
                    # xt
                    xt[j][i] = leakage_dict['xt'][0][j]
                    # cm
                    cm[j][i] = leakage_dict['cm'][0][j]

                    if j < 24:
                        # h
                        h[j][i] = leakage_dict['h'][0][j]

                        if j < 8:
                            sk[j][i] = leakage_dict['sk'][j]

                            if j < 2:
                                xk[j][i] = leakage_dict['xk'][j]

            if traces < 100:
                print "Finished Trace {}".format(i)
            elif i % (traces // 100) == 0:
                print "{}% Complete".format(i / (traces // 100))

        # Save to files!
        extra_string = "extra_" if use_extra_data == 1 else ""
        np.save(REALVALUES_FOLDER + extra_string + 'k.npy', k)
        np.save(REALVALUES_FOLDER + extra_string + 'p.npy', p)
        np.save(REALVALUES_FOLDER + extra_string + 't.npy', t)
        np.save(REALVALUES_FOLDER + extra_string + 's.npy', s)
        np.save(REALVALUES_FOLDER + extra_string + 'mc.npy', mc)
        np.save(REALVALUES_FOLDER + extra_string + 'xt.npy', xt)
        np.save(REALVALUES_FOLDER + extra_string + 'cm.npy', cm)
        np.save(REALVALUES_FOLDER + extra_string + 'h.npy', h)
        np.save(REALVALUES_FOLDER + extra_string + 'sk.npy', sk)
        np.save(REALVALUES_FOLDER + extra_string + 'xk.npy', xk)

        print "Saved and Completed!"
        print_new_line()


# noinspection PyTypeChecker
def point_of_interest_detection(save_file=True, hw=True, variables=None):
    # Correlation!
    # for each variable, load the .npy
    # then load matrix trace_data
    # For each time point in trace_data (loop through), perform correlation and keep track of max (absolute)

    print "Loading Matrix trace_data, may take a while..."
    # trace_data = np.load(TRACEDATA_FILEPATH, mmap_mode='r')
    trace_data = np.transpose(load_trace_data(memory_mapped = MEMORY_MAPPED))[:, :POI_CAP]
    print "...done!"
    print_new_line()

    samples, traces = trace_data.shape

    for var, number_of_nodes in variable_dict.iteritems():

        var_array_real = np.load(REALVALUES_FOLDER + var + '.npy', mmap_mode='r')

        if hw:
            var_array = get_hw_of_vector(var_array_real)
        else:
            var_array = np.copy(var_array_real)

        print "Variable {:3} ({}):\n{}".format(var, var_array.shape, var_array)

        for j in range(number_of_nodes):

            if np.std(var_array[j]) == 0.0:
                print "Standard Deviation of Variable {}[{}] is 0, cannot find trace point!".format(var, j)
            else:

                MAX_ = samples
                TOP_N = 15

                # coeff_array = np.zeros([MAX_])
                # for sample in range(MAX_):
                #     # CHECK STANDARD DEVIATION
                #     if np.std(trace_data[sample]) > 0.0:
                #         coeff = np.abs(np.corrcoef(var_array[j], trace_data[sample])[1, 0])
                #         if coeff > 1:
                #             print "* Error! Coeff = {}.\nVar = {}, j = {}, sample = {}.\nvar_array[j]:\n{}\n\n".format(
                #                 coeff, var, j, sample, var_array[j])
                #             raise ValueError
                #         coeff_array[sample] = coeff

                # print "DEBUG"
                # print var_array[j]
                # print var_array[j].shape
                # print POI_CAP
                # print var_array[j][:POI_CAP]
                # print np.corrcoef(var_array[j][:POI_CAP], sample)
                # print np.abs(np.corrcoef(var_array[j][:POI_CAP], sample))
                # print np.abs(np.corrcoef(var_array[j][:POI_CAP], sample)).shape
                #
                # exit(1)


                coeff_array = np.array([np.abs(np.corrcoef(var_array[j][:POI_CAP], timeslice))[0, 1] for timeslice in trace_data])

                top_n = coeff_array.argsort()[-TOP_N:][::-1]
                print "Top {} Coefficient Indexes for Variable {}[{}]:".format(TOP_N, var, j)
                for c, index in enumerate(top_n):
                    print "Rank {}: Time Point {} ({})".format(c, index, coeff_array[index])
                print "-> Average Time Point: {}".format(int(round(np.mean(top_n))))
                print "-> Median Time Point:  {}".format(np.median(top_n))
                print_new_line()

                # Store!
                if save_file:
                    np.save("{}{}_{}{}.npy".format(COEFFICIENT_FOLDER, var, j, '_HW' if hw else ''), coeff_array)

                # Stop for now!
                # exit(1)

        print_new_line()

        # Just p for now
        # break

    print "Saved and Completed!"
    print_new_line()


def get_time_points_for_each_node(hw=True):

    # print "Loading Matrix trace_data, may take a while..."
    # trace_data = np.load(TRACEDATA_FILEPATH)
    # print "...done!"
    # print_new_line()

    # samples, traces = trace_data.shape

    for var, number_of_nodes in variable_dict.iteritems():

        if PRINT:
            print "* Variable {}".format(var)

        var_array_real = np.load(REALVALUES_FOLDER + var + '.npy', mmap_mode='r')
        # if hw_method:
        #     var_array = get_hw_of_vector(var_array_real)
        # else:
        #     var_array = var_array_real

        time_points = np.zeros([number_of_nodes], dtype=np.uint32)

        for j in range(number_of_nodes):
            # Load
            coeff_array = np.load("{}{}_{}{}.npy".format(COEFFICIENT_FOLDER, var, j, '_HW' if hw else ''), mmap_mode='r')

            # Find the Highest Correlated Time Point
            poi = np.argmax(coeff_array)

            time_points[j] = poi

        print "Variable {}:\n{}\n\n".format(var, time_points)

        # Save
        if save:
            np.save("{}{}.npy".format(TIMEPOINTS_FOLDER, var), time_points)

    print "Saved and Completed!"
    print_new_line()


# def get_power_values_for_each_node():
#
#     for data_count, trace_data_path in enumerate([TRACEDATA_FILEPATH, TRACEDATA_EXTRA_FILEPATH]):
#
#         print "Loading Matrix trace_data {}, may take a while...".format(data_count)
#         trace_data = (load_trace_data(filepath=trace_data_path, memory_mapped = MEMORY_MAPPED))
#         print "...done!"
#         print_new_line()
#
#         traces, samples = trace_data.shape
#         print traces, samples
#
#         for var, number_of_nodes in variable_dict.iteritems():
#
#             if PRINT:
#                 print "* Variable {}".format(var)
#
#             # Get Time Points
#             time_points = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var))
#
#             power_values = np.zeros([traces, number_of_nodes])
#
#             for trace in range(traces):
#                 if PRINT:
#                     print "- Trace {}".format(trace)
#                 print trace_data[trace].shape, time_points
#                 power_values[trace] = np.take(trace_data[trace], time_points)
#
#             # Save
#             extra_string = "extra_" if data_count == 1 else ""
#             np.save("{}{}{}.npy".format(POWERVALUES_FOLDER, extra_string, var), power_values)
#
#     print "Saved and Completed!"
#     print_new_line()


def elastic_alignment(start_trace=0):
    # Load the traces
    print "Loading Matrix trace_data (and transposing), may take a while..."
    # trace_data = np.transpose(np.load(TRACEDATA_FILEPATH))
    trace_data = np.transpose(load_trace_data(memory_mapped = MEMORY_MAPPED))
    print "...done!"
    print_new_line()
    print "Loading Matrix extra_trace_data (and transposing), may take a while..."
    extra_trace_data = np.transpose(load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH, memory_mapped = MEMORY_MAPPED))
    print "...done!"
    print_new_line()

    # Set up Reference Trace
    reference_trace = trace_data[0]

    if PRINT:
        print "Reference Trace:"
        print reference_trace
        print ""

    # BOTH TRACE SETS
    # for set_i, trace_set in enumerate([trace_data, extra_trace_data]):
    # for set_i, trace_set in enumerate([trace_data]):
    for set_i, trace_set in enumerate([extra_trace_data]):

        if PRINT:
            print "Generating Traces for Set {}...\n".format(set_i)

        traces, samples = trace_set.shape

        # Aligned Trace set
        # aligned_trace_data = np.empty(trace_set.shape)

        # For all other traces:
        for trace_index in range(start_trace, traces):

            if not check_file_exists(ELASTIC_FOLDER + '{}.npy'.format(trace_index)):

                # Make the file so it doesn't get overwritten
                pickle.dump(np.empty(1), open(ELASTIC_FOLDER + '{}.npy'.format(trace_index), 'wb'))

                # Set up new samples
                new_trace = np.empty(samples)

                if PRINT:
                    print "FastDTW for Trace {}...".format(trace_index)
                current_trace = trace_set[trace_index]
                distance, path = fastdtw(reference_trace, current_trace, dist=euclidean)
                for sample_i in range(samples):
                    y_values = np.array([current_trace[v[1]] for i, v in enumerate(path) if v[0] == sample_i])
                    new_trace[sample_i] = (1.0 / len(y_values)) * np.sum(y_values)
                # Dump to fastdtw/

                if PRINT:
                    print "...done!\nDistance: {}\nPath:\n{}\nNew Trace:\n{}\n".format(distance, path, new_trace)

                pickle.dump(new_trace, open(ELASTIC_FOLDER + '{}.npy'.format(trace_index), 'wb'))

            else:
                if PRINT:
                    print "Detected file for FastDTW Trace {}, skipping".format(trace_index)

        # # Save
        # if set_i == 0:
        #     pickle.dump(aligned_trace_data, open(NUMPY_ELASTIC_TRACE_FILE, "wb"))
        # else:
        #     pickle.dump(aligned_trace_data, open(NUMPY_ELASTIC_EXTRA_TRACE_FILE, "wb"))
        # print "...saved!\n"


def combine_elastic_alignment(my_traces=10000):
    fastdtw_trace_data = np.zeros((my_traces, 40000))

    for trace_index in range(my_traces):
        next_trace = np.load('{}{}.npy'.format(ELASTIC_FOLDER, trace_index))
        fastdtw_trace_data[trace_index] = next_trace
        np.save("{}elastictracedata.npy".format(ELASTIC_FOLDER), fastdtw_trace_data)


def dimensionality_reduction(tprange=200):
    print "Dimensionality Reduction!"

    # Need: trace_data for samples, Actual Values of each node

    print "Loading Matrix trace_data, may take a while..."
    trace_data = np.transpose(load_trace_data(memory_mapped = MEMORY_MAPPED))
    print "...done!"
    print_new_line()
    samples, traces = trace_data.shape

    # profile_traces = int(traces * 0.8)

    for variable, length in variable_dict.iteritems():

        # if PRINT:
        print "Generating Template for Variable {}, Length {}".format(variable, length)

        real_values = np.load("{}{}.npy".format(REALVALUES_FOLDER, variable))
        time_points = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, variable))

        for i in range(length):

            if PRINT:
                print "Template for {}{}".format(variable, pad_string_zeros(i))

            y = real_values[i, :traces]
            X = trace_data[:, time_points[i] - (tprange / 2):time_points[i] + (tprange / 2)]

            if PRINT:
                print "Real Values: {}".format(y)
                print "Traces Range: {}".format(X)

            # Quick check
            if y.shape[0] != X.shape[0]:
                print "Different shapes encountered, can't fit together! X is {}, y is {}".format(X.shape, y.shape)
                exit(1)

            test_x = np.array([np.arange(200)])

            # # Set up linDisAnalysis
            # lda = linDisAnalysis()
            # lda.fit(X, y)
            # z_lda = lda.predict(test_x)

            # # Set up quadDisAnalysis
            # qda = quadDisAnalysis()
            # qda.fit(X, y)
            # z_qda = qda.predict(test_x)

            # mlp = MLPClassifier()
            # mlp.fit(X, y)
            # z_mlp = mlp.predict(test_x)


def lda_templates(tprange=200, validation_traces=10000):
    print "linDisAnalysis Templates!"

    # Need: trace_data for samples, Actual Values of each node

    print "Loading Matrix trace_data, may take a while..."
    trace_data = load_trace_data(memory_mapped = MEMORY_MAPPED)
    print "...done!"
    print_new_line()
    traces, samples = trace_data.shape

    # profile_traces = int(traces * 0.8)

    for variable, length in variable_dict.iteritems():

        # if variable not in ['xt','k','h']:
        #     print "DEBUG: Skipping {}".format(variable)
        #     continue

        # if PRINT:
        print "Generating Template for Variable {}, Length {}".format(variable, length)

        real_values = np.load("{}{}.npy".format(REALVALUES_FOLDER, variable))
        time_points = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, variable))

        for i in range(length):

            if PRINT:
                print "Template for {}{}".format(variable, pad_string_zeros(i))

            y = real_values[i, :traces-validation_traces]
            X = trace_data[:-validation_traces, time_points[i] - (tprange / 2):time_points[i] + (tprange / 2)]

            if PRINT:
                print "Real Values: {}".format(y)
                print "Traces Range: {}".format(X)

            # Quick check
            if y.shape[0] != X.shape[0]:
                print "Different shapes encountered, can't fit together! X is {}, y is {}".format(X.shape, y.shape)
                exit(1)

            # Set up linDisAnalysis
            lda = linDisAnalysis()
            lda.fit(X, y)

            # Save
            pickle.dump(lda, open("{}{}_{}_{}.p".format(LDA_FOLDER, tprange, variable, i), "wb"))


def get_mean_and_sigma_for_each_node(hw_musig=False, hw_tp=True, save=True, validation_traces=10000):
    musig_dict_path = MUSIGMA_FILEPATH

    NUMBER_OF_TEMPLATES = 256

    # Clear current CSV

    musig_dict = dict()

    # Get Mean and Sigma for each node

    print "Loading Matrix trace_data, may take a while..."
    trace_data = np.transpose(load_trace_data(memory_mapped = MEMORY_MAPPED))
    print "...done!"
    print_new_line()

    # samples, traces = trace_data.shape

    for var, number_of_nodes in variable_dict.iteritems():

        if PRINT:
            print "* Variable {}".format(var)

        var_array = np.load(REALVALUES_FOLDER + var + '.npy')
        if hw_musig:
            var_array = get_hw_of_vector(var_array)

        for j in range(number_of_nodes):

            # Load
            coeff_array = np.load("{}{}_{}{}.npy".format(COEFFICIENT_FOLDER, var, j, '_HW' if hw_tp else ''))

            # Find the Highest Correlated Time Point
            poi = np.argmax(coeff_array)

            # print "Variable {} {}: POI {}".format(var, j, poi)

            # Find All Trace Values for this Time Point
            trace_values = trace_data[poi, :-validation_traces]

            # Save Mean and Sigma
            var_string = "{}{}".format(var, pad_string_zeros(j + 1))
            numpy_array = np.zeros((NUMBER_OF_TEMPLATES, 2))

            # Save Mean and Sigma for Each Hamming Weight
            if PRINT:
                print "For Variable {}:".format(var_string)
            for value in range(NUMBER_OF_TEMPLATES):
                # Get list of traces where hw is used
                target_traces = get_list_of_value_matches(var_array[j], value)
                if len(target_traces) == 0:
                    print_new_line()
                    print "ERROR: No Traces found where Variable {} has Value {}!".format(var_string, value)
                    print_new_line()
                    raise IOError
                # if PRINT:
                #     print "Value {}: {}".format(value, target_traces)
                # Get Power Values for these traces in numpy array

                ### METHOD 1:
                # x_power_values = np.take(trace_values, target_traces) #SLOW
                ### METHOD 2:
                # x_power_values = np.zeros(target_traces.shape[0])
                # for x_count, x_val in enumerate(target_traces):
                #     x_power_values[x_count] = trace_values[x_val]
                ### METHOD 3:
                x_power_values = trace_values[target_traces]

                # Get hw and STD
                mu = np.mean(x_power_values)
                sigma = np.std(x_power_values)
                # Store
                numpy_array[value] = (mu, sigma)

                if PRINT:
                    print "Value {}: {}".format(value, (mu, sigma))



            # Save
            musig_dict[var_string] = numpy_array

    for key, val in musig_dict.iteritems():
        print "* {} *\n\n{}\n\n".format(key, val)

    if save:
        pickle.dump(musig_dict, open(musig_dict_path, 'wb'))
        print "Saved and Completed!"
        print_new_line()


def get_snrs():
    musigma_dict_path = MUSIGMA_FILEPATH
    musigma_dict = pickle.load(open(musigma_dict_path, 'ro'))

    number_of_variables = sum(variable_dict.values())

    snr_list = np.zeros([number_of_variables])
    snr_real_list = np.zeros([number_of_variables])
    snr_log_list = [[] for i in range(30)]
    count = 0

    for var_name, musigma_array in musigma_dict.iteritems():

        snr = np.var(musigma_array[0]) / np.mean(musigma_array[1] ** 2)

        print "Variable {:5}: 2^{:2} ({})".format(var_name, smallest_power_of_two(snr), snr)

        snr_list[count] = smallest_power_of_two(snr)
        snr_real_list[count] = snr

        snr_log_list[-smallest_power_of_two(snr)].append(var_name)
        count += 1

    print_statistics(list(snr_list))

    print "Sorted by Best to Worst (logs):\n"
    for i, lst in enumerate(snr_log_list):
        if len(lst) > 0:
            print "2^-{}:\n{}\n".format(i, lst)
    print_new_line()

    np.save('{}snrlist.npy'.format(TRACE_FOLDER), snr_real_list)


def matching_performance():
    # Match to extra

    extra_plaintexts = np.load(PLAINTEXT_EXTRA_FILEPATH)
    extra_keys = np.load(KEY_EXTRA_FILEPATH)
    extra_traces = np.transpose(load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH, memory_mapped = MEMORY_MAPPED))

    musig_dict_path = MUSIGMA_FILEPATH
    musig_dict = pickle.load(open(musig_dict_path, 'ro'))

    # Containers to hold ranks
    rank_dict = {}
    for v, length in variable_dict.iteritems():
        rank_dict[v] = [[] for _ in range(length)]
    all_ranks = []
    trace_average_rank_holder = []

    try:

        for i, (plaintext, key) in enumerate(zip(extra_plaintexts, extra_keys)):

            print "Trace {:5}: {}".format(i, plaintext)

            # Simulate actual values
            sim = lSimF.LeakageSimulatorAESFurious()
            sim.fix_key(key)

            sim.fix_plaintext(plaintext)

            sim.simulate(read_plaintexts=0, print_all=0, random_plaintexts=0, affect_with_noise=False,
                         hw_leakage_model=False, real_values=True)
            leakage_dict = sim.get_leakage_dictionary()

            # print 'Plaintext:', plaintext

            # For each node in graph, get time point -> corresponding power value from extra trace data
            # Template match this against MuSigma pairs, get probability distribution
            # Check to see how high actual value scores

            trace_average_rank_list = []

            for var, musigma_array in sorted(musig_dict.iteritems()):
                # Get var name and var number
                var_name, var_number, _ = split_variable_name(var)

                # Actual value of the node
                actual_value = int(leakage_dict[var_name][0][var_number - 1])

                # Time point of node in trace
                time_points = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var_name))

                time_point = time_points[var_number - 1]

                # print var
                # print 'Actual Value:', actual_value

                # Power value received
                power_value = extra_traces[i][time_point]
                # print power_value

                # Real Value Match
                matched_dist = real_value_match(var, power_value)
                # print matched_dist
                ranked_dist = np.array(matched_dist).argsort()[::-1]
                # print ranked_dist
                rank = ranked_dist[actual_value]
                # print rank
                #
                # print zip(ranked_dist, matched_dist)
                #
                # exit(1)

                # print "-> {} {}, Ranked: {}".format(var_name, var_number, ranked_dist[actual_value])
                # Add to Ranked List
                rank_dict[var_name][var_number - 1].append(rank)
                all_ranks.append(rank)
                trace_average_rank_list.append(rank)

            trace_average_rank_holder.append(get_average(trace_average_rank_list))

            # if i >= 100:
            #     break

    except KeyboardInterrupt:
        pass
    finally:

        # Print Statistics
        for v, l in rank_dict.iteritems():
            for i, lst in enumerate(l):
                print "{}{}:\n".format(v, pad_string_zeros(i + 1))
                print_statistics(lst)

        # ALL
        print "* ALL NODES RANK STATISTICS *"
        print_statistics(all_ranks)

        print "* AVERAGE RANK PER TRACE STATISTICS *"
        print_statistics(trace_average_rank_holder)


def lda_matching_performance(tprange=200):
    # Match to extra

    extra_plaintexts = np.load(NUMPY_EXTRA_PLAINTEXT_FILE)
    extra_traces = np.transpose(load_trace_data(filepath = NUMPY_EXTRA_TRACE_FILE, memory_mapped = MEMORY_MAPPED))

    print extra_traces.shape

    # Containers to hold ranks
    rank_dict = {}
    for v, length in variable_dict.iteritems():
        rank_dict[v] = [[] for x in range(length)]
    all_ranks = []
    trace_average_rank_holder = []

    try:

        for i, plaintext in enumerate(extra_plaintexts):

            print "Trace {:5}: {}".format(i, plaintext)

            # Simulate actual values
            sim = lSimF.LeakageSimulatorAESFurious()
            print "TODO"
            break
            sim.fix_key(KEY)
            sim.fix_plaintext(plaintext)
            sim.simulate(read_plaintexts=0, print_all=0, random_plaintexts=0, affect_with_noise=False,
                         hw_leakage_model=False, real_values=True)
            leakage_dict = sim.get_leakage_dictionary()

            # For each node in graph, get time point -> corresponding power value from extra trace data
            # Template match this against MuSigma pairs, get probability distribution
            # Check to see how high actual value scores

            trace_average_rank_list = []

            for var_name, vlength in variable_dict.iteritems():

                time_points = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var_name))

                for var_number in range(vlength):
                    # Get Time Point
                    time_point = time_points[var_number - 1]

                    # Load linDisAnalysis
                    lda = pickle.load(
                        open("{}{}_{}_{}.p".format(LDA_FOLDER, tprange, var_name, var_number), "ro"))
                    # Get Trace Data around time point
                    X = extra_traces[i, time_point - (tprange / 2):time_point + (tprange / 2)]

                    # Predict Values
                    predicted_probabilities = lda.predict_proba([X])[0]

                    # Get Actual Values
                    actual_value = (leakage_dict[var_name][0][var_number]).astype(np.uint8)

                    # Get Rank
                    temp = predicted_probabilities.argsort()[::-1]
                    ranked_dist = np.empty_like(temp)
                    ranked_dist[temp] = np.arange(len(predicted_probabilities))

                    rank = ranked_dist[actual_value] + 1
                    # top_ranked = np.where(ranked_dist == 0)
                    max_prob = np.max(predicted_probabilities)
                    # max_index = np.where(predicted_probabilities == max_prob)

                    # if var_name == 't':
                    #     print "TEST!"
                    #     print "Variable {}{}".format(var_name, var_number)
                    #     print "Real Value: {}".format(actual_value)
                    #     print "Predicted Probabilities:\n{}\n".format(predicted_probabilities)
                    #     print "Top Ranked: {} ({})".format(top_ranked, predicted_probabilities[top_ranked])
                    #     print "CHECK: Max Value: {} ({})".format(max_prob, max_index)
                    #     print "Our Rank: {} ({})".format(rank, predicted_probabilities[actual_value])

                    # exit(1)

                    # Add to Ranked List
                    rank_dict[var_name][var_number - 1].append(rank)
                    all_ranks.append(rank)
                    trace_average_rank_list.append(rank)

                trace_average_rank_holder.append(get_average(trace_average_rank_list))

            # exit(1)



    except KeyboardInterrupt:
        pass
    finally:

        # Print Statistics
        for v, l in rank_dict.iteritems():
            for i, lst in enumerate(l):
                print "{}{}:\n".format(v, pad_string_zeros(i + 1))
                print_statistics(lst)

        # ALL
        print "* ALL NODES RANK STATISTICS *"
        print_statistics(all_ranks)

        print "* AVERAGE RANK PER TRACE STATISTICS *"
        print_statistics(trace_average_rank_holder)

        pass


# def get_feature_column_csv(prev_and_sub=True):
#
#     if prev_and_sub:
#         for i in range(2):
#             extra_string = "extra_" if i == 1 else ""
#             trace_data_name = TRACEDATA_EXTRA_FILEPATH if i == 1 else TRACEDATA_FILEPATH
#             print "Loading from {}...".format(trace_data_name)
#             trace_data = np.transpose(load_trace_data(trace_data_name, memory_mapped = MEMORY_MAPPED))
#             print "...done"
#             for var_name, total_vars in variable_dict.iteritems():
#                 time_points = np.load(TIMEPOINTS_FOLDER + var_name + '.npy')
#                 real_values = np.load(REALVALUES_FOLDER + extra_string + var_name + '.npy')
#                 for var_number in range(total_vars):
#                     time_point = time_points[var_number]
#                     data = trace_data[:, time_point-1:time_point+2]
#                     labels = real_values[var_number, :]
#                     print data.shape
#                     print np.unique(data, axis=0).shape
#                     dataframe = pd.DataFrame(data = data, columns=['PreviousPowerValue', 'TargetPowerValue', 'SubsequentPowerValue'])
#
#                     for j, value in enumerate(labels):
#                         if value == 139:
#                             print (dataframe.iloc[j])
#
#                     # Save
#                     np.save('{}{}{}{}.npy'.format(LABEL_FOLDER, extra_string, var_name, var_number), labels)
#                     dataframe.to_csv('{}{}{}{}.csv'.format(FEATURECOLUMN_FOLDER, extra_string, var_name, var_number))
#     else:
#         # Need: real values for each variable, plus power values
#         # Loop through variable names
#         for i in range(2):
#             extra_string = "extra_" if i == 1 else ""
#             for var_name, total_vars in variable_dict.iteritems():
#                 power_values = np.transpose(np.load(POWERVALUES_FOLDER + extra_string + var_name + '.npy'))
#                 real_values = np.load(REALVALUES_FOLDER + extra_string + var_name + '.npy')
#                 for var_number in range(total_vars):
#                     data = power_values[var_number, :]
#                     labels = real_values[var_number, :]
#                     dataframe = pd.DataFrame(data = data, columns=['PowerValue'])
#                     # Save
#                     np.save('{}{}{}{}.npy'.format(LABEL_FOLDER, extra_string, var_name, var_number), labels)
#                     dataframe.to_csv('{}{}{}{}.csv'.format(FEATURECOLUMN_FOLDER, extra_string, var_name, var_number))


def check_directories():
    for folder in ALL_DIRECTORIES:
        if not os.path.isdir(folder):
            os.makedirs(folder)


def ALL(skip_code = 0):

    # Always check file exists
    if not check_file_exists(TRACE_FILE):
        print "!!! Trace file {} does not exist!".format(TRACE_FILE)
        exit(1)

    # FIRST: Check and Set Up Folders!
    # Utility: Check to see what has not been done?
    if skip_code <= 0:
        print "> Checking Directories..."
        check_directories()
        print "> Getting and Setting Meta Data..."
        get_and_save_meta()

    # Get Trace Data!
    if skip_code <= 1:
        print "> Getting trace_data and Plaintexts"
        get_trace_data_and_plaintexts()

    # Simulates and Creates Arrays for each variable
    if skip_code <= 2:
        print "> Simulating Data from Plaintexts"
        simulate_data_from_plaintexts()

    # Correlates hw Arrays to trace_data time points, stores each node
    if skip_code <= 3:
        print "> Points of Interest Detection"
        point_of_interest_detection()

    # Stores Mean and Standard Deviation for each node
    if skip_code <= 4:
        print "> Getting Mean and Sigma for Each Detected Point"
        get_mean_and_sigma_for_each_node()

    # Gets Time and Power Points
    if skip_code <= 5:
        print "> Getting Time Points for Each Detected Point"
        get_time_points_for_each_node()
        # print "> Getting Power Values for Each Detected Time Point"
        # get_power_values_for_each_node()

    # Get linDisAnalysis Templates
    if skip_code <= 6:
        print "> Getting linDisAnalysis Templates"
        lda_templates()

    # # Prints out SNRs
    # if skip_code <= 7:
    #     print "> Printing SNRs"
    #     get_snrs()
    #
    # # Matching Performance
    # if skip_code <= 8:
    #     print "> Matching Performance"
    #     matching_performance()


######################################## STEP 1: PARSE HEADER ########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculates Distance Statistics from Target Node')
    parser.add_argument('--PRINT', action="store_true", dest="PRINT", help='Prints Results (default: False)',
                        default=False)
    parser.add_argument('--CHECK', action="store_true", dest="CHECK_CORRECTNESS",
                        help='Check Correctness (default: False)', default=False)
    parser.add_argument('-tprange', action="store", dest="TPRANGE", help='TPRANGE (default: 200)', type=int,
                        default=200)
    parser.add_argument('-start', action="store", dest="START_TRACE",
                        help='Start Trace for Elastic Alignment (default: 0)', type=int,
                        default=0)

    parser.add_argument('-skip', '-s', action="store", dest="SKIP_CODE",
                        help='Skip Code (default: 0)', type=int,
                        default=0)

    # Target node here
    args = parser.parse_args()
    TPRANGE = args.TPRANGE
    PRINT = args.PRINT
    CHECK_CORRECTNESS = args.CHECK_CORRECTNESS
    START_TRACE = args.START_TRACE
    SKIP_CODE = args.SKIP_CODE

    # TODO
    # ALL(skip_code=SKIP_CODE)

    # get_mean_and_sigma_for_each_node(save=False)

    lda_templates()

    # print "> Points of Interest Detection with Hamming Weights!"
    # point_of_interest_detection(hw=True)

    # a = load_trace_data(memory_mapped = MEMORY_MAPPED)
    #
    # print (a[0])
    # print_new_line()
    # print (a[-1])

    # matching_performance()

    print "All Done! :)"
    exit(1)

# Trace 0: [105 216 224  59 183  59  85 189 197  89  45 141 228  36 137 228]
# Trace 1: [118 115 255 133 173 120 203  38 233  42 127  99 208  89  75  28]
# Trace 2: [ 34 132  68 210 122  81  72  95 118  88 205 245  45 246  19 116]
# Trace 3: [243  64  33 125 236 218 112  74 173  88  53 205  91   5  27 121]
# Trace 4: [ 50 172  63 224 154 169 239   0  45  73 128  31 204 138  70  22]
# Trace 5: [ 57 243 143  49  41  70 194 206 164  61  96 141 156 134 249  54]

# SBOX Output s001: Riscre found Timepoint 13296
# Time point 0, Sample 0 = 16727
