import numpy as np
import linecache
from utility import *

KEY = [0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75]

class RealTraceHandler:

    def __init__(self):
        pass

    def return_all_values(self, traces=1, offset=0, use_lda=True, trace_range=200, unprofiled=True,
                          use_random_traces=True, seed=0, no_print = True, memory_mapped=True, correlation_threshold = None, normalise_each_trace=False):


        # Input: Number of Repeats and Traces to use
        # Output: real_all_values[repeat][node][trace]

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
        if unprofiled:
            # real_trace_data = np.transpose(np.load(TRACEDATA_EXTRA_FILEPATH, mmap_mode='r'))
            real_trace_data = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH, memory_mapped=memory_mapped)
        else:
            # real_trace_data = np.transpose(np.load(TRACEDATA_FILEPATH, mmap_mode='r'))
            real_trace_data = load_trace_data(filepath=TRACEDATA_FILEPATH, memory_mapped=memory_mapped)
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
        for var, length in variable_dict.iteritems():
            real_all_values[var] = [0] * traces

            # Get Time Points for Variable var
            time_points = np.load("{}{}.npy".format(TIMEPOINTS_FOLDER, var))

            for trace in range(traces):
                real_all_values[var][trace] = np.zeros((length, trace_range))

            for node in range(length):
                tp = time_points[node]
                my_start = tp - (trace_range / 2)
                my_end = tp + (trace_range / 2)
                if my_end == my_start: my_end += 1
                trace_selection = real_trace_data[:, my_start:my_end]
                if normalise_each_trace:
                    trace_selection = normalise_neural_traces(trace_selection)
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
            my_trace = random_traces[trace] if use_random_traces else trace + offset
            # if use_lda:
            for i in range(16):
                real_all_values['p'][trace][:16][i] = np.full(trace_range,
                                                                   plaintexts[my_trace][i])
            # else:
            #     real_all_values['p'][trace][:16] = np.array(plaintexts[my_trace])

        # print "In Real Trace Handler, Real All Values:\n{}\n".format(real_all_values)
        return real_all_values
