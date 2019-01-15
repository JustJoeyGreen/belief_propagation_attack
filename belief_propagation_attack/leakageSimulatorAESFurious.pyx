import random
import pickle
import numpy as np
import linecache
from utility import *
cimport numpy as np

shift_rows_s = list([
    0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11
])

BAD_LEAKAGE_CONSTANT = 0.03125 #2**-5

class LeakageSimulatorAESFurious:

    """ Simulate Leakage for AES Furious
    http://point-at-infinity.org/avraes/rijndaelfurious.asm.html
    """
    def __init__(self, seed = 1):
        # print "+ Initialising Simulator with seed {:3} +".format(seed)
        self.seed(seed + 1)
        self.key = get_random_bytes()
        self.plaintext = get_random_bytes()
        self.simulated_dictionary = {}

    def fix_key(self, key):
        self.key = key

    def fix_plaintext(self, plaintext):
        if len(plaintext) != 16:
            print "!!! Plaintext must be 16 bytes, trying to fix {}".format(len(plaintext))
            raise IndexError
        self.plaintext = plaintext

    def randomise_plaintext(self):
        self.plaintext = get_random_bytes()

    def seed(self, seed = 0):
        random.seed(seed)
        np.random.seed(seed)

    def simulate(self, float snr = 32.0, int traces = 1, int offset = 0, int read_plaintexts = 0, int random_plaintexts = 1, badly_leaking_nodes = None, badly_leaking_traces = None, badly_leaking_snr = 0.1, no_noise_nodes = None, threshold = None, int local_leakage = 1, int print_all = 0, affect_with_noise = True, hw_leakage_model = False, real_values = False, rounds_of_aes = 10,
    average_key_nodes = False):

        cdef int i, j, trace, index

        # Set up dictionary
        dictionary = {}

        # Shared Variables (keys)
        cdef np.ndarray k       = get_empty_int_array(size = 16 + (rounds_of_aes * 16))
        cdef np.ndarray sk      = get_empty_int_array(size = (rounds_of_aes * 4))
        cdef np.ndarray xk      = get_empty_int_array(size = rounds_of_aes)
        cdef np.ndarray rc      = get_empty_int_array(size = rounds_of_aes + 1)

        if badly_leaking_nodes is None:
            badly_leaking_nodes = []
        if badly_leaking_traces is None:
            badly_leaking_traces = []
        if no_noise_nodes is None:
            no_noise_nodes = []


        # Key
        k[:len(self.key)] = self.key

        # All Round Keys
        # rc
        rc[0] = 1
        for i in range(rounds_of_aes):
            rc[i+1] = xtimes(rc[i])

        # sk, xk, k
        for i in range(0, (16 * rounds_of_aes), 16):
            sk[(i/4)  ] = sbox[k[i+13]]
            sk[(i/4)+1] = sbox[k[i+14]]
            sk[(i/4)+2] = sbox[k[i+15]]
            sk[(i/4)+3] = sbox[k[i+12]]
            # XOR with rc
            xk[(i/16)]  = sk[(i/4)]   ^ k[i  ]
            # First
            k[i+16]     = xk[(i/16)]  ^ rc[(i/16)]
            # Others
            k[i+17]     = sk[(i/4)+1] ^ k[i+1]
            k[i+18]     = sk[(i/4)+2] ^ k[i+2]
            k[i+19]     = sk[(i/4)+3] ^ k[i+3]
            # Loop the rest
            if rounds_of_aes > 1:
                for j in range(i, i + 12):
                    k[j+20] = k[j+4] ^ k[j+16]

        # Save Hamming Weights to dictionary
        dictionary['key']   = self.key

        # Everything else
        dictionary['k']     = np.zeros((traces,16 + (rounds_of_aes * 16)))
        dictionary['sk']     = np.zeros((traces,(rounds_of_aes * 4)))
        dictionary['xk']     = np.zeros((traces,rounds_of_aes))
        dictionary['p']     = np.zeros((traces,16 + (rounds_of_aes * 16)))
        dictionary['t']     = np.zeros((traces,16 + (max(0, rounds_of_aes - 1) * 16)))
        dictionary['s']     = np.zeros((traces,16 + (max(0, rounds_of_aes - 1) * 16)))
        dictionary['xt']    = np.zeros((traces,min(144, rounds_of_aes * 16)))
        dictionary['cm']    = np.zeros((traces,min(144, rounds_of_aes * 16)))
        dictionary['mc']    = np.zeros((traces,min(144, rounds_of_aes * 16)))
        dictionary['h']     = np.zeros((traces,min(108, rounds_of_aes * 12)))

        for trace in range (traces):

            # Independent
            p       = get_empty_int_array(16 + (rounds_of_aes * 16))
            t       = get_empty_int_array(16 + (max(0, rounds_of_aes - 1) * 16))
            s       = get_empty_int_array(16 + (max(0, rounds_of_aes - 1) * 16))
            shift   = get_empty_int_array(16 + (max(0, rounds_of_aes - 1) * 16))
            xt      = get_empty_int_array(min(144, rounds_of_aes * 16))
            cm      = get_empty_int_array(min(144, rounds_of_aes * 16))
            mc      = get_empty_int_array(min(144, rounds_of_aes * 16))
            h       = get_empty_int_array(min(108, rounds_of_aes * 12))

            # Get Plaintext Bytes
            if read_plaintexts:
                p_backup = np.zeros(16)
                try:
                    for i in range(1, 17):
                        if local_leakage:
                            if hw_leakage_model:
                                line = linecache.getline(PATH_TO_LOCAL_HW + 'printdata.txt', i + ((trace+offset) * 16))
                            else:
                                line = linecache.getline(PATH_TO_LOCAL_ELMO + 'printdata.txt', i + ((trace+offset) * 16))
                        else:
                            line = linecache.getline(PATH_TO_ELMO + 'output/printdata.txt', i + ((trace+offset) * 16))
                        #print 'i = {}, val = 0x{} ({})'.format(i, line, eval('0x' + line))
                        p_backup[i-1] = eval('0x' + line)
                except (IndexError, SyntaxError) as e:
                    print "Caught Error in leakageSimulatorAESFurious: {}".format(e)
                    raise
            elif random_plaintexts:
                p_backup = get_random_bytes()
            else:
                p_backup = self.plaintext



            # Plaintext
            # if read_plaintexts:
            #     p[:len(self.plaintext)] = p_backup
            # else:
            #     p[:len(self.plaintext)] = self.plaintext

            p[:len(self.plaintext)] = p_backup
            if print_all:
                print "* Trace {:3}, Plaintext: {}".format(trace, p_backup)

            # TEST TODO
            # p[:len(self.plaintext)] = hexStringToIntArray('6bc1bee22e409f96e93d7e117393172a')

            for index in range(0, max(1, rounds_of_aes) * 16, 16):

                t[index:index+16] = linear_xor(k[index:index+16], p[index:index+16])
                s[index:index+16] = linear_sbox(t[index:index+16])

                if rounds_of_aes >= 1 and index < 144:

                    # Mix Columns
                    for i in range(index, index + 16):

                        # Shift cheating
                        j = index + (shift_rows_s[i % 16])
                        shift[i] = s[j]

                    for i in range(index, index + 16, 4):

                        # h bits
                        j = (i/4) * 3
                        h[j  ]  = shift[i  ] ^ shift[i+1]
                        h[j+1]  = h[j  ]     ^ shift[i+2]
                        h[j+2]  = h[j+1]     ^ shift[i+3]

                        # Shifty bits
                        mc[i  ] = shift[i  ] ^ shift[i+1]
                        mc[i+1] = shift[i+1] ^ shift[i+2]
                        mc[i+2] = shift[i+2] ^ shift[i+3]
                        mc[i+3] = shift[i+3] ^ shift[i  ]
                        # Normal bits
                        xt[i  ] = xtimes(mc[i  ])
                        xt[i+1] = xtimes(mc[i+1])
                        xt[i+2] = xtimes(mc[i+2])
                        xt[i+3] = xtimes(mc[i+3])
                        # cm
                        cm[i  ] = shift[i  ] ^ xt[i  ]
                        cm[i+1] = shift[i+1] ^ xt[i+1]
                        cm[i+2] = shift[i+2] ^ xt[i+2]
                        cm[i+3] = shift[i+3] ^ xt[i+3]
                        # p
                        p[i+16] = cm[i  ] ^ h[j+2]
                        p[i+17] = cm[i+1] ^ h[j+2]
                        p[i+18] = cm[i+2] ^ h[j+2]
                        p[i+19] = cm[i+3] ^ h[j+2]

                elif rounds_of_aes > 1:

                    for i in range (index, index + 16):

                        shift[i] = s[144+(shift_rows_s[i % 16])]

                        p[i+16] = shift[i] ^ k[i+16]

            if print_all:
                print "Key:         {}".format(k[:16])
                print "Plaintext:   {}".format(p[:16])
                print "Ciphertext:  {}".format(p[-16:])
                print_new_line()
                print "Key:         {}".format(get_list_as_hex_string(k[:16]))
                print "Plaintext:   {}".format(get_list_as_hex_string(p[:16]))
                print "Ciphertext:  {}".format(get_list_as_hex_string(p[-16:]))
                # print "expected: {}".format(hexStringToIntArray('3ad77bb40d7a3660a89ecaf32466ef97'))

            # TODO
            # print "Trace {}".format(trace)
            # print "Key:         {}".format(k[:16])
            # print "Plaintext:   {}".format(p[:16])
            # print "t Values:    {}".format(t[:16])
            # print_new_line()

            if real_values:
                p_ham = p
            elif hw_leakage_model:
                p_ham = linear_get_hamming_weights(p)
            else:
                p_ham = linear_get_elmo_values(p, 'p')

            p_ham[:16] = p[:16]
            dictionary['p'] [trace]     = p_ham

            # Maybe sort out here?

            if real_values:
                dictionary['k']  [trace]     = k
                dictionary['sk'] [trace]     = sk
                dictionary['xk'] [trace]     = xk
                dictionary['t']  [trace]     = t
                dictionary['s']  [trace]     = s
                dictionary['xt'] [trace]     = xt
                dictionary['cm'] [trace]     = cm
                dictionary['mc'] [trace]     = mc
                dictionary['h']  [trace]     = h
            elif hw_leakage_model:
                dictionary['k']  [trace]     = linear_get_hamming_weights(k)
                dictionary['sk'] [trace]     = linear_get_hamming_weights(sk)
                dictionary['xk'] [trace]     = linear_get_hamming_weights(xk)
                dictionary['t']  [trace]     = linear_get_hamming_weights(t)
                dictionary['s']  [trace]     = linear_get_hamming_weights(s)
                dictionary['xt'] [trace]     = linear_get_hamming_weights(xt)
                dictionary['cm'] [trace]     = linear_get_hamming_weights(cm)
                dictionary['mc'] [trace]     = linear_get_hamming_weights(mc)
                dictionary['h']  [trace]     = linear_get_hamming_weights(h)
            else:
                dictionary['k']  [trace]     = linear_get_elmo_values(k, 'k')
                dictionary['sk'] [trace]     = linear_get_elmo_values(sk, 'sk')
                dictionary['xk'] [trace]     = linear_get_elmo_values(xk, 'xk')
                dictionary['t']  [trace]     = linear_get_elmo_values(t, 't')
                dictionary['s']  [trace]     = linear_get_elmo_values(s, 's')
                dictionary['xt'] [trace]     = linear_get_elmo_values(xt, 'xt')
                dictionary['cm'] [trace]     = linear_get_elmo_values(cm, 'cm')
                dictionary['mc'] [trace]     = linear_get_elmo_values(mc, 'mc')
                dictionary['h'] [trace]      = linear_get_elmo_values(h, 'h')


        # Save dictionary
        self.simulated_dictionary = dictionary

        # Affect with Noise
        if affect_with_noise:
            self.affect_dictionary_with_noise(snr = snr, badly_leaking_nodes = badly_leaking_nodes, badly_leaking_traces = badly_leaking_traces, badly_leaking_snr = badly_leaking_snr, no_noise_nodes = no_noise_nodes, threshold = threshold, hw_leakage_model = hw_leakage_model, average_key_nodes = average_key_nodes)


    def elmo_simulation(self, snr = 32.0, traces = 1, offset = 0, read_plaintexts = True, badly_leaking_nodes = None, badly_leaking_traces = None, badly_leaking_snr = 0.1, no_noise_nodes = None, threshold = None, local_leakage = True, affect_with_noise = True, hw_leakage_model = True):

        if badly_leaking_nodes is None:
            badly_leaking_nodes = []
        if badly_leaking_traces is None:
            badly_leaking_traces = []
        if no_noise_nodes is None:
            no_noise_nodes = []

        # Open asm trace
        if local_leakage:
            if hw_leakage_model:
                f = open(PATH_TO_LOCAL_HW + 'asmtrace00001.txt')
            else:
                f = open(PATH_TO_LOCAL_ELMO + 'asmtrace00001.txt')
        else:
            f = open(PATH_TO_ELMO + 'output/asmoutput/asmtrace00001.txt')

        # Get trigger dict
        triggers = dict()
        triggers['all'] = list()
        triggers['k'] = list()
        triggers['p'] = list()
        triggers['t'] = list()
        triggers['s'] = list()
        triggers['xt'] = list()
        triggers['cm'] = list()
        triggers['mc'] = list()
        triggers['h'] = list()
        triggers['sk'] = list()
        triggers['xk'] = list()

        count = 0
        # print len(f)

        # Find and add all triggers
        for line in f:
            if string_contains(line, '0xF'):
                triggers['all'].append((count+1))
                # Sort into respective trigger file
                if string_contains(line, '0xF0'):
                    triggers['k'].append((count+1))
                elif string_contains(line, '0xF1'):
                    triggers['p'].append((count+1))
                elif string_contains(line, '0xF2'):
                    triggers['t'].append((count+1))
                elif string_contains(line, '0xF3'):
                    triggers['s'].append((count+1))
                elif string_contains(line, '0xF4'):
                    triggers['xt'].append((count+1))
                elif string_contains(line, '0xF5'):
                    triggers['cm'].append((count+1))
                elif string_contains(line, '0xF6'):
                    triggers['mc'].append((count+1))
                elif string_contains(line, '0xF7'):
                    triggers['h'].append((count+1))
                elif string_contains(line, '0xF8'):
                    triggers['sk'].append((count+1))
                elif string_contains(line, '0xF9'):
                    triggers['xk'].append((count+1))


            count += 1

        # Close the file
        f.close()

        # Print out Trigger Points
        # print "Found {} out of {} Trigger Points".format(len(triggers['all']), 1298)

        # printDictionary(triggers, getLen = True)

        dictionary = dict()
        dictionary['key']   = self.key
        dictionary['k']     = list()
        dictionary['sk']    = list()
        dictionary['xk']    = list()

        dictionary['p']     = [0] * traces
        dictionary['t']     = [0] * traces
        dictionary['s']     = [0] * traces
        dictionary['xt']    = [0] * traces
        dictionary['cm']    = [0] * traces
        dictionary['mc']    = [0] * traces
        dictionary['h']     = [0] * traces


        # Open Each Trace File
        for trace in range (traces):

            try:
                trace_path = PATH_TO_ELMO + 'output/traces/trace{}.trc'.format(pad_string_zeros(trace + 1 + offset, 5))
                if local_leakage:
                    if hw_leakage_model:
                        trace_path = PATH_TO_LOCAL_HW + 'trace{}.trc'.format(pad_string_zeros(trace + 1 + offset, 5))
                    else:
                        trace_path = PATH_TO_LOCAL_ELMO + 'trace{}.trc'.format(pad_string_zeros(trace + 1 + offset, 5))

                # print "Path to Trace File {}: {}".format(trace, trace_path)
                f = open(trace_path)
            except IOError:
                print "IOError: Can't open file named {}".format(
                    'Leakage/trace{}.trc'.format(pad_string_zeros(trace + 1 + offset, 5)))
                raise

            hw_leaks = dict()
            hw_leaks['k'] = list()
            hw_leaks['p'] = list()
            hw_leaks['t'] = list()
            hw_leaks['s'] = list()
            hw_leaks['xt'] = list()
            hw_leaks['cm'] = list()
            hw_leaks['mc'] = list()
            hw_leaks['h'] = list()
            hw_leaks['sk'] = list()
            hw_leaks['xk'] = list()

            count = 0
            t = 0

            for line in f:
                if count in triggers['all']:
                    # Put into right list
                    if count in triggers['k']:
                        hw_leaks['k'].append(strip_zero_trail(line))
                    elif count in triggers['p']:
                        hw_leaks['p'].append(strip_zero_trail(line))
                    elif count in triggers['t']:
                        hw_leaks['t'].append(strip_zero_trail(line))
                    elif count in triggers['s']:
                        hw_leaks['s'].append(strip_zero_trail(line))
                    elif count in triggers['xt']:
                        hw_leaks['xt'].append(strip_zero_trail(line))
                    elif count in triggers['cm']:
                        hw_leaks['cm'].append(strip_zero_trail(line))
                    elif count in triggers['mc']:
                        hw_leaks['mc'].append(strip_zero_trail(line))
                    elif count in triggers['h']:
                        hw_leaks['h'].append(strip_zero_trail(line))
                    elif count in triggers['sk']:
                        hw_leaks['sk'].append(strip_zero_trail(line))
                    elif count in triggers['xk']:
                        hw_leaks['xk'].append(strip_zero_trail(line))

                    t += 1
                count += 1

            # Close the file
            f.close()

            # for key_, value_ in hw_leaks.iteritems():
            #     print key_, value_, '\n\n'

            # Get Plaintext Bytes
            if read_plaintexts:
                p_backup = [0] * 16
                try:
                    for i in range(1, 17):
                        if local_leakage:
                            if hw_leakage_model:
                                line = linecache.getline(PATH_TO_LOCAL_HW + 'printdata.txt', i + ((trace+offset) * 16))
                            else:
                                line = linecache.getline(PATH_TO_LOCAL_ELMO + 'printdata.txt', i + ((trace+offset) * 16))
                        else:
                            line = linecache.getline(PATH_TO_ELMO + 'output/printdata.txt', i + ((trace+offset) * 16))
                        #print 'i = {}, val = 0x{} ({})'.format(i, line, eval('0x' + line))
                        p_backup[i-1] = eval('0x' + line)
                except (IndexError, SyntaxError) as e:
                    print "Caught Error in leakageSimulatorAES: {}".format(e)
                    raise
            else:
                p_backup = self.plaintext

            # print "'Trace' {}, read_plaintexts {}, p_backup: {}".format(trace, read_plaintexts, p_backup)

            # Change order of things that need changing
            standard_swap = ['s', 't', 'p', 'k']
            first_only_swap = ['k', 'p']

            mix_column_swap4 = ['mc','xt','p','cm']
            mix_column_swap3 = ['h']

            for swap_node in standard_swap:
                first_only = False
                if swap_node in first_only_swap: first_only = True
                hw_leaks[swap_node] = reverse_list(hw_leaks[swap_node], chunks = 16, first_only = first_only)

            # Save first 16 bytes of p
            # p_backup = hw_leaks['p'][:16]

            for swap_node in mix_column_swap4:
                # Swap!
                hw_leaks[swap_node] = reverse_list( reverse_list(hw_leaks[swap_node], chunks = 4), chunks = 16 )

            for swap_node in mix_column_swap3:
                # Swap!
                hw_leaks[swap_node] = reverse_list( reverse_list(hw_leaks[swap_node], chunks = 3), chunks = 12 )

            # Replace first 16 bytes of p
            hw_leaks['p'][:16] = p_backup

            # Take last 16 bytes of t and append to p
            last_t = hw_leaks['t'][-16:]
            hw_leaks['t'] = hw_leaks['t'][:-16]
            hw_leaks['p'].extend(last_t)

            # Now all sorted, add to dictionary

            # Keys
            if trace == 0:
                key_variables = ['k','sk','xk']
                for variable in key_variables:
                    dictionary[variable] = hw_leaks[variable]

            # Everything else
            l = ['p','t','s','xt','cm','h','mc']
            for variable in l:
                dictionary[variable][trace] = hw_leaks[variable]

            # TODO Check Categories!
            # for variable in l:
            #     leaked_val = hw_leaks[variable][16]
            #     found_category = getCategoryBF(leaked_val)
            #     supposed_category = get_category(variable)
            #     matchstr = "MATCH" if (found_category == supposed_category) else "!!! NOT A MATCH !!!"
            #     print "Variable {}, Supposedly Catgegory {}, Found Category {}. {}".format(variable, supposed_category, found_category, matchstr)

            # print "Key Leaks from Trace (No Noise):"
            # print hw_leaks['k']
            # print "sk Leaks from Trace (No Noise):"
            # print hw_leaks['sk'][:2]
            # print "xk Leaks from Trace (No Noise):"
            # print hw_leaks['xk'][:2]
            # exit(1)

        # Save dictionary
        self.simulated_dictionary = dictionary

        # Affect with Noise
        if affect_with_noise:
            self.affect_dictionary_with_noise(snr = snr, badly_leaking_nodes = badly_leaking_nodes, badly_leaking_traces = badly_leaking_traces, badly_leaking_snr = badly_leaking_snr, no_noise_nodes = no_noise_nodes, threshold = threshold, hw_leakage_model = hw_leakage_model, average_key_nodes = average_key_nodes)




    def save_simulation(self):
        save_leakage(self.simulated_dictionary, 'furious_dict')

    def affect_dictionary_with_noise(self, float snr, badly_leaking_nodes = None, badly_leaking_traces = None, badly_leaking_snr = 0.1, no_noise_nodes = None, threshold = None, hw_leakage_model = False, average_key_nodes = False):

        if badly_leaking_nodes is None:
            badly_leaking_nodes = []
        if badly_leaking_traces is None:
            badly_leaking_traces = []
        if no_noise_nodes is None:
            no_noise_nodes = []

        dictionary = self.simulated_dictionary

        # print "* Affecting with noise, HW Leakage Model: {}".format(hw_leakage_model)

        cdef int traces, trace

        # key_variables = ['k','sk','xk']
        # variables = ['p','t','s','mc','xt','cm','h']
        variables = ['k','sk','xk','p','t','s','mc','xt','cm','h']

        traces = len(dictionary['p'])

        if len(badly_leaking_traces) == 0:
            badly_leaking_traces = range(traces)

        # Non-key bytes
        for trace in range(traces):

            print_out = False

            if print_out:
                print "Leakage Simulator: Adding Noise to Trace {}".format(trace)

            for variable in variables:

                if variable in no_noise_nodes:
                    if print_out:
                        print "+++ NO NOISE for Variable {}".format(variable)
                    pass
                else:

                    if print_out:
                        print_new_line()
                        print_new_line()
                        print "*** NOISE FOR VARIABLE {}".format(variable)
                        print_new_line()
                        print_new_line()

                    badly_leaking_node = (trace in badly_leaking_traces and variable in badly_leaking_nodes)

                    if variable is 'p':
                        dictionary[variable][trace] = np.append(dictionary[variable][trace][:16][:], self.affect_array_with_noise(dictionary[variable][trace][16:][:], snr, threshold = threshold, hw_leakage_model = hw_leakage_model, category = get_category(variable), bad_leak = badly_leaking_node, badly_leaking_snr = badly_leaking_snr))
                    else:
                        # print "* Noise for Variable {} Trace {} *".format(variable, trace)
                        # print "Before:\n{}\n".format(dictionary[variable][trace])
                        dictionary[variable][trace] = self.affect_array_with_noise(dictionary[variable][trace][:], snr, threshold = threshold, hw_leakage_model = hw_leakage_model, category = get_category(variable), print_out = print_out, bad_leak = badly_leaking_node, badly_leaking_snr = badly_leaking_snr)
                        # print "After:\n{}\n\n".format(dictionary[variable][trace])

        # Key Bytes

        # for variable in key_variables:
        #     if variable in no_noise_nodes:
        #         pass
        #     else:
        #         badly_leaking_node = variable in badly_leaking_nodes
        #         # Average noise out
        #         # holder = np.zeros((traces,len(dictionary[variable][:])))
        #
        #         # Shared keys are just a single array, so average the noise out?
        #
        #         dictionary[variable] = self.affect_key_array_with_noise(dictionary[variable], snr, threshold = threshold, hw_leakage_model = hw_leakage_model, category = get_category(variable), bad_leak = badly_leaking_node, badly_leaking_snr = badly_leaking_snr, traces = traces)
        #
        #
        #         # print "\n*** Variable {}\n".format(variable)
        #         # print "Holder [{}]:\n{}\n".format(len(holder), holder)
        #         # print "Thresholded: {}".format(thresholded)
        #         # print "Dictionary [{}]:\n{}\n\n".format(len(dictionary[variable]), dictionary[variable])



        self.simulated_dictionary = dictionary

    def affect_key_array_with_noise(self, array, float snr, threshold = None, hw_leakage_model = True, category = 1, print_out = False, bad_leak = False, badly_leaking_snr = 0.1, traces = 1):
        # Average out noise and apply
        # cdef DTYPE_t i
        # print "affecting with noise:\n{}\n".format(array)
        a = np.array([self.affect_elmo_value_with_noise(i, snr, category = category, threshold = threshold, print_out = print_out, bad_leak = bad_leak, badly_leaking_snr = badly_leaking_snr, average = traces) for i in array])
        # print "\nnow:\n{}\n\n".format(a)
        return a
        # return np.array([self.affect_elmo_value_with_noise(i, snr, category = category, threshold = threshold, print_out = print_out, bad_leak = bad_leak, badly_leaking_snr = badly_leaking_snr, average = traces) for i in array])

    def affect_array_with_noise(self, array, float snr, threshold = None, hw_leakage_model = True, category = 1, print_out = False, bad_leak = False, badly_leaking_snr = 0.1):
        # cdef DTYPE_t i
        if hw_leakage_model:
            print "!!! Removed support for hw_leakage_model in array noise!"
            raise ValueError
        if print_out:
            print "Affecting Array with Noise"
        return np.array([self.affect_elmo_value_with_noise(i, snr, category = category, threshold = threshold, print_out = print_out, bad_leak = bad_leak, badly_leaking_snr = badly_leaking_snr) for i in array])

    def affect_hw_value_with_noise(self, int value, float snr, threshold = None, bad_leak = False, badly_leaking_snr = 0.1):
        cdef float noise, temp, sigma
        sigma = get_sigma(snr)
        if sigma <= 0:
            return value
        try:
            noise = np.random.normal(0, sigma, 1)[0]
            # TODO
            temp = value + noise
            if bad_leak:
                temp = temp - value + getHW(getRandomByte())
            if temp < 0: temp = 0
            if temp > 8: temp = 8
            if threshold is not None and (abs(noise) > threshold):
                return -1

            # if noise > 1:
            #     print "NOISE GREATER THAN 1: Value = {}, Noise = {}, New Value = {}".format(value, noise, temp)

            return temp
        except ValueError:
            print "Error with affect_hw_value_with_noise: value = {}, sigma = {}, threshold = {}".format(value, sigma, threshold)
            raise

    def affect_elmo_value_with_noise(self, float value, float snr, int category, threshold = None, print_out = False, bad_leak = False, badly_leaking_snr = 0.1, average = 1):
        if category > 4:
            print "ERROR: Category must be between 0 and 4, cannot be {}!".format(category)
            raise ValueError
        cdef float noise, temp, sigma
        sigma = get_sigma(snr, hw = False, category = category)
        if sigma <= 0:
            return value
        try:
            if bad_leak:
                noise = np.mean(np.random.normal(0, get_sigma(badly_leaking_snr), average))
            else:
                # TEST
                noise = np.mean(np.random.normal(0, sigma, average))
                # noisy_bits = np.random.normal(0, sigma, average)
                # noise = np.mean(noisy_bits)

            # print "NOISE Category {}: Value = {}, Noise = {}, Temp = {}, Difference = {}".format(category, value, noise, temp, abs(abs(value) - abs(noise)))

            # TODO
            temp = value + noise

            # if average > 1 and not bad_leak:
                # print "Now adding noise to a key var\nNoisyBits: {}\nMean: {}\nso {} + {} = {}".format(noisy_bits, noise, value, noise, temp)

            if print_out:
                print "--------> noise {} + value {} = {}".format(noise, value, temp)


            if threshold is not None and (abs(noise) > threshold):
                return -1

            # if noise > 1:
            #     print "NOISE GREATER THAN 1: Value = {}, Noise = {}, New Value = {}".format(value, noise, temp)

            return temp

        except ValueError:
            print "Error with affect_elmo_value_with_noise: value = {}, sigma = {}, category = {}, threshold = {}".format(value, sigma, category, threshold)
            raise

    def get_leakage_dictionary(self):
        return self.simulated_dictionary

    def load_simulation(self):
        dictionary = load_leakage('test_dict')
        return dictionary



if __name__ == "__main__":

    # TEST
    test_key = [0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75]
    test_plaintext = [105, 216, 224, 59, 183, 59, 85, 189, 197, 89, 45, 141, 228, 36, 137, 228]

    l_sim = LeakageSimulatorAESFurious()
    l_sim.fix_key(test_key)
    l_sim.fix_plaintext(test_plaintext)

    l_sim.simulate(read_plaintexts = 0, print_all = 0, affect_with_noise = False, hw_leakage_model = False)
