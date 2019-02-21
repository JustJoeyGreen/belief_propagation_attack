##############################################################################
# JOEY, IF YOU ARE READING THIS, AES FURIOUS IS THE ONE YOU SHOULD BE USING
##############################################################################

import random
import pickle
import numpy as np
import linecache
from utility import *

HW_LEAKAGE_MODEL    = True

shift_rows_xt = list([
    1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3, 16, 13, 10, 7, 4
])

shift_rows_xa1 = list([
    2, 15, 9, 5, 6, 3, 13, 9, 10, 7, 1, 13, 14, 11, 5, 1
])

shift_rows_xa2 = list([
    3, 16, 12, 6, 7, 4, 16, 10, 11, 8, 4, 14, 15, 12, 8, 2
])

shift_rows_p = list([
    20, 17, 18, 19, 24, 21, 22, 23, 28, 25, 26, 27, 32, 29, 30, 31
])

shift_rows_s = list([
    0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11
])

rcon = list([
    1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108
])



BAD_SIGMA = 32.0

class leakageSimulatorAES:

    """ Simulate Leakage for AES """
    def __init__(self, seed = 1):
        self.seed(seed)
        self.key = get_random_bytes()
        self.plaintext = get_random_bytes()
        self.simulatedDictionary = {}

    def fixKey(self, key):
        self.key = key

    def fixPlaintext(self, plaintext):
        self.plaintext = plaintext

    def randomisePlaintest(self):
        self.plaintext = get_random_bytes()

    def seed(self, seed = 0):
        random.seed(seed)
        np.random.seed(seed)

    def simulate(self, float sigma = 0.0, int traces = 1, int read_plaintexts = 1, badly_leaking_nodes=None,
                 threshold = None, int local_leakage = 1, affect_with_noise = True):

        # cdef int [176] k
        if badly_leaking_nodes is None:
            badly_leaking_nodes = []
        cdef int [40] sk
        cdef int [10] xk
        cdef int [11] rc

        # cdef int [176] p
        # cdef int [160] t
        # cdef int [160] s
        # cdef int [160] shift
        # cdef int [144] xt
        # cdef int [144] cm
        # cdef int [144] xa
        # cdef int [144] xb
        #
        cdef int i, j, trace, index
        #
        # cdef int [16] p_backup
        # cdef int [16] p_ham

        # Set up dictionary
        dictionary = {}

        # Shared Variables (keys)
        k = get_empty_int_array(size=176)
        sk = get_empty_int_array(size=40)
        xk = get_empty_int_array(size=10)
        rc = get_empty_int_array(size=11)

        # Key
        k[:len(self.key)] = self.key

        # All Round Keys
        # rc
        rc[0] = 1
        for i in range(10):
            rc[i+1] = xtimes(rc[i])

        # sk, xk, k
        for i in range(0, 160, 16):
            sk[(i / 4)] = sbox[k[i + 13]]
            sk[(i / 4) + 1] = sbox[k[i + 14]]
            sk[(i / 4) + 2] = sbox[k[i + 15]]
            sk[(i / 4) + 3] = sbox[k[i + 12]]
            # XOR with rc
            xk[(i / 16)] = sk[(i / 4)] ^ k[i]
            # First
            k[i + 16] = xk[(i / 16)] ^ rc[(i / 16)]
            # Others
            k[i + 17] = sk[(i / 4) + 1] ^ k[i + 1]
            k[i + 18] = sk[(i / 4) + 2] ^ k[i + 2]
            k[i + 19] = sk[(i / 4) + 3] ^ k[i + 3]
            # Loop the rest
            for j in range(i, i + 12):
                k[j + 20] = k[j + 4] ^ k[j + 16]

        # Save Hamming Weights to dictionary
        dictionary['key']   = self.key
        dictionary['k'] = linear_get_hamming_weights(k)
        dictionary['sk'] = linear_get_hamming_weights(sk)
        dictionary['xk'] = linear_get_hamming_weights(xk)
        dictionary['rc']    = rc




        # Everything else
        dictionary['p']     = [0] * traces
        dictionary['t']     = [0] * traces
        dictionary['s']     = [0] * traces
        dictionary['xt']    = [0] * traces
        dictionary['cm']    = [0] * traces
        dictionary['xa']    = [0] * traces
        dictionary['xb']    = [0] * traces

        for trace in range (traces):

            # Independent
            p = get_empty_int_array(176)
            t = get_empty_int_array(160)
            s = get_empty_int_array(160)
            shift = get_empty_int_array(160)
            xt = get_empty_int_array(144)
            cm = get_empty_int_array(144)
            xa = get_empty_int_array(144)
            xb = get_empty_int_array(144)

            # Get Plaintext Bytes
            if read_plaintexts:
                p_backup = [0] * 16
                try:
                    for i in range(1, 17):
                        if local_leakage:
                            if HW_LEAKAGE_MODEL:
                                line = linecache.getline(PATH_TO_LOCAL_HW + 'printdata.txt', i + (trace * 16))
                            else:
                                line = linecache.getline(PATH_TO_LOCAL_ELMO + 'printdata.txt', i + (trace * 16))
                        else:
                            line = linecache.getline(PATH_TO_ELMO + 'output/printdata.txt', i + (trace * 16))
                        #print 'i = {}, val = 0x{} ({})'.format(i, line, eval('0x' + line))
                        p_backup[i-1] = eval('0x' + line)
                except (IndexError, SyntaxError) as e:
                    print "Caught Error in leakageSimulatorAES: {}".format(e)
                    raise
            else:
                p_backup = get_random_bytes()

            print "* Trace {:3}, Plaintext: {}".format(trace, p_backup)

            # Plaintext
            if read_plaintexts:
                p[:len(self.plaintext)] = p_backup
            else:
                p[:len(self.plaintext)] = self.plaintext

            for index in range(0, 160, 16):

                t[index:index + 16] = linear_xor(k[index:index + 16], p[index:index + 16])
                s[index:index + 16] = linear_sbox(t[index:index + 16])

                if index < 144:

                    # Mix Columns
                    for i in range(index, index + 16):

                        # Shift cheating
                        j = index + (shift_rows_s[i % 16])
                        shift[i] = s[j]
                        # xtimes
                        xt[i] = xtimes(shift[i])
                        # Xor of shift and xtimes
                        cm[i] = shift[i] ^ xt[i]

                    for i in range(index, index + 16, 4):

                        # Shifty bits
                        xa[i  ] = shift[i+2] ^ shift[i+3]
                        xa[i+1] = shift[i  ] ^ shift[i+3]
                        xa[i+2] = shift[i  ] ^ shift[i+1]
                        xa[i+3] = shift[i+1] ^ shift[i+2]
                        # Normal bits
                        xb[i  ] = xa[i  ] ^ xt[i  ]
                        xb[i+1] = xa[i+1] ^ xt[i+1]
                        xb[i+2] = xa[i+2] ^ xt[i+2]
                        xb[i+3] = xa[i+3] ^ xt[i+3]
                        # Last shifty p
                        p[i+16] = xb[i  ] ^ cm[i+1]
                        p[i+17] = xb[i+1] ^ cm[i+2]
                        p[i+18] = xb[i+2] ^ cm[i+3]
                        p[i+19] = xb[i+3] ^ cm[i  ]

                else:

                    for i in range (index, index + 16):
                        shift[i] = s[144 + (shift_rows_s[i % 16])]

                        p[i + 16] = shift[i] ^ k[i + 16]

            # # Print out if necessary
            # print "*** ALL ROUND KEYS ***"
            # printListAsHex(k, chunks = 16)
            # printNewLine()
            # print "*** ALL PLAINTEXT BYTES ***"
            # printListAsHex(p, chunks = 16)

            # print "shift:  {}".format(shift[:16])
            # print "xt:     {}".format(xt[:16])
            # print "cm:     {}".format(cm[:16])
            # print "xa:     {}".format(xa[:16])
            # print "xb:     {}".format(xb[:16])
            # print "p:      {}".format(p[16:32])
            # print "last p: {}".format(p[-3:])

            # Turn all values into their Hamming Weights WITH NOISE
            # Apart from rc, and p1 - p16

            p_ham = linear_get_hamming_weights(p)

            p_ham[:16] = p[:16]
            dictionary['p'] [trace]     = p_ham

            dictionary['t'][trace] = linear_get_hamming_weights(t)
            dictionary['s'][trace] = linear_get_hamming_weights(s)
            dictionary['xt'][trace] = linear_get_hamming_weights(xt)
            dictionary['cm'][trace] = linear_get_hamming_weights(cm)
            dictionary['xa'][trace] = linear_get_hamming_weights(xa)
            dictionary['xb'][trace] = linear_get_hamming_weights(xb)


        # Save dictionary
        self.simulatedDictionary = dictionary

        # Affect with Noise
        if affect_with_noise:
            self.affectDictionaryWithNoise(sigma = sigma, badly_leaking_nodes = badly_leaking_nodes, threshold = threshold)










    def isELMOPowerModel(self, local_leakage = True):
        if local_leakage:
            if HW_LEAKAGE_MODEL:
                return False
            else:
                return True
        else:
            line = linecache.getline(PATH_TO_ELMO + 'output/traces/trace00001.trc', 1)
        if eval(line) == 1.0:
            return False
        else:
            return True

    def elmoSimulation(self, sigma = 1.0, traces = 1, read_plaintexts = True, badly_leaking_nodes=None,
                       threshold = None, local_leakage = True, affect_with_noise = True):
        # TODO: Force elmo to generate some traces

        # Open asm trace
        # f = open('Leakage/asmtrace00001.txt')
        if badly_leaking_nodes is None:
            badly_leaking_nodes = []
        if local_leakage:
            if HW_LEAKAGE_MODEL:
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
        triggers['xa'] = list()
        triggers['xb'] = list()
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
                    triggers['xa'].append((count+1))
                elif string_contains(line, '0xF7'):
                    triggers['xb'].append((count+1))
                elif string_contains(line, '0xF8'):
                    triggers['sk'].append((count+1))
                elif string_contains(line, '0xF9'):
                    triggers['xk'].append((count+1))


            count += 1

        # Close the file
        f.close()

        # Print out Trigger Points
        # print "Found {} out of {} Trigger Points".format(len(triggers['all']), 1298)

        # print_dictionary(triggers, getLen = True)

        dictionary = dict()
        dictionary['key']   = self.key
        dictionary['k']     = list()
        dictionary['sk']    = list()
        dictionary['xk']    = list()
        dictionary['rc']    = rcon[:]

        dictionary['p']     = [0] * traces
        dictionary['t']     = [0] * traces
        dictionary['s']     = [0] * traces
        dictionary['xt']    = [0] * traces
        dictionary['cm']    = [0] * traces
        dictionary['xa']    = [0] * traces
        dictionary['xb']    = [0] * traces


        # Open Each Trace File
        for trace in range (traces):

            try:
                # f = open('Leakage/trace{}.trc'.format(pad_string_zeros(trace+1, 5)))
                if local_leakage:
                    if HW_LEAKAGE_MODEL:
                        f = open(PATH_TO_LOCAL_HW + 'trace{}.trc'.format(pad_string_zeros(trace + 1, 5)))
                    else:
                        f = open(PATH_TO_LOCAL_ELMO + 'trace{}.trc'.format(pad_string_zeros(trace + 1, 5)))
                else:
                    f = open(PATH_TO_ELMO + 'output/traces/trace{}.trc'.format(pad_string_zeros(trace + 1, 5)))
            except IOError:
                print "IOError: Can't open file named {}".format(
                    'Leakage/trace{}.trc'.format(pad_string_zeros(trace, 5)))
                raise

            hw_leaks = dict()
            hw_leaks['k'] = list()
            hw_leaks['p'] = list()
            hw_leaks['t'] = list()
            hw_leaks['s'] = list()
            hw_leaks['xt'] = list()
            hw_leaks['cm'] = list()
            hw_leaks['xa'] = list()
            hw_leaks['xb'] = list()
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
                    elif count in triggers['xa']:
                        hw_leaks['xa'].append(strip_zero_trail(line))
                    elif count in triggers['xb']:
                        hw_leaks['xb'].append(strip_zero_trail(line))
                    elif count in triggers['sk']:
                        hw_leaks['sk'].append(strip_zero_trail(line))
                    elif count in triggers['xk']:
                        hw_leaks['xk'].append(strip_zero_trail(line))

                    t += 1
                count += 1

            # Close the file
            f.close()

            # Get Plaintext Bytes
            if read_plaintexts:
                p_backup = [0] * 16
                try:
                    for i in range(1, 17):
                        if local_leakage:
                            if HW_LEAKAGE_MODEL:
                                line = linecache.getline(PATH_TO_LOCAL_HW + 'printdata.txt', i + (trace * 16))
                            else:
                                line = linecache.getline(PATH_TO_LOCAL_ELMO + 'printdata.txt', i + (trace * 16))
                        else:
                            line = linecache.getline(PATH_TO_ELMO + 'output/printdata.txt', i + (trace * 16))
                        #print 'i = {}, val = 0x{} ({})'.format(i, line, eval('0x' + line))
                        p_backup[i-1] = eval('0x' + line)
                except (IndexError, SyntaxError) as e:
                    print "Caught Error in leakageSimulatorAES: {}".format(e)
                    raise
            else:
                p_backup = self.plaintext

            # Change order of things that need changing
            standard_swap = ['s', 't', 'p', 'k']
            first_only_swap = ['k', 'p']
            mixcol_swap = ['xa','xb','xt','p','cm']
            mixcol_specialcases = ['cm']

            for swapnode in standard_swap:
                first_only = False
                if swapnode in first_only_swap: first_only = True
                hw_leaks[swapnode] = reverse_list(hw_leaks[swapnode], chunks=16, first_only=first_only)

            # Save first 16 bytes of p
            # p_backup = hw_leaks['p'][:16]

            for swapnode in mixcol_swap:
                if swapnode in mixcol_specialcases:
                    # cm: rotate first
                    hw_leaks[swapnode] = rotate_list(hw_leaks[swapnode], chunks=4)
                # Swap!
                hw_leaks[swapnode] = reverse_list(reverse_list(hw_leaks[swapnode], chunks=4), chunks=16)

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
            l = ['p','t','s','xt','cm','xa','xb']
            for variable in l:
                dictionary[variable][trace] = hw_leaks[variable]

        # Save dictionary
        self.simulatedDictionary = dictionary

        # Affect with Noise
        if affect_with_noise:
            self.affectDictionaryWithNoise(sigma = sigma, badly_leaking_nodes = badly_leaking_nodes, threshold = threshold)





    def saveSimulation(self):
        save_leakage(self.simulatedDictionary, 'test_dict')

    def affectDictionaryWithNoise(self, float sigma, badly_leaking_nodes=None, threshold = None):
        if badly_leaking_nodes is None:
            badly_leaking_nodes = []
        dictionary = self.simulatedDictionary

        cdef int traces, trace

        key_variables = ['k','sk','xk']
        variables = ['p','t','s','xa','xb','xt','cm']
        traces = len(dictionary['p'])

        # Non-key bytes
        for trace in range(traces):

            for variable in variables:
                SIG = sigma
                if variable in badly_leaking_nodes: SIG = BAD_SIGMA

                if variable is 'p':
                    dictionary[variable][trace] = dictionary[variable][trace][:16][:] + self.affect_array_with_noise(
                        dictionary[variable][trace][16:][:], SIG, threshold)
                else:
                    dictionary[variable][trace] = self.affect_array_with_noise(dictionary[variable][trace][:], SIG,
                                                                               threshold)

        # Key Bytes
        for variable in key_variables:
            SIG = sigma
            if variable in badly_leaking_nodes: SIG = BAD_SIGMA
            # Average noise out
            holder = [0] * traces
            for trace in range(traces):
                holder[trace] = self.affect_array_with_noise(dictionary[variable][:], SIG, threshold)

            thresholded = False
            if threshold is not None: thresholded = True
            dictionary[variable] = arrays_average(holder, thresholded)

        self.simulatedDictionary = dictionary

    def affect_array_with_noise(self, array, float sigma, threshold = None):
        cdef int i
        if self.isELMOPowerModel():
            for i in range(len(array)):
                array[i] = self.affect_elmo_with_noise(array[i], sigma, threshold)
        else:
            for i in range(len(array)):
                array[i] = self.affect_hw_with_noise(array[i], sigma, threshold)
        return array

    def affect_elmo_with_noise(self, int value, float sigma, threshold = None):
        if sigma <= 0:
            return value
        try:
            temp = value + np.random.normal(0, sigma, 1)[0]
            return temp
        except ValueError:
            print "Error with affect_elmo_with_noise: value = {}, sigma = {}, threshold = {}".format(value, sigma,
                                                                                                     threshold)
            raise

    def affect_hw_with_noise(self, int value, float sigma, threshold = None):
        cdef float noise, temp
        if sigma <= 0:
            return value
        try:
            noise = np.random.normal(0, sigma, 1)[0]
            temp = value + noise
            if temp < 0: temp = 0
            if temp > 8: temp = 8
            if threshold is not None and (abs(value - temp) > threshold):
                return -1
            return temp
        except ValueError:
            print "Error with affect_hw_with_noise: value = {}, sigma = {}, threshold = {}".format(value, sigma,
                                                                                                   threshold)
            raise

    def getLeakageDictionary(self):
        return self.simulatedDictionary

    def loadSimulation(self):
        dictionary = loadLeakage('test_dict')
        return dictionary



if __name__ == "__main__":

    # TEST
    lSim = leakageSimulatorAES()

    # lSim.fixKey([0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a])
    # lSim.fixPlaintext([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c])
    lSim.fixKey([0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75])
    lSim.fixPlaintext([0x54, 0x77, 0x6F, 0x20, 0x4F, 0x6E, 0x65, 0x20, 0x4E, 0x69, 0x6E, 0x65, 0x20, 0x54, 0x77, 0x6F])


    t = 3

    # lSim.simulate(traces = t)
    lSim.elmoSimulation(traces = t)

    print_dictionary(lSim.getLeakageDictionary())
    # var = 'sk'
    #
    # print "*** Before: ***"
    # printNewLine()
    # print lSim.getLeakageDictionary()[var]
    # printNewLine()
    #
    # lSim.affectDictionaryWithNoise(sigma = 0.1)
    #
    # print "*** After: ***"
    # printNewLine()
    # print lSim.getLeakageDictionary()[var]
    # printNewLine()

    # dict_elmo = lSim.getLeakageDictionary()
    #
    # lSim.simulate(random_plaintexts = False, traces = 1)
    # dict_reg = lSim.getLeakageDictionary()
    #
    # print_dictionary(dict_elmo)

    # variables = ['k','xk','sk']
    #
    # for v in variables:
    #     print "*** {} ***".format(v)
    #     printNewLine()
    #     print "Regular ({}):".format(len(dict_reg[v]))
    #     print dict_reg[v]
    #     printNewLine()
    #     print "Elmo ({}):".format(len(dict_elmo[v]))
    #     print dict_elmo[v]
    #     printNewLine()
    #
    # variables = ['p','t','s']
    #
    # for v in variables:
    #     print "*** {} ***".format(v)
    #     printNewLine()
    #     print "Regular ({}):".format(len(dict_reg[v][0]))
    #     print dict_reg[v]
    #     printNewLine()
    #     print "Elmo ({}):".format(len(dict_elmo[v][0]))
    #     print dict_elmo[v]
    #     printNewLine()

    # print_dictionary(lSim.getLeakageDictionary())
