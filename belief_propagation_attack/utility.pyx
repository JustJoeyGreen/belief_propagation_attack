########################################
## utility.pyx
########################################

from sys import exit
import numpy as np
import pickle
import random
import re
import ast
import linecache
import struct
from scipy import stats
import multiprocessing
try:
    import matplotlib.pyplot as plt
except RuntimeError:
    pass
from fractions import gcd
from datetime import datetime
import cython
from cython.parallel import prange, parallel
cimport numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
import os.path
import platform
from keras.models import load_model

# Read Paths from PATH_FILE.txt
with open("PATH_FILE.txt","r") as f:
    content = f.readlines()
PATH_DICT = dict([i.replace('\n','').strip("=").split("=")
    for i in content if i[0] != '#'])

# Temp Paths: Remove if unneeded
PATH_TO_ELMO        = PATH_DICT['PATH_TO_ELMO']
PATH_TO_LOCAL_HW    = PATH_DICT['PATH_TO_LOCAL_HW']
PATH_TO_LOCAL_ELMO  = PATH_DICT['PATH_TO_LOCAL_ELMO']
PATH_TO_TRACES      = PATH_DICT['PATH_TO_TRACES']
TRACE_NAME          = PATH_DICT['TRACE_NAME']

TRACE_FILE = PATH_TO_TRACES + TRACE_NAME + '.trs'
TRACE_FOLDER = PATH_TO_TRACES + TRACE_NAME + '/'

# Find Rounds of AES from TRACE_FILE
r_of_aes = int(re.search('^.*_G(\d+)_.*', TRACE_NAME).group(1))
variable_dict = {'k': 16 + (r_of_aes * 16), 'sk': (r_of_aes * 4),
                'xk': r_of_aes, 'p': 16 + (r_of_aes * 16),
                't': 16 + (max(0, r_of_aes - 1) * 16),
                's': 16 + (max(0, r_of_aes - 1) * 16),
                'mc': min(144, r_of_aes * 16),'xt': min(144, r_of_aes * 16),
                'cm': min(144, r_of_aes * 16),'h': min(108, r_of_aes * 12)}

# FOLDERS
TIMEPOINTS_FOLDER   = TRACE_FOLDER + 'timepoints/'
POWERVALUES_FOLDER  = TRACE_FOLDER + 'powervalues/'
REALVALUES_FOLDER   = TRACE_FOLDER + 'realvalues/'
LDA_FOLDER          = TRACE_FOLDER + 'lda/'
ELASTIC_FOLDER      = TRACE_FOLDER + 'elastic/'
COEFFICIENT_FOLDER  = TRACE_FOLDER + 'coefficientarrays/'
PLAINTEXT_FOLDER    = TRACE_FOLDER + 'plaintexts/'
KEY_FOLDER          = TRACE_FOLDER + 'keys/'
TRACEDATA_FOLDER    = TRACE_FOLDER + 'tracedata/'
TENSORFLOW_FOLDER   = TRACE_FOLDER + 'tensorflow/'
TEMP_FOLDER         = TRACE_FOLDER + 'temp/'
FEATURECOLUMN_FOLDER = TENSORFLOW_FOLDER + 'featurecolumns/'
LABEL_FOLDER        = TENSORFLOW_FOLDER + 'labels/'
OUTPUT_FOLDER       = 'output/'
MODEL_FOLDER        = 'models/'
NEURAL_MODEL_FOLDER = TRACE_FOLDER + 'models/'

ALL_DIRECTORIES     = [TRACE_FOLDER, TIMEPOINTS_FOLDER, POWERVALUES_FOLDER,
                        LDA_FOLDER, ELASTIC_FOLDER, COEFFICIENT_FOLDER,
                        PLAINTEXT_FOLDER, TRACEDATA_FOLDER, REALVALUES_FOLDER,
                        TENSORFLOW_FOLDER, FEATURECOLUMN_FOLDER, LABEL_FOLDER,
                        KEY_FOLDER]

# FILES
MUSIGMA_FILEPATH            = TRACE_FOLDER + 'musigma.dict'
TRACEDATA_FILEPATH          = TRACEDATA_FOLDER + 'tracedata.npy'
TRACEDATA_EXTRA_FILEPATH    = TRACEDATA_FOLDER + 'extratracedata.npy'
PLAINTEXT_FILEPATH          = PLAINTEXT_FOLDER + 'plaintexts.npy'
PLAINTEXT_EXTRA_FILEPATH    = PLAINTEXT_FOLDER + 'extraplaintexts.npy'
KEY_FILEPATH                = KEY_FOLDER + 'keys.npy'
KEY_EXTRA_FILEPATH          = KEY_FOLDER + 'extrakeys.npy'
METADATA_FILEPATH           = TRACE_FOLDER + 'metadata'

OUTPUT_FILE_PREFIX          = OUTPUT_FOLDER + 'ranks'


CNN_ASCAD_FILEPATH          = (NEURAL_MODEL_FOLDER +
    'cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5')
MLP_ASCAD_FILEPATH          = (NEURAL_MODEL_FOLDER +
    'mlp_best_ascad_desync0_node200_layernb6_epochs200_' +
    'classes256_batchsize100.h5')

TESTING_MODELS_CSV          = TRACE_FOLDER + 'testing_models.csv'

# Ignore numpy warnings
old_settings = np.seterr(all='ignore')

# Various Variables
SIZE_OF_ARRAY       = 256
RANKING_START       = 500
NORMALISE_MAX       = 30000000000
NORMALISE_MIN       = 1e-16
DIVIDED_MAX         = 10000
DIVIDED_THRESH      = 100
ALLOW_EMPTY         = True
NO_WIPE             = False
SQRT2               = np.sqrt(2)
MAX_SHIFT           = 10

HAMMING_WEIGHT_ELMO_OVERRIDE = False
PRELOAD_NEURAL_NETWORKS = False

sbox = list([
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

inv_sbox = list([
        0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
        0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
        0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
        0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
        0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
        0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
        0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
        0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
        0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
        0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
        0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
        0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
        0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
        0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
        0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
])

hamming_weights = list([
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
])

if HAMMING_WEIGHT_ELMO_OVERRIDE:
    operand2_leaks = list([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])
else:
    operand2_leaks = list([
        [0.00032264103873057862,  0.00162482308659243598, 0.00055496385532201006, 0.00024241858302044848, 0.00245695178220806413],
        [0.00011569732117225246,  0.00096274087108277299, 0.00063881305781466702, 0.00064992107996750205, 0.00203676186148181492],
        [0.00018557119345498249, -0.00074214692573633368, 0.00052642048984816441, 0.00042728881147138179, 0.00221489069370904785],
        [0.00018877092618485187, -0.00089728751983009464, 0.00031294379562479602, 0.00047041451387940740, 0.00230451479282279799],
        [0.00019477183420417587, -0.00093607225733040767, 0.00055134895620813026, 0.00038894099478035997, 0.00155828651238453398],
        [0.00010315879491062759, -0.00054269665404572248, 0.00055068611255488825, 0.00020954220342829575, 0.00124321156960919112],
        [0.00020078519438770999, -0.00063850621376071697, 0.00007631835022128497, 0.00005405075678367042, 0.00045426844756310411],
        [0.00004496283356609092, -0.00033136170408038602, 0.00001206797630264493, 0.00010569079443697714, 0.00068799652663467203]
    ])

operand2_sigmas = np.var(np.transpose(operand2_leaks), axis=1)

################ TESTING FUNCTIONS PLEASE IGNORE ################

def make_8(v):
    return v[:8]

def make_256(v):
    return v + ([0] * (256 - len(v)))

def my_xor(v1, v2):
    return make_8(array_xor(make_256(v1), make_256(v2)))

def my_xor2(v1, v2, v3):
    return make_8(array_xor2(make_256(v1), make_256(v2), make_256(v3)))

def my_mult(v1, v2):
    return make_8(array_multiply(make_256(v1), make_256(v2)))

def my_mult2(v1, v2, v3):
    return make_8(array_multiply(make_256(v1), make_256(v2), make_256(v3)))

#################################################################

def string_contains(string, substr):
    """Returns true if substr is a substring of string"""
    return substr in string

def check_file_exists(filename):
    """Checks if filename is a file using os package"""
    return os.path.isfile(filename)

def get_elmo_leakage_value(value, instruction_category):
    """Return the elmo leakage value given a value and an instruction category,
    see ELMO for more details"""
    if instruction_category < 1 or instruction_category > 5:
        print "ERROR: instruction_category must be with 1 and 5 (currently \
            {})".format(instruction_category)
        raise ValueError
    bin_rep = pad_string_zeros(bin(value).replace('0b',''), 8)
    # print "Value {} -> {}".format(valu
    total = 0
    for i, bit in enumerate(bin_rep):
        if bit == '1':
            # Categories mapped 1 to 5 -> 0 to 4
            total += operand2_leaks[7-i][instruction_category-1]
    return total

def clear_screen(n = 50):
    """Clears the screen, make the terminal nice and neat"""
    s1 = "* " * 80
    s2 = " " + s1
    for i in range (n):
        print s1
        print s2
    print_new_line()

def pad_string_zeros(string, pad_length = 3):
    """Pad the string with leading zeros, used for compatibility
    with variable node names"""
    if type(string) != str:
        string = str(string)
    while len(string) < pad_length:
        string = '0' + string
    return string

def xtimes(int x, int b = 8):
    """Returns xtimes(x) in base b"""
    if (x & (1<<(b-1))) != 0:
        if b == 8:
            return ((x & ~(1<<(b-1))) << 1) ^ 27
        else:
            return ((x & ~(1<<(b-1))) << 1) ^ 5
    return x << 1

def inv_xtimes(int x, int b = 8):
    """Returns inverse xtimes(x) in base b"""
    if x % 2:
        if b == 8:
            return ((x ^ 27) >> 1) | (1 << 7)
        else:
            return ((x ^ 5) >> 1) | (1 << b-1)

    return x >> 1


def get_hw(n):
    """Returns the Hamming Weight of a value"""
    return bin(n).count("1")

get_hw_of_vector = np.vectorize(get_hw)

def big_endian_to_int(byte_array, number_of_bytes, signed_int = False,
    float_coding = False):
    """Converts a big endian byte array into an int"""
    int_val = 0
    for b in range(number_of_bytes-1, -1, -1):
        # # TODO
        # print len(byte_array), b, number_of_bytes
        int_val += byte_array[b]
        if b != 0:
            int_val <<= 8
    if signed_int and int_val >= (2 ** ((8*number_of_bytes) - 1)):
        int_val -= (2 ** (8 * number_of_bytes))
    if float_coding:
        s = struct.pack('>l', int_val)
        return struct.unpack('>f', s)[0]
    return int_val

def my_mod(int i, int n = 16):
    """Returns my mod function, tweaked over standard mod"""
    return ((i-1)%n)+1

# noinspection PyUnresolvedReferences
def gaussian_probability_density(float x, float mu, float sigma):
    """Returns probability density function using x, mean mu, and standard
    deviation sigma"""
    cdef float s2, r, t, q, w, a, b, out
    if sigma > 0:
        s2 = sigma**2
        r = (x - mu)**2
        t = (-1*r)/(2 * s2)
        if t < -275:
            return 0.0
        q = 2*np.pi*s2
        w = np.sqrt(q)
        a = 1/w
        b = np.exp(t)
        out = a * b
        return out
    else:
        if x == mu:
            return 1.0
        else:
            return 0.0

def strip_off_trace(var):
    """Strips off the trace part of the variable string"""
    return var.split('-')[0]

def split_eval(string, split_val):
    """Splits and evals a string into a tuple"""
    a, b = string.split(split_val)
    return eval(a), eval(b)

def get_detrimental_traces(v):
    """Return the traces that perform detrimentally from a list of ranks"""
    # Assume first trace is good
    out = list()
    for i, val in enumerate(v):
        if i > 0 and val > v[i-1]:
            out.append(i)
    return out

def smallest_power_of_two(int x):
    """Returns the smallest power of two beneath value x"""
    for i in range(1000):
        if (2**-i) < x:
            return -i
    print "ERROR: Could not find lower power of two for {}\n".format(x)
    raise ValueError

def get_round_of_variable(var):
    """Returns round from variable name"""
    var_name, var_number, var_trace = split_variable_name(var)
    if var_name != 'h':
        return get_round_from_number(var_number)
    else:
        print "TODO: Handle getting round of h variables"
        exit(1)

def get_round_from_number(int var_number):
    """Returns round from number"""
    return ((var_number - 1) // 16) + 1

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
        if PRELOAD_NEURAL_NETWORKS:
            neural_network = neural_network_dict[var]
        else:
            neural_network = load_sca_model('{}{}_mlp5_nodes200_window700_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, var))

        probabilities = normalise_array(
            neural_network.predict(np.resize(
                power_value, (1, power_value.size)))[0])
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

def get_plaintext_array(int value):
    v = get_zeros_array()
    v[value] = 1.0
    return v

def get_no_knowledge_array(int size = 256):
    return np.full(size, 1.0/size, dtype=np.float32)

def get_empty_array(int size = 256):
    return np.empty(size, dtype=np.float32)

def get_zeros_array(int size = 256):
    return np.zeros(size, dtype=np.float32)

def get_filled_array(float val, int size = 256):
    return np.full(size, val, dtype=np.float32)

# noinspection PyUnresolvedReferences
def get_empty_int_array(int size = 256):
    return np.zeros(size, dtype=np.int)

def is_no_knowledge_array(np.ndarray v, int size = 256):
    return np.unique(v).size == 1 and np.unique(v) == 1.0/size

def is_plaintext_array(np.ndarray v, int size = 256):
    return np.unique(v).size == 2 and np.max(v) == 1.0 and np.min(v) == 0.0

def is_zeros_array(np.ndarray v):
    return np.unique(v).size == 1 and np.unique(v) == 0.0

def array_median(np.ndarray v):
    return np.median(v)

def array_variance(np.ndarray v):
    return np.var(v)

def array_min(np.ndarray v):
    if is_zeros_array(v):
        return 0
    else:
        return np.min(v[v>0])

def array_max(np.ndarray v):
    if is_zeros_array(v):
        return 0
    else:
        try:
            return np.max(v[v>0])
        except ValueError:
            print "!!! Value Error in array_max, v:\n{}\n".format(v)
            raise ValueError

def array_divide_float(np.ndarray v, double x):
    if x > 0.0:
        return v / x
    print "array_divide_float, trying to divide by {}:\n{}\n".format(x, v)
    raise ValueError

def array_add(np.ndarray v1, np.ndarray v2):
    return v1 + v2

def array_subtract(np.ndarray v1, np.ndarray v2):
    return v1 - v2

def plug_zeros(np.ndarray v):
    cdef np.ndarray v_out = np.copy(v)
    v_out[v_out==0] = NORMALISE_MIN
    return v_out

def all_positive(np.ndarray v):
    return np.absolute(v)

def wipe_low(np.ndarray v, float x, keep_above_zero = True):
    cdef np.ndarray v_out = np.copy(v)
    if keep_above_zero:
        v_out[v_out < x] = NORMALISE_MIN
    else:
        v_out[v_out < x] = 0
    return v_out

def wipe_lowest(np.ndarray v):
    return wipe_low(v, array_min(v))

def max_index(np.ndarray v):
    return np.argmax(v)

def my_hw_array(x, bits=3, norm=True):
    hw = get_hw(x)
    a = [0] * (2**bits)
    c = 0
    for i in range(2**bits):
        if get_hw(i) == hw:
            c += 1
    v = 1.0 / c
    for i in range(2**bits):
        if get_hw(i) == hw:
            a[i] = v if norm else 1
    return a

def make_sum_to_one(v):
    s = sum(v)
    for i, val in enumerate(v):
        v[i] = val / (s + 0.0)
    return v

def replace_zeros(v, x=0.9):
    for i, val in enumerate(v):
        if val == 0:
            v[i] = x
    return v

def get_hamming_weight_array(float hw, float sigma = 0.1):

    # Handle HW < 0 (no knowledge array)
    if hw < 0:
        return get_no_knowledge_array()

    # Set up array
    cdef int i
    cdef np.ndarray v = get_no_knowledge_array()

    for i in range(len(v)):
        v[i] = gaussian_probability_density(hamming_weights[i], hw, sigma)

    # Wipe all that are lower than the min
    v = wipe_low(v, NORMALISE_MIN)

    # Normalise
    return normalise_array(v)

def linear_get_hamming_weights(np.ndarray v):
    return np.array([get_hw(i) for i in v])

def linear_get_elmo_values(np.ndarray v, name):
    return np.array([get_elmo_leakage_value(val, get_category(name + pad_string_zeros(i+1))) for i, val in enumerate(v)])

def get_power_modelled_key_values(np.ndarray v, elmo = False):
    if elmo:
        return np.array([get_elmo_leakage_value(i, 4) for i in v])
    else:
        return linear_get_hamming_weights(v)

def get_sigma(snr, hw = True, category = 1):
    if hw:
        return np.sqrt(2.0/snr)
    return np.sqrt(operand2_sigmas[category-1]/snr)

def gcd_of_array(v):
    cdef int g, i
    g = v[0]
    if type(g) == np.float64:
        return -1
    for i in v:
        g = gcd(g,i)
    return g

def normalise_2d_array(np.ndarray v):
    cdef np.ndarray v_copy = np.copy(v)
    cdef int i
    for i in range(len(v_copy)):
        v_copy[i] = normalise_array(v_copy[i])
    return v_copy

def roll_and_pad(np.ndarray v, int x):
    cdef np.ndarray v_copy = np.roll(v, x)
    if x > 0:
        v_copy[:x] = v_copy[x]
    elif x < 0:
        v_copy[x:] = v_copy[x-1]
    return v_copy


@cython.boundscheck(False)
@cython.wraparound(False)
def normalise_array(v):
    cdef np.ndarray divided
    cdef np.ndarray norm
    # First, check for empty array
    if is_zeros_array(v):
        if ALLOW_EMPTY:
            return np.copy(v)
        else:
            return get_no_knowledge_array()
    # Next, check the case where all values are below NORMALISE_MIN
    if array_max(v) < NORMALISE_MIN:
        # Make sure at least some of them above NORMALISE_MIN!
        divided = array_divide_float(v, array_max(v))
    else:
        divided = np.copy(v)
    # Then, remove small values
    if NO_WIPE:
        wiped = divided
    else:
        wiped = wipe_low(divided, NORMALISE_MIN)
    # Check if already normalised
    if np.sum(wiped) == 1.0:
        # Already normalised
        return wiped
    # Else, normalise by dividing all by sum
    norm = array_divide_float(wiped, np.sum(wiped))

    if len(norm[norm < 0]) > 0 or len(norm[norm > 2]) > 0:
        print "Negative / Over 2 here!"
        print "v:\n{}\n".format(v)
        print "Max: {}, Min: {}".format(np.max(v), np.min(v))
        print "divided:\n{}\n".format(divided)
        print "Max: {}, Min: {}".format(np.max(divided), np.min(divided))
        print "wiped:\n{}\n".format(wiped)
        print "Max: {}, Min: {}".format(np.max(wiped), np.min(wiped))
        print "norm:\n{}\n".format(norm)
        print "Max: {}, Min: {}".format(np.max(norm), np.min(norm))
        raise ValueError

    return norm

def convert_to_probability_distribution(np.ndarray v):
    return array_divide_float(v, np.sum(v))

def array_range(v):
    return array_max(v) - array_min(v)

def message_inv_sbox(v):
    if type(v) == type(tuple()):
        return inv_sbox[v[0]], v[1]
    else:
        return array_inv_sbox(v)

def array_inv_sbox(np.ndarray v):
    cdef np.ndarray v_sbox = get_empty_array()
    cdef int i
    for i in range(256):
        v_sbox[i] = v[sbox[i]]
    return v_sbox

def linear_sbox(v):
    cdef int i
    cdef np.ndarray v_sbox = np.zeros(len(v))
    for i in range(len(v)):
        v_sbox[i] = sbox[v[i]]
    return v_sbox

def message_sbox(v):
    if type(v) == type(tuple()):
        return sbox[v[0]], v[1]
    else:
        return array_sbox(v)

def array_sbox(np.ndarray v):
    cdef np.ndarray v_sbox = get_empty_array()
    cdef int i
    for i in range(256):
        v_sbox[i] = v[inv_sbox[i]]
    return v_sbox

def message_inv_xtimes(v):
    if type(v) == type(tuple()):
        return inv_xtimes(v[0]), v[1]
    else:
        return array_inv_xtimes(v)

def array_inv_xtimes(v):
    cdef np.ndarray v_xtimes = get_empty_array()
    cdef int i
    for i in range(256):
        v_xtimes[i] = v[xtimes(i)]
    return v_xtimes

def linear_xtimes(v):
    cdef np.ndarray v_xtimes = np.zeros(len(v))
    for i in range(len(v)):
        v_xtimes[i] = xtimes(v[i])
    return v_xtimes

def message_xtimes(v):
    if type(v) == type(tuple()):
        return xtimes(v[0]), v[1]
    else:
        return array_xtimes(v)

def array_xtimes(v):
    cdef np.ndarray v_xtimes = get_empty_array()
    cdef int i
    for i in range(256):
        v_xtimes[i] = v[inv_xtimes(i)]
    return v_xtimes

def message_xor(v1, v2):
    # return v1
    # if type(v1) == type(tuple()) or type(v2) == type(tuple()):
    #     # Tuple XOR here
    #     if type(v1) == type(tuple()):
    #         v = v2
    #         tp = v1
    #     else:
    #         v = v1
    #         tp = v2
    #     if tp[1] == 0:
    #         return get_zeros_array()
    #     else:
    #         # Permutate with XOR
    #         return array_xor_permutate(v, tp[0])
    # else:
    #     return array_xor(v1, v2)
    return array_xor(v1, v2)

def array_xor8(v1, v2, norm = True):
    v_xor = [0] * len(v1)
    for i, val in enumerate(zip(v1,v2)):
        v_xor[0] += val[0] * val[1]
    for i in range(len(v1)):
        for j in range(i+1, len(v1)):
            v_xor[i^j] += (v1[i] * v2[j]) + (v1[j] * v2[i])
    if norm:
        return make_sum_to_one(v_xor)
    return v_xor

@cython.boundscheck(False)
@cython.wraparound(False)
def array_xor_(double [:] v1, double [:] v2):
    # METHOD 3
    cdef double [:] v_xor = np.zeros(256)
    cdef int i, j
    for i, val in enumerate(zip(v1,v2)):
        v_xor[0] += val[0] * val[1]
    for i in range(256):
        for j in range(i+1, 256):
            v_xor[i^j] += (v1[i] * v2[j]) + (v1[j] * v2[i])
    return v_xor

@cython.boundscheck(False)
@cython.wraparound(False)
# def array_xor(object [DTYPE_t, ndim=1] v1, object [DTYPE_t, ndim=1] v2):
def array_xor(np.ndarray [DTYPE_t, ndim=1] v1, np.ndarray [DTYPE_t, ndim=1] v2):
    # METHOD 3
    cdef np.ndarray [DTYPE_t, ndim=1] v_xor = get_zeros_array()
    cdef int i, j
    for i, val in enumerate(zip(v1,v2)):
        v_xor[0] += val[0] * val[1]
    for i in range(256):
        for j in range(i+1, 256):
            v_xor[i^j] += (v1[i] * v2[j]) + (v1[j] * v2[i])
    return normalise_array(v_xor)
    # return v1

@cython.boundscheck(False)
@cython.wraparound(False)
def array_xor2(float [:] v1, float [:] v2):
    # METHOD 3
    cdef float [:] v_xor = get_zeros_array()
    cdef int i, j
    for i, val in enumerate(zip(v1,v2)):
        v_xor[0] += val[0] * val[1]
    with nogil, parallel(num_threads=4):
        for i in prange(256, schedule='dynamic'):
            for j in range(i+1, 256):
                v_xor[i^j] += (v1[i] * v2[j]) + (v1[j] * v2[i])
    return normalise_array(np.array(v_xor))

def euclidian_distance(v1, v2):
    cdef int i
    cdef float v_sum
    v_sum = 0.0
    for i, val in enumerate(zip(v1,v2)):
        v_sum += (val[0] - val[1]) ** 2
    return v_sum / 256.0

def normal_distance(v1, v2):
    cdef int i
    cdef float v_max, current
    v_max = 0
    for i, val in enumerate(zip(v1,v2)):
        current = abs(val[0] - val[1])
        if current > v_max:
            v_max = current
    return v_max

def array_xor_permutate(v, x):
    cdef np.ndarray v_xor = get_empty_array()
    cdef int i
    for i in range(256):
        v_xor[i] = v[i^x]
    return v_xor

def linear_xor(np.ndarray v1, np.ndarray v2):
    if len(v1) != len(v2):
        print "Error in linear_xor; v1 and v2 must be same size! v1: {}, v2: {}".format(v1, v2)
        raise AssertionError
    return v1 ^ v2

def message_xor_constant(v, x):
    # Handle Xor Constant (used in Key Scheduling to replace rc)
    if type(v) == type(tuple()):
        return v[0] ^ x, v[1]
    return array_xor_permutate(v, x)


def array_multiply_constant(np.ndarray v1, int c):
    cdef np.ndarray v_mult = get_empty_array()
    cdef int i
    for i in range(256):
        v_mult[i] = v1[i] * c
    return v_mult

def array_cast_to_int(np.ndarray v):
    cdef np.ndarray v_int = get_empty_array()
    cdef int i
    for i in range(256):
        v_int[i] = int(round(v[i]))
    return v_int


def array_multiply(np.ndarray v1, np.ndarray v2):
    cdef np.ndarray v_mult = get_empty_array()
    cdef int i
    for i in range(256):
        v_mult[i] = v1[i] * v2[i]
    # Normalise
    return normalise_array(v_mult)

def array_multiply8(v1, v2, norm = True):
    v_mult = [0] * 8
    for i in range(8):
        v_mult[i] = v1[i] * v2[i]
    if norm:
        return make_sum_to_one(v_mult)
    return v_mult

def array_multiply2(v1, v2, v3):
    cdef float [256] v_mult
    cdef int i
    for i in range(256):
        v_mult[i] = v1[i] * v2[i] * v3[i]
    # Normalise
    v_mult = normalise_array(v_mult)
    return v_mult

def array_2d_multiply(np.ndarray vv1, np.ndarray vv2):
    # Each list is a list of arrays
    return normalise_2d_array(vv1 * vv2)

def array_2d_add(np.ndarray vv1, np.ndarray vv2):
    # Each list is a list of arrays
    return vv1 + vv2

def array_swap(v, tup_list):
    temp = v[:]
    for t in tup_list:
        temp[t[0]], temp[t[1]] = temp[t[1]], temp[t[0]]
    return temp

def string_contains_any(string, lst):
    for l in lst:
        if string_contains(string, l):
            return True
    return False

def get_variable_name(string):
    # sk102-0 -> sk
    return re.search(r'^[a-z]+', string).group(0)

def get_variable_number(string):
    # sk102-0 -> 102
    try:
        return int(re.search(r'\d{3}', string).group(0))
    except AttributeError:
        return None

def get_variable_trace(string):
    # sk102-0 -> 0
    try:
        return int((re.search(r'-\d+$', string).group(0)).replace('-',''))
    except AttributeError:
        # Key or None
        if string_contains(string, '-K'):
            return 0
        else:
            return None

def split_factor_name(string):
    sp = string.split('_')
    return sp[1], sp[2]

def split_variable_name(string):
    return get_variable_name(string), get_variable_number(string), get_variable_trace(string)

def string_starts_with(string, substr):
    return string[:len(substr)] == substr

def string_ends_with(string, substr):
    return string[-len(substr):] == substr

def is_xor_node(string):
    return string_contains(string, '_Xor_')

def is_xor_constant_node(string):
    return string_contains(string, '_XorConstant_')

def is_sbox_node(string):
    return string_contains(string, '_Sbox_')

def is_xtimes_node(string):
    return string_contains(string, '_Xtimes_')

def is_xor_xtimes_node(string):
    return string_contains(string, '_XorXtimes_')

def print_new_line():
    print ""

def get_average(l):
    return sum(l) / float(len(l))

def get_mode(l):
    return max(set(l), key=l.count)

def bit_length(x):
    return int(round(x)).bit_length() - 1

def pad_string(string, length = 3):
    out = string
    if len(out) < length:
        out += " " * (length - len(out))
    return out

def print_length(string, length):
    print pad_string(string, length)

def print_length_append(str1, str2, length):
    print pad_string(str1, length), str2

def print_dictionary(my_dict, get_len = False):
    if get_len:
        for k, v in my_dict.iteritems():
            print "{} ({}):\n{}\n".format(k, len(v), v)
    else:
        for k, v in my_dict.iteritems():
            print "{}:\n{}\n".format(k, v)

def print_list_of_lists(lst):
    for i in range(len(lst)):
        print "i = {}:\n{}\n".format(i, lst[i])

def print_list_as_hex_list(lst, chunks = None):
    if chunks is None:
        out = "["
        for i in range(len(lst)):
            out += hex(lst[i]) + ", "
        print out[:-2] + "]"
    else:
        out = ""
        for i in range(len(lst)):
            if (i % chunks) == 0:
                out = "i = {}: [".format(i // chunks)
            out += hex(lst[i]) + ", "
            if ((i+1) % chunks) == 0:
                print out[:-2] + "]"
                out = ""

def get_list_as_hex_string(lst, little_endian = False):
    hexstr = ""

    if little_endian:
        for i in reversed(lst):
            hexstr += format(int(i), '02x')
    else:
        for i in lst:
            hexstr += format(int(i), '02x')
    return hexstr

def byte_list_to_int(np.ndarray v, little_endian = False):
    out = 0
    if little_endian:
        for i, val in enumerate(np.flip(v, 0)):
            out += val << (8 * i)
    else:
        for i, val in enumerate(v):
            out += val << (8 * i)
    return out

def get_repeating_pattern(lst):
    for size in range(2, (len(lst) / 2) + 1):
        pattern = lst[:size]
        valid = True
        for i in range(size, (len(lst) - (len(lst) % size)), size):
            if pattern != lst[i:i+size]:
                valid = False
                break
        if valid:
            return pattern
    return None

def has_converged(l, threshold = 20):
    if len(l) < 2:
        return False
    last_val = l[-1]
    for i in range(1, threshold):
        try:
            if l[-i] != last_val:
                return False
        except IndexError:
            pass
    return True

def get_round_converged(l):
    last_val = l[-1]
    i = 1
    while i <= len(l) and l[-i] == last_val:
        i += 1
    return len(l) - i + 1

def save_leakage(obj, name):
    save_object(obj, 'Leakage/' + name)

def load_leakage(name):
    return load_object('Leakage/' + name)

def save_meta(tup):
    save_object(tup, METADATA_FILEPATH)

def load_meta():
    # returns (profile_traces, attack_traces, samples, np.float32 if float_coding else np.int16)
    return load_object(METADATA_FILEPATH)

def save_object(obj, name, output=False):
    with open(('output/' if output else '') + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(name, output=False):
    with open(('output/' if output else '') + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_random_byte():
    return random.randint(0,255)

def get_random_numbers(int v_size = 1, int v_min = 0, int v_max = 10):
    return np.random.randint(v_min, v_max, size=v_size)

def get_random_bytes(number_bytes = 16, seed = None):
    if seed is not None:
        random.seed(seed)
    a = np.zeros(number_bytes)
    for i in range(number_bytes):
        a[i] = get_random_byte()
    return a

def arrays_average(np.ndarray v, int thresholded = False):
    # return a list that contains the elementwise average
    return np.mean(v, axis=0)

def arrays_median(np.ndarray v):
    # return a list that contains the elementwise average
    return np.median(v, axis=0)

def arrays_trim_mean(np.ndarray v, threshold = 0.3):
    return stats.trim_mean(v, threshold, axis=0)

def load_sca_model(model_file):
    check_file_exists(model_file)
    try:
        model = load_model(model_file)
    except:
        print("Error: can't load Keras model file '%s'" % model_file)
        exit(-1)
    return model

def find_values_in_list(lst, val):
    out = list()
    for i, item in enumerate(lst):
        if item == val:
            out.append(i)
    return out

def find_value_in_list(lst, val):
    for i, item in enumerate(lst):
        if item == val:
            return i
    return -1

def get_key_rank_index_list(lst_in):
    # Return list of sorted most likely values
    # e.g. [11, 4, 8, 3, 8, 0] => [0, [2, 4], 1, 3, 5]
    lst = lst_in[:] # get copy of input
    out = list()
    while len(lst) > 0:
        max_val = max(lst)
        max_indexes = find_values_in_list(lst_in, max_val)
        for i in range(len(max_indexes)):
            lst.remove(max_val)
        if len(max_indexes) == 1: max_indexes = max_indexes[0]
        out.append(max_indexes)

    return out

def get_rank_from_index_list(lst, index):
    # Each item in lst could be int or lst
    for i in range(len(lst)):
        if (type(lst[i]) == int and lst[i] == index) or (type(lst[i]) == list and index in lst[i]):
            return i + 1
    print "ERROR: Could not find index {} in list {}".format(index, lst)
    raise IndexError


def list_to_file_string(lst):
    output = ""
    lst = sorted(lst)
    for item in lst:
        output += str(item) + "-"
    return output[:-1]

def get_all_variables_that_match(variables, target_vars):
    # print "Getting Variables that match!"
    # print "Variables: {}\n\nTargets: {}\n\n".format(variables, target_vars)
    outlst = []
    for target_var in target_vars:
        matched = get_variables_that_match(variables, target_var)
        # print "Target {}, Matched {}".format(target_var, matched)
        outlst += matched
    return outlst


def get_variables_that_match(variables, target_var):
    if target_var is None:
        return []
    matching_variables = []
    target_name, target_number, target_trace = split_variable_name(target_var)
    # All variables in graph
    for var in variables:
        # Split name
        var_name, var_number, var_trace = split_variable_name(var)
        # print target_name, target_number, target_trace, var_name, var_number, var_trace
        # Check for match
        if (target_number is not None and target_trace is not None and target_name == var_name and target_number == var_number and target_trace == var_trace) or (target_number is not None and target_trace is None and target_name == var_name and target_number == var_number) or (target_number is None and target_trace is None and target_name == var_name):
            matching_variables.append(var)
    return matching_variables

def get_parent_factor_node(factors, target_var):
    for fac in factors:
        if string_ends_with(fac, target_var):
            return fac
    return False

def strip_zero_trail(n):
    return float(n.rstrip())

def strip_zero_trail_in_list(lst):
    if len(lst) <= 1:
        return lst
    for i, val in enumerate(reverse_list(lst)):
        if val != 0:
            return lst[:-i]
    return []

def get_index_of_sublist(lst, sublst):
    for i in range(len(lst) - len(sublst) + 1):
        if (lst[i:i+len(sublst)] == sublst).all():
            return i
    return -1

def reverse_list(lst, chunks = None, first_only = False):
    if chunks is None:
        return list(reversed(lst))
    out = [0] * len(lst)
    if first_only:
        out = lst[:]
        out[0:chunks] = reverse_list(lst[0:chunks])
    else:
        for i in range(0, len(lst), chunks):
            out[i:i+chunks] = reverse_list(lst[i:i+chunks])
    return out

def rotate_list(lst, n = 1, chunks = None):
    if chunks is None:
        return lst[-n:] + lst[:-n]
    out = [0] * len(lst)
    for i in range(0, len(lst), chunks):
        out[i:i+chunks] = rotate_list(lst[i:i+chunks])
    return out

def get_top_values(dist, n=5):
    npdist = np.array(dist)
    return list(npdist.argsort()[-n:][::-1])

def get_template(template_id):
    f = open("Templates/template_{}.txt".format(template_id),"r")
    line = f.read()
    try:
        template = ast.literal_eval(line)
    except ValueError:
        f.close()
        print "*** ValueError evaluating line: {}".format(line)
        raise
    f.close()
    return template

def template_match(var, target_value, snr, bits = 8, normalise = True):
    # myvar_list = ['k029-0', 'k030-0', 'k031-0']
    myvar_list = []
    probdist = get_empty_array()
    category = get_category(var)
    # std = max(get_sigma(snr, hw = False, category = category), 0.0001)
    std = get_sigma(snr, hw = False, category = category)

    if var in myvar_list:
        print "Template Matching Here! Variable {}, Target Value {} (Category {}, SNR {}, std = {})".format(var, target_value, category, snr, std)

    for i in range(2**bits):
        mean = get_elmo_leakage_value(i, category)
        probdist[i] = gaussian_probability_density(target_value, mean, std)
        if var in myvar_list:
            print "Mean for Value {}: {} (probability = {})".format(i, mean, probdist[i])

    if var in myvar_list:
        print "MOST LIKELY VALUE: {}".format(max_index(probdist))

    if is_zeros_array(probdist):
        return get_no_knowledge_array()
    if normalise:
        return normalise_array(probdist)
    else:
        return probdist

def get_rank_list_from_prob_dist(np.ndarray probdist):
    temp = probdist.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(probdist))
    return ranks

def get_rank_from_prob_dist(np.ndarray probdist, int index, worst_case = True):
    if worst_case:
        return np.size(np.where(probdist >= probdist[index]))
    else:
        return np.size(np.where(probdist > probdist[index])) + 1

def get_category(variable):
    # category_dict = {'h':1, 'mc':1, 'cm':1, 'xk':1, 't':3, 's':3, 'p':3, 'xt':4, 'sk':4, 'k':[4,3]}
    category_dict = {'h':1, 'mc':1, 'cm':1, 'xk':1, 't':3, 's':3, 'p':3, 'xt':4, 'sk':4, 'k':4}
    var_name, var_number, _ = split_variable_name(variable)
    # if var_name == 'k':
    #     if var_number <= 16:
    #         return category_dict[var_name][0]
    #     else:
    #         return category_dict[var_name][1]
    # else:
    #     return category_dict[var_name]
    return category_dict[var_name]


def brute_force_elmo_value(value, chosen_category = None):
    # Try all values
    for i in range(256):
        for category in range(1,6):
            testval = get_elmo_leakage_value(i, category)
            if abs(value - testval) < 0.0000001:
                return i, category
    # Try closest
    closest_val = -1
    closest_difference = 1
    closest_category = -1
    closest_elmo = -1
    for i in range(256):
        if chosen_category is None:
            for category in range(1,6):
                testval = get_elmo_leakage_value(i, category)
                if abs(value - testval) < closest_difference:
                    closest_val = i
                    closest_difference = abs(value - testval)
                    closest_category = category
                    closest_elmo = testval
        else:
            testval = get_elmo_leakage_value(i, chosen_category)
            if abs(value - testval) < closest_difference:
                closest_val = i
                closest_difference = abs(value - testval)
                closest_category = chosen_category
                closest_elmo = testval
    print "Brute Force Unsuccessful. Closest Value {}, Category {} (Val {}, Difference {})".format(closest_val, closest_category, closest_elmo, closest_difference)
    return closest_val, closest_category

def get_category_bf(value):
    return brute_force_elmo_value(value)[1]

def get_value_bf(value):
    return brute_force_elmo_value(value)[0]

def get_list_of_value_matches(v, x):
    return np.where(v==x)[0]

def strip_list(lst):
    length = len(lst)
    for i in range(length):
        try:
            if len(lst[-1]) <= 1:
                lst.pop()
            else:
                return lst
        except TypeError:
            lst.pop()
    print "oh no, strip list did a bad"
    return lst

def mad_based_outlier(lst, thresh=3.5):
    points = np.array(lst)
    # https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    if med_abs_deviation <= 0:
        med_abs_deviation = 0.00000001
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / SQRT2

cdef martin_rc(int j, int small_w, int i, int m, int big_w1, int big_w2, int big_w):
    if small_w + big_w >= big_w2:
        # reject is -1
        return -1
    elif j == m - 1:
        if small_w + big_w < big_w1:
            return -1
        else:
            # accept is -2
            return -2
    else:
        return small_w + big_w

def martin_rank(big_w, int m = 16, int n = 256, int big_w1 = 0, factor = 100000):
    cdef int i, j, little_w, big_w2, child
    little_w = 0
    big_w2 = 0
    for i in range(m):
        big_w[i] = array_cast_to_int(array_multiply_constant(big_w[i], factor))
        big_w2 += sum(big_w[i])
    count = [0] * big_w2
    old_count = [0] * big_w2
    for j in range(m - 1, -1, -1):
        print "Outer j: {}".format(j)
        for w in range(big_w2):
            for i in range(n - 1, -1, -1):
                child = martin_rc(j, little_w, i, m, big_w1, big_w2, big_w[j][i])
                if child == -2:
                    count[little_w] += 1
                elif child != -1:
                    count[little_w] += old_count[child]
        old_count = count
        count = [0] * big_w2
    return old_count[0]

def check_for_outliers(incoming_messages_list, t=3.5):
    trace_outlier_count = [0] * len(incoming_messages_list[0][0])
    for i in range(16):
        for j in range(256):
            value_list = incoming_messages_list[i][j]
            # Do stuff
            check_outliers = mad_based_outlier(value_list, thresh=t)
            outliers_index = list(np.nonzero(check_outliers)[0])
            for index in outliers_index:
                trace_outlier_count[index] += 1
    return trace_outlier_count

def dump_result(test = "Standard", connecting_method = "Large Factor Graph", traces = 1, rounds = 5, repeats = 1, snrexp = 5.5, noleak = None, badly = None, removed = None, threshold = 2, epsilon = 0.015, epsilon_s = 10, rank = "Unknown", traces_needed = "Not Applicable"):
    if noleak is None:
        noleak = []
    if badly is None:
        badly = []
    if removed is None:
        removed = []
    append_dump(format_dump_string(test, connecting_method, traces, rounds, repeats, snrexp, noleak, badly, removed, threshold, epsilon, epsilon_s, rank, traces_needed))

def format_dump_string(test = "Standard", connecting_method = "Large Factor Graph", traces = 1, rounds = 5, repeats = 1, snrexp = 5.5, noleak = None, badly = None, removed = None, threshold = 2, epsilon = 0.015, epsilon_s = 10, rank = "Unknown", traces_needed = "Not Applicable"):
    if noleak is None:
        noleak = []
    if badly is None:
        badly = []
    if removed is None:
        removed = []
    dump_string = "\n*** Test {} ***\nTime is {}\n".format(test, datetime.now())
    dump_string += "Connecting with: {}\n".format(connecting_method)
    dump_string += "Traces:    {:10}\nRounds:    {:10}\nRepeats:   {:10}\nSNR Exp:   {:10}\n".format(traces, rounds, repeats, snrexp)
    dump_string += "Threshold: {:10}\nEpsilon:   {:10}\nEpsilon s: {:10}\n".format(threshold, epsilon, epsilon_s)
    dump_string += "Not Leaking Nodes:     {:10}\nBadly Leaking Nodes:   {:10}\nRemoved Nodes:         {:10}\n".format(noleak, badly, removed)
    dump_string += "Average Traces Needed: {:10}\n".format(traces_needed)
    dump_string += "Final Key Rank: {:10}".format(rank)
    return dump_string

def clear_csv(filepath):
    f = open(filepath, 'w+')
    f.write("")
    f.close()

def append_csv(filepath, string):
    f = open(filepath, 'a+')
    f.write(string)
    f.close()

def read_line_from_csv(file_path, line_number):
    return linecache.getline(file_path, line_number)

def load_from_csv(filepath):
    f = open(filepath, 'r')
    a = f.readline()
    f.close()
    return eval(a)

def append_dump(string):
    append_csv('output/data_dump.txt', '\n' + string)

def remove_brackets(lst):
    string = str(lst)
    return (string.replace(']','')).replace('[','')

def get_statistics_string(l, log = False):
    if len(l) == 0:
        return "List is empty!"
    # Max, Min, Average, Median, Range, Variance
    max_l = max(l)
    min_l = min(l)
    avg_l = get_average(l)
    med_l = array_median(l)
    if type(max_l) is not long and np.isinf(max_l):
        range_l = np.inf
    elif type(min_l) is not long and np.isnan(min_l):
        range_l = np.nan
    else:
        range_l = max_l - min_l
    var_l = array_variance(l)
    # Extra strings
    if log:
        max_l   = bit_length(max_l)
        min_l   = bit_length(min_l)
        avg_l   = bit_length(avg_l)
        med_l   = bit_length(med_l)
        range_l = bit_length(range_l)
        var_l   = bit_length(var_l)
    # Print
    return "{}, {}, {}, {}, {}, {}".format(max_l, min_l, avg_l, med_l, range_l, var_l)

def get_log_list(l):
    return [bit_length(i) for i in l]

def print_statistics(l, log = False, top = False):
    l = np.array(l)
    if len(l) == 0:
        print "List l is empty!"
    else:
        # Max, Min, Average, Median, Range, Variance
        max_l = max(l)
        min_l = min(l)
        avg_l = get_average(l)
        med_l = array_median(l)
        if type(max_l) is not long and np.isinf(max_l):
            range_l = np.inf
        elif type(min_l) is not long and np.isnan(min_l):
            range_l = np.nan
        else:
            range_l = max_l - min_l
        var_l = array_variance(l)

        # Extra strings
        max_l_log = ""
        min_l_log = ""
        avg_l_log = ""
        med_l_log = ""
        range_l_log = ""
        var_l_log = ""
        if log:
            loglist = get_log_list(l)
            geo_average = int(round(get_average(loglist)))
            geo_average2 = 2**geo_average
            max_l_log = " (~2^{})".format(bit_length(max_l))
            min_l_log = " (~2^{})".format(bit_length(min_l))
            avg_l_log = " (~2^{})".format(bit_length(avg_l))
            geo_l_log = " (~2^{})".format(geo_average)
            med_l_log = " (~2^{})".format(bit_length(med_l))
            range_l_log = " (~2^{})".format(bit_length(range_l))
            var_l_log = " (~2^{})".format(bit_length(var_l))
        # Print
        if top:
            top_list = [1,5,10,20]
            for t in top_list:
                top_1 = ((np.where(l<=t)[0].size) / (l.size + 0.0)) * 100
                print "Top{:2}:{:40}%".format(t, top_1)
        print "Max:  {:40} {}".format(max_l, max_l_log)
        print "Min:  {:40} {}".format(min_l, min_l_log)
        print "AriM: {:40} {}".format(avg_l, avg_l_log)
        if log:
            print "GeoM: {:40} {}".format(geo_average2, geo_l_log)
        print "Med:  {:40} {}".format(med_l, med_l_log)
        print "Rng:  {:40} {}".format(range_l, range_l_log)
        print "Var:  {:40} {}".format(var_l, var_l_log)
        print_new_line()

def save_statistics(name, l):
    l = np.array(l)
    if len(l) == 0:
        print "List l is empty!"
    else:
        # Max, Min, Average, Median, Range, Variance
        max_l = max(l)
        min_l = min(l)
        avg_l = get_average(l)
        med_l = array_median(l)
        if type(max_l) is not long and np.isinf(max_l):
            range_l = np.inf
        elif type(min_l) is not long and np.isnan(min_l):
            range_l = np.nan
        else:
            range_l = max_l - min_l
        var_l = array_variance(l)

        top_list = [1,5,10,20]
        top_l = [0] * len(top_list)
        for i, t in enumerate(top_list):
            top_l[i] = (np.where(l<=t)[0].size / (l.size + 0.0)) * 100

        my_string = '{},{},{},{},{},{},{},{},\n'.format(name, top_l[0], top_l[1], top_l[2], top_l[3], med_l, avg_l, max_l)
        append_csv(TESTING_MODELS_CSV, my_string)

def clear_statistics():
    clear_csv(TESTING_MODELS_CSV)
    append_csv(TESTING_MODELS_CSV, 'Name,Top 1 (%),Top 5 (%),Top 10 (%),Top 20 (%),Median,Average,Max,\n')
    append_csv(TESTING_MODELS_CSV, '* BASELINE *,0.4,1.9,3.9,7.8,128,128,255,\n')


def hex_string_to_int_array(hex_string):
    if (len(hex_string) % 2) == 1:
        print "Hex String {} has {} characters, must be even".format(hex_string, len(hex_string))
        raise ValueError
    out = [0] * (len(hex_string)/2)
    for i in range(0, len(hex_string), 2):
        hexbyte_str = hex_string[i:i+2]
        try:
            hexbyte = eval('0x'+hexbyte_str)
        except NameError:
            print "Can't evaluate byte {}".format(hexbyte_str)
            raise
        out[i/2] = hexbyte
    return out

def plot_two_distributions(dist1, dist2, dist1_name = 'Key Byte 001', dist2_name = 'Plaintext Byte 001'):
    plt.subplot(1, 2, 1)
    plt.plot(dist1)
    plt.xticks(np.arange(0, 257, 32))
    plt.title(dist1_name)
    plt.ylabel('Probability')
    plt.xlabel('Index')

    plt.subplot(1, 2, 2)
    plt.plot(dist2)
    plt.xticks(np.arange(0, 257, 32))
    plt.title(dist2_name)
    plt.ylabel('Probability')
    plt.xlabel('Index')

    plt.tight_layout()

    plt.show()


# Print iterations progress
def printProgressBar(iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)),'\r'
    # Print New Line on Complete
    if iteration == total:
        print ""

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def load_trace_data(filepath=TRACEDATA_FILEPATH, memory_mapped = True):
    if memory_mapped:
        profile_traces, attack_traces, samples, coding = load_meta()
        if filepath == TRACEDATA_FILEPATH:
            # used_traces = int(traces * cutoff_percent)
            used_traces = profile_traces
        elif filepath == TRACEDATA_EXTRA_FILEPATH:
            # used_traces = int(traces * (1 - cutoff_percent))
            used_traces = attack_traces
        print ">>> Loading Trace Data, used_traces = {}, memory_mapped: {}".format(used_traces, memory_mapped)
        return np.memmap(filepath, dtype=coding, mode='r+', shape=(used_traces, samples))
    else:
        return np.load(filepath, mmap_mode='r')

def print_details(x):
    print "Type: {}, Contents: {}".format(type(x), x)









def get_value_from_plaintext_array(v):
    return np.where(v==1)[0][0]

#### bpann helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the bpann
# database
def load_bpann(variable, load_metadata=False, normalise_traces=True, input_length=700, training_traces=50000, sd = 100, augment_method=2):

    # Load meta
    profile_traces, attack_traces, samples, coding = load_meta()

    # print "Loading BPA NN Files for variable {}!".format(variable)
    var_name, var_number, _ = split_variable_name(variable)

    # TRY LOADING FIRST
    filename = '{}_meta{}_norm{}_input{}_training{}_sd{}_aug{}.pkl'.format(variable, load_metadata, normalise_traces, input_length, training_traces, sd, augment_method)

    try:
        return load_object(TEMP_FOLDER + filename)
    except IOError:

        # Get time point for variable
        time_point = np.load('{}{}.npy'.format(TIMEPOINTS_FOLDER, var_name))[var_number-1]

        start_window = max(0, time_point - (input_length/2))
        end_window = min(samples, time_point + (input_length/2))

        trace_data = load_trace_data()[:, start_window:end_window]
        traces, data_length = trace_data.shape
        type = trace_data.dtype
        real_values = np.load('{}{}.npy'.format(REALVALUES_FOLDER, var_name))[var_number-1,:]

        if training_traces > traces:
            print 'Augmenting {} Traces!'.format(training_traces - traces)

            # X_profiling = np.empty((training_traces, data_length), dtype=type)
            X_profiling = np.memmap('{}tmp_{}_{}_sd{}_window{}_aug{}.mmap'.format(TRACE_FOLDER, variable, training_traces, sd, input_length, augment_method), shape=(training_traces, data_length), mode='w+', dtype=type)
            Y_profiling = np.empty(training_traces, dtype=int)

            X_profiling[:traces] = trace_data
            Y_profiling[:traces] = real_values

            for train_trace in range(traces, training_traces):

                # AUGMENT METHODS
                # 0 - gaussian noise
                # 1 - time warping
                # 2 - averaging traces

                # Get Random Number
                random_number = np.random.randint(0, traces)

                # Add label
                Y_profiling[train_trace] = real_values[random_number]

                if augment_method == 0:

                    # GAUSSIAN NOISE
                    random_noise = np.random.normal(0, sd, data_length).round().astype(int)

                    # Add to Profiling after applying noise
                    X_profiling[train_trace] = (trace_data[random_number] + random_noise).astype(type)

                elif augment_method == 1:

                    # TIME WARPING
                    random_shift = 0
                    while random_shift == 0:
                        random_shift = np.random.randint(-MAX_SHIFT, MAX_SHIFT)
                        print random_shift

                    X_profiling[train_trace] = roll_and_pad(trace_data[random_number], random_shift)

                elif augment_method == 2:

                    # AVERAGING TRACES (need to be have same value!)
                    pot = np.where(real_values == real_values[random_number])[0]
                    other_trace = pot[np.random.randint(0,len(pot)-1)]

                    X_profiling[train_trace] = np.mean(np.array([trace_data[random_number], trace_data[other_trace]]), axis=0)



        else:
            # Load profiling traces
            X_profiling = trace_data[:training_traces, :]
            # Load profiling labels
            Y_profiling = real_values[:training_traces]

        # Load attack traces
        X_attack = load_trace_data(filepath=TRACEDATA_EXTRA_FILEPATH)[:, start_window:end_window]
        # Load attacking labels
        Y_attack = np.load('{}extra_{}.npy'.format(REALVALUES_FOLDER, var_name))[var_number-1,:]

        if normalise_traces:
            print "£ Normalising traces"
            X_profiling = normalise_neural_traces(X_profiling)
            X_attack = normalise_neural_traces(X_attack)

        # Save!
        save_object(((X_profiling, Y_profiling), (X_attack, Y_attack)), TEMP_FOLDER + filename)

        return (X_profiling, Y_profiling), (X_attack, Y_attack)

def normalise_neural_trace(v):
    # Shift up
    return v - np.min(v)

def normalise_neural_traces(X):
    if X.shape[0] > 200000:
        # Memory error: do sequentially
        out = np.empty(X.shape)
        for i in range(X.shape[0]):
            temp = normalise_neural_trace(X[i])
            out[i] = np.divide(temp, np.max(temp) + 0.0)
        return out
    else:
        temp = (np.apply_along_axis(normalise_neural_trace, 1, X))
        return np.divide(temp, np.max(temp) + 0.0)

def get_input_length(str):
    try:
        return int(re.search('^.*_window(\d+)_.*', str).group(1))
    except AttributeError:
        return 700

def get_training_traces(str):
    try:
        return int(re.search('^.*_traces(\d+).*', str).group(1))
    except AttributeError:
        return 50000

# variable_list = ['{}{}'.format(k, pad_string_zeros(i+1)) for k, v in variable_dict.iteritems() for i in range(v)]
# variable_list = ['{}{}'.format(vk, vi) for vi in range(vv) for vk,vv in variable_dict.iteritems()]
# variable_list = ['{}{}'.format(vk, vi) for vk,vi in variable_dict.iteritems()]


if PRELOAD_NEURAL_NETWORKS:
    print ">> Preloading Neural Networks..."
    neural_network_dict = dict()
    for v, length in variable_dict.iteritems():
        for i in range(length):
            varname = '{}{}'.format(v, pad_string_zeros(i+1))
            neural_network_dict[varname] = load_sca_model('{}{}_mlp5_nodes200_window700_epochs6000_batchsize200_sd100_traces200000_aug0.h5'.format(NEURAL_MODEL_FOLDER, varname))
    print ">> ...done!"
