from utility import *
from numpy import genfromtxt
from os.path import expanduser
from scipy.stats import entropy
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser(description='Calculates Distance Statistics from Target Node')
parser.add_argument('--FIRST_ONLY', action="store_true", dest="FIRST_ONLY",
                    help='Only compares to first key byte (default: False)', default=False)
parser.add_argument('--KL_DIV', action="store_true", dest="KL_DIV", help='Uses KL Divergence (default: False)',
                    default=False)
parser.add_argument('--EUC_DIV', action="store_true", dest="EUC_DIV", help='Uses Euclidian Distance (default: False)',
                    default=False)
parser.add_argument('--CSV', action="store_true", dest="CSV", help='Adds all to CSV (default: False)', default=False)
parser.add_argument('--TUPLE', action="store_true", dest="TUPLE", help='Uses Tuple Results (default: False)',
                    default=False)
parser.add_argument('--HW', action="store_true", dest="HW", help='Uses HW Results (default: False)', default=False)
parser.add_argument('--PLAINTEXT', action="store_true", dest="PLAINTEXT",
                    help='Uses Plaintext Results (default: False)', default=False)
parser.add_argument('--PRINT', action="store_true", dest="PRINT", help='Prints Results (default: False)', default=False)

# Test
snrexp = 1
fsnrexp = str(float(snrexp))

# Target node here
args = parser.parse_args()
FIRST_ONLY = args.FIRST_ONLY
KL_DIV = args.KL_DIV
EUC_DIV = args.EUC_DIV
CSV = args.CSV
PLAINTEXT = args.PLAINTEXT
HW = args.HW
TUPLE = args.TUPLE
PRINT = args.PRINT

key_nodes = ['k{}-K'.format(padStringZeroes(i + 1)) for i in range(16)]

home = expanduser("~")

# csv_path = home + "/Desktop/BPAResults/FixingValues/OneRound/"
# csv_path = home + "/Desktop/BPAResults/FixingValues/Converges/"
csv_path = home + "/Desktop/BPAResults/FixingValues/Converges/ELMOArray/"

# if not PLAINTEXT and not HW and not TUPLE:
#     print "|| No Results Selected, selecting Tuple"
#     TUPLE = True
#
# if TUPLE:
#     print "Computing Tuple"
#     csv_path += "Tuple/"
# elif PLAINTEXT:
#     print "Computing Plaintext"
#     csv_path += "PlaintextArray/"
# elif HW:
#     print "Computing Hamming Weight Array"
#     csv_path += "HammingWeightArray/"
# else:
#     print "Something went horribly wrong, aborting"
#     exit(1)

snr_path = csv_path + "snr{}/".format(snrexp)

for seed in range(5):

    print "+++ Seed {} +++".format(seed)
    print_new_line()

    pr_k = genfromtxt(snr_path + 'marginaldist_K_{}_.csv'.format(seed), delimiter=',')

    # Get set of all available files
    onlyfiles = [f for f in listdir(snr_path) if isfile(join(snr_path, f))]
    variable_nodes = set()
    for filename in onlyfiles:
        try:
            nodename = filename.split('_')[1]
            myseed = filename.split('_')[4]
            if seed == int(myseed) and string_contains(nodename, '-'):
                variable_nodes.add(nodename)
        except IndexError:
            pass

    print variable_nodes

    # For all target nodes

    for target_node in variable_nodes:

        print "*** Computing Marginal Distances from node {} ***".format(target_node)

        # Store all csv data in list
        pr_node = [0] * 16
        for i, key_node in enumerate(key_nodes):
            key_node_path = snr_path + 'marginaldist_{}_{}_{}_{}_.csv'.format(target_node, key_node, fsnrexp, seed)
            try:
                pr_node[i] = genfromtxt(key_node_path, delimiter=',')
            except IOError:
                print "!!! Error: Could not find file for key node {} at {}".format(key_node, key_node_path)
                raise

        # All Stored
        # print "...all data stored successfully."

        # print "* Statistics *"
        # print_new_line()

        kldiv_list = [0] * 16
        # Loop through key nodes
        for i, key_node in enumerate(key_nodes):
            kldiv_list_keynode = [0] * 256
            # Marginal Distribution of Key Node (no fix)
            margdist_keynode = plug_zeros(pr_k[i])
            # Marginal Distributions of fixed value, specific Key Node
            margdist_fixedvalues = pr_node[i]
            # This will be of size 256: Loop over all and store the KL Divergence of Keynode to Fixed Value
            # noinspection PyTypeChecker
            for j in range(len(margdist_fixedvalues)):
                if KL_DIV:
                    # kldiv_list_keynode[j] = entropy(margdist_keynode, plug_zeros(margdist_fixedvalues[j]))
                    kldiv_list_keynode[j] = entropy(plug_zeros(margdist_fixedvalues[j]), margdist_keynode)
                elif EUC_DIV:
                    kldiv_list_keynode[j] = euclidian_distance(margdist_fixedvalues[j], margdist_keynode)
                else:
                    # TODO
                    kldiv_list_keynode[j] = normal_distance(margdist_fixedvalues[j], margdist_keynode)

            kldiv_list[i] = kldiv_list_keynode

            # Print statistics
            if PRINT:
                print "> Key Node {} to Node {}".format(key_node, target_node)
                print_statistics(kldiv_list_keynode)

            if CSV:
                # Put in CSV
                out_string = "{}, {}, {}, {}\n".format(snrexp, key_node, target_node,
                                                       get_statistics_string(kldiv_list_keynode))
                # print out_string
                prefix = "" + ("KL" * KL_DIV) + ("EUC" * EUC_DIV) + ("DIS" * (not EUC_DIV and not KL_DIV))
                f = open(csv_path + '{}Results_{}.csv'.format(prefix, seed), 'a+')
                f.write("{}".format(out_string))
                f.close()

            if FIRST_ONLY:
                break

        # print kldiv_list
        # print_new_line()
