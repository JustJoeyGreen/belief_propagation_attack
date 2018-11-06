from utility import *
from collections import OrderedDict
# from scipy.special import entr
from scipy.stats import entropy
from os import listdir
from os.path import isfile, join, expanduser

import argparse
import leakageSimulatorAESFurious as lSimF

KEY = np.array([0x54, 0x68, 0x61, 0x74, 0x73, 0x20, 0x6D, 0x79, 0x20, 0x4B, 0x75, 0x6E, 0x67, 0x20, 0x46, 0x75])
PLAINTEXT = np.array([56, 117, 74, 5, 214, 142, 164, 47, 254, 220, 30, 85, 184, 182, 239, 108])

list_of_targets = ['s', 't', 'mc', 'xt', 'cm']
# list_of_targets = ['s']
alphabet = ['A', 'B', 'C', 'D']

# distance_metrics = {'entropy': (0, 3)}
# distance_metrics = {'probability of key': (0, 1)}
# distance_metrics = {'euclidean distance': (0, 10)}
# distance_metrics = {'KL divergence': (0, 10)}
distance_metrics = {'hellinger distance': (0, 1.2)}

# distance_metrics = {'entropy': (0, 5), 'probability of key': (0, 1),
#                     'euclidean distance': (0, 2), 'KL divergence': (0, 128)}

parser = argparse.ArgumentParser(description='Calculates Distance Statistics from Target Node')
parser.add_argument('--FIRST_ONLY', action="store_true", dest="FIRST_ONLY",
                    help='Only compares to first key byte (default: False)', default=False)
parser.add_argument('--FIRST_COLUMN', '--ONE_COLUMN', action="store_true", dest="FIRST_COLUMN",
                    help='Only compares to first column (default: False)', default=False)
parser.add_argument('--FIRST_KEYBYTE', '--ONE_BYTE', action="store_true", dest="FIRST_KEYBYTE",
                    help='Only compares to first key byte (default: False)', default=False)
parser.add_argument('--PRINT', action="store_true", dest="PRINT", help='Prints Results (default: False)', default=False)
parser.add_argument('--PLOT', action="store_true", dest="PLOT", help='Plots Results (default: False)', default=False)
parser.add_argument('--CHECK', '--CHECK_KEY', action="store_false", dest="CHECK_KEY", help='Checks Key (default: True)', default=True)
parser.add_argument('--ENTROPY', action="store_true", dest="ENTROPY", help='Uses Entropy (default: False)', default=False)
parser.add_argument('--NORMALISE', action="store_true", dest="NORMALISE", help='Normalises (default: False)', default=False)
parser.add_argument('-snrexp', action="store", dest="SNR_exp", help='SNR Exponent, s.t. SNR = 2**SNR_exp (default: 0)',
                    type=int, default=0)
parser.add_argument('-maxround', '-r', action="store", dest="MAX_ROUND", help='Maximum Round to plot up to (default: 4)',
                    type=int, default=4)

# Target node here
args = parser.parse_args()
FIRST_ONLY = args.FIRST_ONLY
PRINT = args.PRINT
PLOT = args.PLOT
ENTROPY = args.ENTROPY
NORMALISE = args.NORMALISE
SNR_exp = args.SNR_exp
CHECK_KEY = args.CHECK_KEY
FIRST_COLUMN = args.FIRST_COLUMN
FIRST_KEYBYTE = args.FIRST_KEYBYTE
MAX_ROUND = args.MAX_ROUND

TITLE = False

if FIRST_KEYBYTE and FIRST_COLUMN:
    FIRST_COLUMN = False

fsnrexp = str(float(SNR_exp))

NOISY = True

if NOISY:
    PATH_TO_MARGDISTS = '~/Desktop/BPAResults/FixingValues/Noisy/'
else:
    PATH_TO_MARGDISTS = '~/Desktop/BPAResults/FixingValues/NoNoise/'
    fsnrexp = '5'

# PATH_TO_MARGDISTS = 'output/'
# PATH_TO_MARGDISTS = '~/Desktop/'



if '~' in PATH_TO_MARGDISTS:
    PATH_TO_MARGDISTS = os.path.expanduser(PATH_TO_MARGDISTS)

key_nodes = ['k{}-K'.format(pad_string_zeros(i + 1)) for i in range(16)]

# Get set of all available files

marginal_distributions_files = [f.replace('.npy', '') for f in listdir(PATH_TO_MARGDISTS) if
                                isfile(join(PATH_TO_MARGDISTS, f)) and string_starts_with(f, 'marginaldist_')]

# print marginal_distributions_files

sim = lSimF.LeakageSimulatorAESFurious()
sim.fix_key(KEY)
sim.fix_plaintext(PLAINTEXT)
sim.simulate(read_plaintexts=0, print_all=0, random_plaintexts=0, affect_with_noise=False,
             hw_leakage_model=False, real_values=True)
leakage_dict = sim.get_leakage_dictionary()

# Get Default Key Distribtion
key_file = [k_file for k_file in marginal_distributions_files if '_K_' in k_file]
if len(key_file) > 1:
    print "TODO: Handle multiple key files: {}".format(key_file)
    exit(1)
elif len(key_file) < 1:
    print "!!! Error: No Key File Found! List of possible files:\n\n{}\n".format(marginal_distributions_files)
    exit(1)

default_key_distribution = np.load("{}{}.npy".format(PATH_TO_MARGDISTS, key_file[0]))

# CHECK FIRST
if CHECK_KEY:
    print "Checking Key..."
    check_key = default_key_distribution
    checked = np.argmax(check_key, axis=1)
    print "KEY:\n{}".format(KEY)
    print "GOT:\n{}".format(checked)
    print "MATCH: {}".format(np.array_equal(KEY, checked))
    print ""

for distance_metric, axes_minmax in distance_metrics.iteritems():

    max_values = list()

    for target_variable in list_of_targets:

        seed = 0

        if PRINT:
            print "*** Seed {} ***".format(seed)

        if PLOT:
            fig = plt.figure()
            plt.rc('text', usetex=True)
            if not FIRST_KEYBYTE:
                plt.rcParams.update({'font.size': 10})
                st = fig.suptitle("{} as a Distance Metric Fixing Variable {}".format(distance_metric.title(), target_variable), fontsize="x-large", )
                plt.rcParams.update({'font.size': 3})


        for key_byte in range(16):

            if (FIRST_COLUMN and key_byte not in [0, 5, 10, 15]) or (FIRST_KEYBYTE and key_byte not in [0]):
                pass
            else:

                if PRINT:
                    print "*** k{} ***".format(pad_string_zeros(key_byte + 1))

                variable_nodes = dict()

                for filename in marginal_distributions_files:
                    try:
                        _, file_variable, file_snr, file_seed = filename.split('_')
                        if seed == int(file_seed) and file_snr == fsnrexp:
                            variable_nodes[file_variable] = filename
                    except IndexError:
                        print "Ignoring IndexError on file {}".format(filename)
                        pass
                    except ValueError:
                        print "Splitting the file name throws an error: filename {} splits to {} (expected length 5)".format(
                            filename, filename.split('_'))
                        raise ValueError

                # Now we have variable nodes, get max and min
                for variable, filename in sorted(variable_nodes.iteritems()):

                    if string_starts_with(variable, target_variable):

                        if PRINT:
                            print "**************** Variable {} ****************\n".format(variable)

                        if string_contains(variable, '65') or string_contains(variable, '81'):
                            print "...ignoring"
                            continue

                        # Swap axes from margdist[keynode][fixedval] to margdist[keynode][node]
                        numpyfile = pickle.load(open(PATH_TO_MARGDISTS + filename + '.npy', 'ro'))

                        # numpyfile is shape (16, 256, 256) -> (key_byte, fixed_value, index_of_marginal)
                        for i_count in range(numpyfile.shape[0]):
                            for j_count in range(numpyfile.shape[1]):
                                if is_zeros_array(numpyfile[i_count][j_count]):
                                    print "!!! In Checking file {}, Zeros array found: {} {}\n{}".format(filename, i_count, j_count,
                                                                                    numpyfile[i_count][j_count])
                                    exit(1)

                        if CHECK_KEY:
                            failed_fixed_values = list()
                            for fixed_value in range(numpyfile.shape[1]):
                                estimated_key = np.argmax(numpyfile[:, fixed_value, :], axis=1)
                                # print "FIXED VALUE {:3}, ESTIMATED:\n{}\n".format(fixed_value, estimated_key)
                                if not np.array_equal(KEY, estimated_key):
                                    failed_fixed_values.append(fixed_value)
                            if len(failed_fixed_values) > 0:
                                print "Failed Fixed Values:\n{}\nTotal Failed: {}\nPercentage Failed: {}%\n".format(failed_fixed_values, len(failed_fixed_values), (len(failed_fixed_values) / 256.0 * 100))
                            else:
                                print "Success on all fixed values."

                        # If entropy, do something different!!!
                        # numpyfile is shape (16, 256, 256) -> (key_byte, fixed_value, index_of_marginal)
                        # Get entropy to be shape (16, 256) -> (key_byte, fixed_value)
                        if distance_metric == 'entropy':
                            our_results = np.empty((numpyfile.shape[0], numpyfile.shape[1]))
                            for i_count in range(numpyfile.shape[0]):
                                for j_count in range(numpyfile.shape[1]):
                                    if is_zeros_array(numpyfile[i_count][j_count]):
                                        print "!!! Zeros array found, {} {}\n{}".format(i_count, j_count, numpyfile[i_count][j_count])
                                        exit(1)
                                    current_entropy = np.abs(entropy(numpyfile[i_count][j_count]))
                                    if np.isnan(current_entropy) or np.isinf(current_entropy):
                                        current_entropy = 0.0
                                    our_results[i_count][j_count] = current_entropy

                        elif distance_metric == 'KL divergence':
                            our_results = np.empty((numpyfile.shape[0], numpyfile.shape[1]))
                            for i_count in range(numpyfile.shape[0]):
                                for j_count in range(numpyfile.shape[1]):
                                    current_entropy = np.abs(entropy(numpyfile[i_count][j_count], default_key_distribution[i_count]))
                                    if np.isnan(current_entropy) or np.isinf(current_entropy):
                                        current_entropy = 0.0
                                    our_results[i_count][j_count] = current_entropy

                        elif distance_metric == 'hellinger distance':
                            our_results = np.empty((numpyfile.shape[0], numpyfile.shape[1]))
                            for i_count in range(numpyfile.shape[0]):
                                for j_count in range(numpyfile.shape[1]):
                                    current_distance = np.abs(hellinger_distance(numpyfile[i_count][j_count], default_key_distribution[i_count]))
                                    if np.isnan(current_distance) or np.isinf(current_distance):
                                        current_distance = 0.0
                                    our_results[i_count][j_count] = current_distance
                                    # print our_results
                            print our_results

                        elif distance_metric == 'probability of key':
                            numpyfile = np.swapaxes(numpyfile, 1, 2)
                            our_results = numpyfile[np.arange(16), KEY]

                        elif distance_metric == 'euclidean distance':
                            our_results = np.empty((numpyfile.shape[0], numpyfile.shape[1]))
                            for i_count in range(numpyfile.shape[0]): # Loop over all key nodes, should probably remove this at some point...
                                for j_count in range(numpyfile.shape[1]):
                                    current_distance = np.linalg.norm(numpyfile[i_count][j_count] - default_key_distribution[i_count]) # Do the thing here
                                    our_results[i_count][j_count] = current_distance
                        else:
                            print "!!! Unknown Distance Metric: {}".format(distance_metric)
                            break
                            # exit(1)

                        max_values.append(np.max(our_results[key_byte]))
                        if PLOT:

                            plot_array = our_results[key_byte]
                            # TEST
                            # print "Largest value in Plot Array for Metric {}, Key Node {}, Fixed Node {}: {}".format(distance_metric, key_byte, variable, np.max(plot_array))
                            if NORMALISE:
                                plot_array = normalise_array(plot_array)
                            if FIRST_COLUMN:
                                ax = fig.add_subplot(4, 1, (key_byte // 5) + 1)
                            elif FIRST_KEYBYTE:
                                ax = fig.add_subplot(1, 1, 1)
                            else:
                                ax = fig.add_subplot(4, 4, key_byte + 1)
                            var_name, var_number, _ = split_variable_name(variable)
                            real_val = int(leakage_dict[get_variable_name(variable)][0][get_variable_number(variable)-1])
                            ax.plot(np.arange(our_results[key_byte].shape[0]), plot_array, label=r'$%s_{%d}$, Round %d' % (var_name, var_number, get_round_of_variable(variable)), linewidth=0.3)
                            ttl = ax.title
                            # ttl.set_position([.5, 0.75])
                            # ax.set_title("Key node $k_{%d}$" % ((key_byte + 1), KEY[key_byte]))
                            if not FIRST_KEYBYTE:
                                ax.set_title("Plot {}".format(alphabet[key_byte // 5]))
                            ax.set_xlim([0, 255])
                            # ax.set_ylim([axes_minmax[0], axes_minmax[1]])

                            ax.spines['right'].set_color('none')
                            ax.spines['top'].set_color('none')


                        if PRINT:
                            for i, key_node in enumerate(our_results):
                                # print "-> Key node k{}-K".format(pad_string_zeros(i+1))
                                if np.max(key_node) - np.min(key_node) == 0:
                                    # print "+ NO CHANGE IN PROBABILITIES (all values {})\n".format(np.min(key_node))
                                    pass
                                else:
                                    print "-> Key node k{}-K".format(pad_string_zeros(i + 1))
                                    print_array = key_node
                                    if NORMALISE:
                                        print_array = normalise_array(print_array)
                                    print_statistics(print_array)

                if PLOT:
                    plt.xlabel('Value of fixed node(s)')
                    # plt.xlabel('Fixed Value\nKey node $k_{%d}$ (Real Value %d)' % ((key_byte + 1), KEY[key_byte]))

                    plt.ylabel(distance_metric.title())
                    if FIRST_COLUMN:
                        plt.xticks(np.arange(0, 257, 16))
                    else:
                        plt.xticks(np.arange(0, 257, 64))
                    # if ENTROPY:
                    #     plt.title('Entropy Values on Distribution k{} (Correct Val {}), fixing Variable {}'.format(pad_string_zeros(key_byte + 1), KEY[key_byte], target_variable))
                    # else:
                    #     plt.title('Marginal Distribution of k{}'.format(pad_string_zeros(key_byte + 1)))
                    # plt.show()

                if FIRST_ONLY:
                    break

        if not FIRST_KEYBYTE:
            plt.rcParams.update({'font.size': 5})
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        file_format = 'eps'
        if FIRST_KEYBYTE:
            plt.savefig('Output/{}_fixing_{}_k1.{}'.format(distance_metric.replace(" ", ""), target_variable, file_format), format=file_format, dpi=1200)
        else:
            plt.savefig('Output/{}_fixing_{}.{}'.format(distance_metric.replace(" ", ""), target_variable, file_format), format=file_format, dpi=1200)
        plt.show()

    print "Max Value for Metric {}: {}".format(distance_metric, max(max_values))