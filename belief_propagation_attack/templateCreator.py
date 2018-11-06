import timing
import linecache
import numpy as np
from utility import *

# n = 50000
n = 2
sample_size = 2
# sample_size = n

DEBUG = False
# DEBUG = True

# k
# target_line = 20 # Without
# target_line = 24 # With, 1
# target_line = 25 # With, 2

targets = [('111', 20), ('131', 31), ('134', 42), ('144', 53), ('431', 64), ('441', 75), ('444', 86)]
# targets = [('111', 19)]

# t
# target_line = 29 # Without
# target_line = 36 # With, 1
# target_line = 37 # With, 2

for target in targets:
    print "Target Name: {}, Target Line: {}".format(target[0], target[1])

    template = list()

    # For each set of templates
    for template_index in range(1, (n * 256) + 1, n):

        if DEBUG:
            print "Template for {}".format(template_index // n)

        # Only use a set number of templates
        current_lst = list()

        for i in range(sample_size):

            try:
                # value = float(linecache.getline('ELMOTraceTests/trace{}.trc'.format(padStringZeroes(template_index + i, 5)), 16).strip())
                value = float(linecache.getline(
                    '../ELMO/output/traces/trace{}.trc'.format(padStringZeroes(template_index + i, 5)),
                    target[1]).strip())
            except ValueError:
                print "ERROR: Value Error"
                raise

            current_lst.append(value)

        # Last one in group
        # y_values[(tracefile - 1) // n] = current_lst[:]
        array = np.array(current_lst)
        mean = np.mean(array)
        std = np.std(array)
        template.append((mean, std))

    # Once template list has been created, write to a file
    f = open("Templates/template_{}.txt".format(target[0]), "w+")
    f.write(str(template))
    f.close()
