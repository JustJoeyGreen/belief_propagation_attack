import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from utility import *

REMOVING_VARIATION = 1  # 0 for old method, 1 for complete removal of adjacent factors, 2 remove parent factors only
EDGES_TO_FILE = True
DEBUG = False

valid_nodes = list()

# First round - just
round_valid = [list() for i in range(3)]

round_valid[0] = [(var + pad_string_zeros(num)) for var in ['k', 'p', 't', 's'] for num in range(1, 17)]

round_valid[1] = [(var + pad_string_zeros(num)) for var in ['t', 's', 'mc', 'xt', 'cm'] for num in
                     range(1, 17)]
round_valid[1] += [(var + pad_string_zeros(num)) for var in ['k','p'] for num in range(1, 33)]
round_valid[1] += [('h' + pad_string_zeros(num)) for num in range(1, 13)]
round_valid[1] += [('xk' + pad_string_zeros(num)) for num in range(1, 2)]
round_valid[1] += [('sk' + pad_string_zeros(num)) for num in range(1, 5)]

round_valid[2] = [(var + pad_string_zeros(num)) for var in ['mc', 'xt', 'cm'] for num in
                     range(1, 17)]
round_valid[2] += [(var + pad_string_zeros(num)) for var in ['k','p','t','s'] for num in range(1, 33)]
round_valid[2] += [('h' + pad_string_zeros(num)) for num in range(1, 13)]
round_valid[2] += [('xk' + pad_string_zeros(num)) for num in range(1, 3)]
round_valid[2] += [('sk' + pad_string_zeros(num)) for num in range(1, 9)]


shift = list([0,
              1, 6, 11, 16,
              5, 10, 15, 4,
              9, 14, 3, 8,
              13, 2, 7, 12
              ])


def check_variable_factor_edge_match(variables, factors, edges):
    error = False
    for edge in edges:
        if edge[0] not in variables:
            if DEBUG: print "!!! Edge {}, Variable {} not previously defined".format(edge, edge[0])
            error = True
        elif edge[1] not in factors:
            if DEBUG: print "!!! Edge {}, Factor {} not previously defined".format(edge, edge[1])
            error = True
    if error:
        print_new_line()
        raise ValueError


def check_unconnected_nodes(variables, factors, edges):
    error = False
    to_remove = list()
    for node in variables:
        # TODO
        for edge in edges:
            if edge[0] == node or edge[1] == node:
                break
        else:
            if DEBUG: print "!!! {} is not connected to anything, removing".format(node)
            to_remove.append(node)
    for node in factors:
        # TODO
        for edge in edges:
            if edge[0] == node or edge[1] == node:
                break
        else:
            if DEBUG: print "!!! {} is not connected to anything, removing".format(node)
            to_remove.append(node)
    if error:
        print_new_line()
        # raise
    for node in to_remove:
        remove_node(variables, factors, edges, node)


def check_factor_in_round(factor, round):
    _, var = split_factor_name(factor)
    return check_variable_in_round(var, round)


def check_variable_in_round(variable, round):
    var_name, var_number, _ = split_variable_name(variable)
    if (var_name + pad_string_zeros(var_number)) in round_valid[round]:
        return True
    return False


def match_nodes(variables, removed_nodes):
    r_node_all = list()
    for r_node in removed_nodes:
        if r_node in variables:
            # Add node to list
            r_node_all.append(r_node)
        else:
            r_node_name, r_node_number, r_node_trace = split_variable_name(r_node)
            if r_node_number is None and r_node_trace is None:
                # Match all corresponding variables
                r_node_matches = get_variables_that_match(variables, r_node)
                if len(r_node_matches) > 0:
                    r_node_all += r_node_matches
                else:
                    print "! Cannot find matching nodes for node {}".format(r_node)
            elif r_node_trace is None:
                # Loop through variables
                for variable in variables:
                    v_name, v_number, _ = split_variable_name(variable)
                    if v_name == r_node_name and v_number == r_node_number:
                        r_node_all.append(variable)
            else:
                print "! Couldn't find node {} in graph".format(r_node)
    return r_node_all


def remove_node(variables, factors, edges, current_node, variation=1):
    # Old method: Merge
    if variation == 0:

        # Get Parent and Child Factor Nodes
        parent = None
        # parent_index = None
        child = None
        # child_index = None

        for edge in edges:

            if current_node == edge[0]:
                # Edge found. Check if parent or child
                if string_ends_with(edge[1], current_node):
                    parent = edge[1]
                else:
                    child = edge[1]
                # Break if done
                if parent is not None and child is not None:
                    # Remove both edges
                    edges.remove((current_node, parent))
                    edges.remove((current_node, child))
                    break
        else:
            print "* Warning: Variable {} has parent {} and child {}. Cannot remove!".format(current_node, parent,
                                                                                             child)
            pass

        # Only go on if both parent and child are present
        if parent and child:

            # print "Current Node {}: Rerouting edges from {} to {}...".format(current_node, parent, child)

            # Handle XOR - XOR (easiest)
            if is_xor_node(parent) and is_xor_node(child):

                # For each incoming edge to parent, delete it and add a new edge going to child
                for edge in list(edges):

                    if parent == edge[1]:
                        # Edge Found
                        # Add new edge
                        edges.append((edge[0], child))
                        # Delete old edge
                        edges.remove((edge[0], parent))

                # Finally, delete the parent and the target variable node
                factors.remove(parent)
                variables.remove(current_node)


            else:
                print "* Warning: Parent {} and Child {} not both XOR nodes. Cannot merge! (Work in Progress)".format(
                    parent, child)

            # Get all edges coming in to the parent factor node (list of variables)

    # Method 1: Remove factor either side
    elif variation == 1:

        # For each variable node, get the adjacent factors
        # Then just strip off the factor nodes, along with edges
        # print "Removing Node {}".format(current_node)
        adjacent_factors = [i for i in factors if (current_node, i) in edges]
        # TODO
        # print "Removing: {}".format(adjacent_factors)
        # print "Factors adjacent to {}: {}".format(current_node, adjacent_factors)
        for adjacent_factor in adjacent_factors:
            # Strip all edges
            connected_variables = [i for i in variables if (i, adjacent_factor) in edges]
            for connected_variable in connected_variables:
                edges.remove((connected_variable, adjacent_factor))
            # Delete Factor
            factors.remove(adjacent_factor)
        # Delete Variable
        variables.remove(current_node)

    # Method 2: Remove back factor only
    elif variation == 2:

        # For each variable node, get the adjacent factors
        # Then just strip off the factor nodes, along with edges
        # print "Removing Node {}".format(current_node)
        parent_factor = get_parent_factor_node(factors, current_node)
        # print "Parent factor of {}: {}".format(current_node, parent_factor)

        # Strip all edges
        connected_variables = [i for i in variables if (i, parent_factor) in edges]
        for connected_variable in connected_variables:
            edges.remove((connected_variable, parent_factor))
        # Delete Factor
        factors.remove(parent_factor)

    return variables, factors, edges


def create_factor_graph(number_of_traces, h_case=1, use_cm=True, use_mc=True):
    factor_graph = nx.Graph()

    edges = list()
    variables = list()
    factors = list()

    v = ['p', 't', 's', 'xt']

    if use_cm:
        v.append('cm')
    if use_mc:
        v.append('mc')

    v_p = ['p17', 'p18', 'p19', 'p20']

    # First, key byte nodes
    k_index = '-K'
    for i in range(1, 5):
        variables.append('k' + str(i) + k_index)

    # Now, loop and get set of nodes for each trace
    for t in range(number_of_traces):
        t_index = '-' + str(t)

        for var in v:
            for i in range(1, 5):
                variables.append(var + str(i) + t_index)
        # for i in range(1,4):
        for i in range(h_case, 4):
            variables.append('h' + str(i) + t_index)
        for i in range(17, 21):
            variables.append('p' + str(i) + t_index)

        for var in variables:
            if string_contains(var, 'k'):
                pass
            elif string_contains(var, 'p1-') or string_contains(var, 'p2-') or string_contains(var,
                                                                                               'p3-') or string_contains(
                var, 'p4-'):
                pass
            elif string_contains(var, 's'):
                factors.append('_Sbox_' + var)
            elif string_contains(var, 'xt'):
                if use_mc:
                    factors.append('_Xtimes_' + var)
                else:
                    factors.append('_XorXtimes_' + var)
            else:
                factors.append('_Xor_' + var)

        # edges forward
        for var in variables:
            # find factor that matches
            for fac in factors:
                if string_contains(fac, var):
                    edges.append((var, fac))
                    break

        # manual edges backwards
        for i in range(1, 5):
            i_index = str(i)
            index = i_index + t_index

            edges.append(('k' + i_index + k_index, '_Xor_t' + index))
            edges.append(('p' + index, '_Xor_t' + index))

            edges.append(('t' + index, '_Sbox_s' + index))

            if use_mc:
                edges.append(('s' + index, '_Xor_mc' + index))
                edges.append(('mc' + index, '_Xtimes_xt' + index))
            else:
                edges.append(('s' + index, '_XorXtimes_xt' + index))

            if use_cm:
                edges.append(('s' + index, '_Xor_cm' + index))
                edges.append(('xt' + index, '_Xor_cm' + index))
                edges.append(('cm' + index, '_Xor_p' + str(i + 16) + t_index))
            else:
                edges.append(('s' + index, '_Xor_p' + str(i + 16) + t_index))
                edges.append(('xt' + index, '_Xor_p' + str(i + 16) + t_index))

        for i in range(17, 21):
            i_index = str(i)
            index = i_index + t_index
            h3_var = 'h3' + t_index
            edges.append((h3_var, '_Xor_p' + index))

        if h_case == 1:

            for i in range(1, 3):
                i_index1 = str(i)
                i_index2 = str(i + 1)
                index1 = i_index1 + t_index
                index2 = i_index2 + t_index
                edges.append(('h' + index1, '_Xor_h' + index2))

            # ALL 3 h NODES
            edges.append(('s1' + t_index, '_Xor_h1' + t_index))
            edges.append(('s2' + t_index, '_Xor_h1' + t_index))
            edges.append(('s3' + t_index, '_Xor_h2' + t_index))
            edges.append(('s4' + t_index, '_Xor_h3' + t_index))

        elif h_case == 2:

            for i in range(2, 3):
                i_index1 = str(i)
                i_index2 = str(i + 1)
                index1 = i_index1 + t_index
                index2 = i_index2 + t_index
                edges.append(('h' + index1, '_Xor_h' + index2))

            # ONLY h2 AND h3 (no h1)
            edges.append(('s1' + t_index, '_Xor_h2' + t_index))
            edges.append(('s2' + t_index, '_Xor_h2' + t_index))
            edges.append(('s3' + t_index, '_Xor_h2' + t_index))
            edges.append(('s4' + t_index, '_Xor_h3' + t_index))

        elif h_case == 3:

            # ONLY h3
            edges.append(('s1' + t_index, '_Xor_h3' + t_index))
            edges.append(('s2' + t_index, '_Xor_h3' + t_index))
            edges.append(('s3' + t_index, '_Xor_h3' + t_index))
            edges.append(('s4' + t_index, '_Xor_h3' + t_index))

        # s -> mc
        if use_mc:
            edges.append(('s1' + t_index, '_Xor_mc4' + t_index))
            edges.append(('s2' + t_index, '_Xor_mc1' + t_index))
            edges.append(('s3' + t_index, '_Xor_mc2' + t_index))
            edges.append(('s4' + t_index, '_Xor_mc3' + t_index))
        else:
            edges.append(('s1' + t_index, '_XorXtimes_xt4' + t_index))
            edges.append(('s2' + t_index, '_XorXtimes_xt1' + t_index))
            edges.append(('s3' + t_index, '_XorXtimes_xt2' + t_index))
            edges.append(('s4' + t_index, '_XorXtimes_xt3' + t_index))

    # Add to graph

    # print "Factors ({}): {}\n\nVariables: {}\n\n".format(len(factors), factors, variables)
    # print "Edges: {}".format(edges)

    factor_graph.add_nodes_from(variables, bipartite=0)
    factor_graph.add_nodes_from(factors, bipartite=1)
    factor_graph.add_edges_from(edges)

    # Final amendments
    # for t in range (number_of_traces):
    #     for i in range(1,5):
    #         factor_graph.node[str(t) + '-' + '_SBOX' + str(i)]['in'] = str(t) + '-' + 't' + str(i)
    #         factor_graph.node[str(t) + '-' + '_XTIMES' + str(i)]['in'] = str(t) + '-' + 'mc' + str(i)

    try:
        nx.write_gexf(factor_graph,
                      'graphs/{}_trace_first_round_graph_hcase-{}_cm-{}_mc-{}.graph'.format(number_of_traces, h_case, use_cm,
                                                                                      use_mc))
        # print "{} Trace Graph Done!".format(number_of_traces)
    except IOError:
        print "IOError Encountered when writing Factor Graph!"
        raise


def create_factor_graph_full_aes(number_of_traces, removed_nodes=None, key_scheduling=False):

    # TODO
    print "!!! Need to implement for ARM AES in graph creator!"
    raise Exception

    if removed_nodes is None:
        removed_nodes = []
    factor_graph = nx.Graph()

    edges = list()
    variables = list()
    factors = list()

    variables_without_factors = list()

    # Now, loop and get set of nodes for each trace
    for t in range(number_of_traces):
        t_index = '-' + str(t)

        # Cover all 10 Rounds
        for i in range(1, 177):
            i_index = pad_string_zeros(str(i), 3)

            # Plaintext Bytes
            variables.append('p{}{}'.format(i_index, t_index))

            if i > 16:
                # factors.append('_Xor_p{}{}'.format(i_index, t_index))
                # Key Bytes
                variables.append('k{}{}'.format(i_index, t_index))
                if not key_scheduling:
                    variables_without_factors.append('k{}{}'.format(i_index, t_index))
            else:
                variables.append('k{}-K'.format(i_index))
                variables_without_factors.append('k{}-K'.format(i_index))
                variables_without_factors.append('p{}{}'.format(i_index, t_index))

            if i <= 160:

                # t and s
                variables.append('t{}{}'.format(i_index, t_index))
                variables.append('s{}{}'.format(i_index, t_index))
                # factors.append('_Xor_t{}{}'.format(i_index, t_index))
                # factors.append('_Sbox_s{}{}'.format(i_index, t_index))

                if i <= 144:

                    # xt, cm, xa, xb
                    variables.append('xt{}{}'.format(i_index, t_index))
                    variables.append('cm{}{}'.format(i_index, t_index))
                    variables.append('xa{}{}'.format(i_index, t_index))
                    variables.append('xb{}{}'.format(i_index, t_index))

                    if key_scheduling and (i <= 40):

                        # sk
                        variables.append('sk{}{}'.format(i_index, t_index))

                        if i <= 11:

                            # rc
                            variables.append('rc{}{}'.format(i_index, t_index))

                            if i <= 10:

                                # xk
                                variables.append('xk{}{}'.format(i_index, t_index))

                                if i == 1:
                                    # rc1 doesn't have a parent factor node
                                    variables_without_factors.append('rc{}{}'.format(i_index, t_index))

    # Factors: do all
    for var in variables:
        if var not in variables_without_factors:

            # Add factor node according to what variable node is
            if string_starts_with(var, 's'):
                # SBOX
                factors.append('_Sbox_{}'.format(var))
            elif string_starts_with(var, 'xt') or string_starts_with(var, 'rc'):
                # XTIMES
                factors.append('_Xtimes_{}'.format(var))
            else:
                # XOR
                factors.append('_Xor_{}'.format(var))

    # Edges forward (Factor -> Variable created by Factor Operation)
    for fac in factors:
        # find variable that matches
        for var in variables:
            if string_contains(fac, '_' + var):
                edges.append((var, fac))
                break

    # All other edges
    # Standard
    for t in range(number_of_traces):

        t_index = '-' + str(t)

        # Cover all 10 Rounds
        for i in range(1, 177):

            i_index = pad_string_zeros(str(i), 3)
            index = i_index + t_index

            # Non-key scheduling
            if i <= 160:

                i_index2 = pad_string_zeros(str(i + 16), 3)

                edges.append(('p{}'.format(index), '_Xor_t{}'.format(index)))
                edges.append(('t{}'.format(index), '_Sbox_s{}'.format(index)))

                if i <= 16:
                    edges.append(('k{}-K'.format(i_index), '_Xor_t{}'.format(index)))
                else:
                    edges.append(('k{}'.format(index), '_Xor_t{}'.format(index)))

                if i <= 144:
                    edges.append(('xt{}'.format(index), '_Xor_cm{}'.format(index)))

                    #  xtN = Xtimes(shift[N%16])
                    which_round = get_round_from_number(i)
                    shift_number = shift[my_mod(i, 16)]
                    s_index = pad_string_zeros(str(((get_round_from_number(i) - 1) * 16) + shift_number), 3)

                    edges.append(('s{}{}'.format(s_index, t_index), '_Xtimes_xt{}'.format(index)))
                    edges.append(('s{}{}'.format(s_index, t_index), '_Xor_cm{}'.format(index)))

                    edges.append(('xa{}'.format(index), '_Xor_xb{}'.format(index)))
                    edges.append(('xt{}'.format(index), '_Xor_xb{}'.format(index)))

                    edges.append(('xb{}'.format(index), '_Xor_p{}{}'.format(i_index2, t_index)))

            else:

                edges.append(('k{}'.format(index), '_Xor_p{}'.format(index)))

                shift_number = pad_string_zeros(str(shift[i - 160] + 144))

                edges.append(('s{}{}'.format(shift_number, t_index), '_Xor_p{}'.format(index)))

        # Non-Standard

        # Mix Columns
        for i in range(1, 145, 16):
            # i = 1, 17, 33, ...
            i_index = pad_string_zeros(str(i))
            which_round = get_round_from_number(i)

            # Each column
            for j in range(i, i + 16, 4):
                # j = 1, 5, 9, 13 (first column)
                shift1 = pad_string_zeros(str(shift[my_mod(j)] + ((which_round - 1) * 16)))
                shift2 = pad_string_zeros(str(shift[my_mod(j + 1)] + ((which_round - 1) * 16)))
                shift3 = pad_string_zeros(str(shift[my_mod(j + 2)] + ((which_round - 1) * 16)))
                shift4 = pad_string_zeros(str(shift[my_mod(j + 3)] + ((which_round - 1) * 16)))

                # xa1
                edges.append(
                    ('s{}{}'.format(shift3, t_index), '_Xor_xa{}{}'.format(pad_string_zeros(str(j)), t_index)))
                edges.append(
                    ('s{}{}'.format(shift4, t_index), '_Xor_xa{}{}'.format(pad_string_zeros(str(j)), t_index)))

                # xa2
                edges.append(
                    ('s{}{}'.format(shift1, t_index), '_Xor_xa{}{}'.format(pad_string_zeros(str(j + 1)), t_index)))
                edges.append(
                    ('s{}{}'.format(shift4, t_index), '_Xor_xa{}{}'.format(pad_string_zeros(str(j + 1)), t_index)))

                # xa3
                edges.append(
                    ('s{}{}'.format(shift1, t_index), '_Xor_xa{}{}'.format(pad_string_zeros(str(j + 2)), t_index)))
                edges.append(
                    ('s{}{}'.format(shift2, t_index), '_Xor_xa{}{}'.format(pad_string_zeros(str(j + 2)), t_index)))

                # xa4
                edges.append(
                    ('s{}{}'.format(shift2, t_index), '_Xor_xa{}{}'.format(pad_string_zeros(str(j + 3)), t_index)))
                edges.append(
                    ('s{}{}'.format(shift3, t_index), '_Xor_xa{}{}'.format(pad_string_zeros(str(j + 3)), t_index)))

                # cm -> Xor_p
                edges.append(('cm{}{}'.format(pad_string_zeros(str(j + 1)), t_index),
                              '_Xor_p{}{}'.format(pad_string_zeros(str(j + 16)), t_index)))
                edges.append(('cm{}{}'.format(pad_string_zeros(str(j + 2)), t_index),
                              '_Xor_p{}{}'.format(pad_string_zeros(str(j + 17)), t_index)))
                edges.append(('cm{}{}'.format(pad_string_zeros(str(j + 3)), t_index),
                              '_Xor_p{}{}'.format(pad_string_zeros(str(j + 18)), t_index)))
                edges.append(('cm{}{}'.format(pad_string_zeros(str(j)), t_index),
                              '_Xor_p{}{}'.format(pad_string_zeros(str(j + 19)), t_index)))

        # Key scheduling
        if key_scheduling:

            # rc
            for i in range(1, 11):
                i_index1 = pad_string_zeros(str(i))
                i_index2 = pad_string_zeros(str(i + 1))
                edges.append(('rc{}{}'.format(i_index1, t_index), '_Xtimes_rc{}{}'.format(i_index2, t_index)))

            for i in range(0, 160, 16):
                i_index = pad_string_zeros(str(i))
                # REMEMBER TO +1 FOR STRINGS

                # sk
                edges.append(('k{}{}'.format(pad_string_zeros(str(i + 14)), t_index),
                              '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 1)), t_index)))
                edges.append(('k{}{}'.format(pad_string_zeros(str(i + 15)), t_index),
                              '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 2)), t_index)))
                edges.append(('k{}{}'.format(pad_string_zeros(str(i + 16)), t_index),
                              '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 3)), t_index)))
                edges.append(('k{}{}'.format(pad_string_zeros(str(i + 13)), t_index),
                              '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 4)), t_index)))

                # XOR with rc
                edges.append(('sk{}{}'.format(pad_string_zeros(str((i / 4) + 1)), t_index),
                              '_Xor_xk{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index)))

                if i < 16:
                    edges.append(('k{}-K'.format(pad_string_zeros(str(i + 1))),
                                  '_Xor_xk{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index)))
                else:
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 1)), t_index),
                                  '_Xor_xk{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index)))

                # First
                edges.append(('xk{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index),
                              '_Xor_k{}{}'.format(pad_string_zeros(str(i + 17)), t_index)))
                edges.append(('rc{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index),
                              '_Xor_k{}{}'.format(pad_string_zeros(str(i + 17)), t_index)))

                # Others
                edges.append(('sk{}{}'.format(pad_string_zeros(str((i / 4) + 2)), t_index),
                              '_Xor_k{}{}'.format(pad_string_zeros(str(i + 18)), t_index)))
                edges.append(('sk{}{}'.format(pad_string_zeros(str((i / 4) + 3)), t_index),
                              '_Xor_k{}{}'.format(pad_string_zeros(str(i + 19)), t_index)))
                edges.append(('sk{}{}'.format(pad_string_zeros(str((i / 4) + 4)), t_index),
                              '_Xor_k{}{}'.format(pad_string_zeros(str(i + 20)), t_index)))

                if i < 16:
                    edges.append(('k{}-K'.format(pad_string_zeros(str(i + 2))),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 18)), t_index)))
                    edges.append(('k{}-K'.format(pad_string_zeros(str(i + 3))),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 19)), t_index)))
                    edges.append(('k{}-K'.format(pad_string_zeros(str(i + 4))),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 20)), t_index)))
                else:
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 2)), t_index),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 18)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 3)), t_index),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 19)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 4)), t_index),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 20)), t_index)))

                # Loop the rest
                for j in range(i, i + 12):
                    # k[j+20] = k[j+4] ^ k[j+16];
                    if (j + 5) < 16:
                        edges.append(('k{}-K'.format(pad_string_zeros(str(j + 5))),
                                      '_Xor_k{}{}'.format(pad_string_zeros(str(j + 21)), t_index)))
                    else:
                        edges.append(('k{}{}'.format(pad_string_zeros(str(j + 5)), t_index),
                                      '_Xor_k{}{}'.format(pad_string_zeros(str(j + 21)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(j + 17)), t_index),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(j + 21)), t_index)))

    # Removal of nodes? Start with nodes that have a degree of 2 (temporary, only used for intermediate computation)
    # XOR - XOR:    cm, xa, xb
    # XOR - SBOX:
    # XOR - XTIMES:

    # For each variable that we want to remove:
    # Get list of all variables that match the target name (e.g. cm -> [cm001-0, cm002-0, etc])
    # Find parent factor node (contains the name of the var)
    # Find child factor node (doesn't contain name of char, but for now the only other factor node that edges with the target variable)
    # Get all edges coming in to the parent factor node (list of variables)
    # For each of these variables:
    #   Remove the edge going to parent factor node
    #   Add a new edge going to child factor node
    # Finally, remove both target variable node and parent factor node

    # For each variable that we want to remove:
    for r_node in removed_nodes:

        print "+=+ Removing {} +=+".format(r_node)

        # Get list of all variables that match the target name (e.g. cm -> [cm001-0, cm002-0, etc])
        r_node_all = get_variables_that_match(variables, r_node)

        # Sanity check: Make sure list isn't empty
        if len(r_node_all) == 0:
            print "* Error: {} isn't a valid variable!".format(r_node)

        else:
            print "* Attempting to remove {} variables".format(len(r_node_all))

            # For each of these nodes...
            for current_node in r_node_all:

                # Get Parent and Child Factor Nodes
                parent = None
                # parent_index = None
                child = None
                # child_index = None

                for edge in edges:

                    if current_node == edge[0]:
                        # Edge found. Check if parent or child
                        if string_ends_with(edge[1], current_node):
                            parent = edge[1]
                        else:
                            child = edge[1]
                        # Break if done
                        if parent is not None and child is not None:
                            # Remove both edges
                            edges.remove((current_node, parent))
                            edges.remove((current_node, child))
                            break
                else:
                    print "* Warning: Variable {} has parent {} and child {}. Cannot remove!".format(current_node,
                                                                                                     parent, child)
                    pass

                # Only go on if both parent and child are present
                if parent and child:

                    # print "Current Node {}: Rerouting edges from {} to {}...".format(current_node, parent, child)

                    # Handle XOR - XOR (easiest)
                    if is_xor_node(parent) and is_xor_node(child):

                        # For each incoming edge to parent, delete it and add a new edge going to child
                        for edge in list(edges):

                            if parent == edge[1]:
                                # Edge Found
                                # Add new edge
                                edges.append((edge[0], child))
                                # Delete old edge
                                edges.remove((edge[0], parent))

                        # Finally, delete the parent and the target variable node
                        factors.remove(parent)
                        variables.remove(current_node)


                    else:
                        print "* Warning: Parent {} and Child {} not both XOR nodes. Cannot merge! (Work in Progress)".format(
                            parent, child)

                    # Get all edges coming in to the parent factor node (list of variables)

    # Make sure no unconnected nodes
    check_unconnected_nodes(variables, factors, edges)

    # Check for erroneous edges
    check_variable_factor_edge_match(variables, factors, edges)

    # Remove duplicates
    variables = set(variables)
    factors = set(factors)
    edges = set(edges)

    # Add to graph
    factor_graph.add_nodes_from(variables, bipartite=0)
    factor_graph.add_nodes_from(factors, bipartite=1)
    factor_graph.add_edges_from(edges)



    # Print number of variables / traces / edges
    print "Variables: {}, Factors: {}, Edges: {}".format(len(variables), len(factors), len(edges))

    # Save to file
    try:
        nx.write_gexf(factor_graph, 'graphs/{}_trace_fullAES_removednodes-{}_keysched-{}.graph'.format(number_of_traces,
                                                                                                 list_to_file_string(
                                                                                                     removed_nodes),
                                                                                                 key_scheduling))
        # print "{} Trace Graph Done!".format(number_of_traces)
    except IOError:
        print "IOError Encountered when writing Factor Graph!"
        raise


def create_factor_graph_full_aes_furious(number_of_traces, removed_nodes=None, key_scheduling=False, rounds_of_aes = 10,
                                         remove_cycle=False):


    if removed_nodes is None:
        removed_nodes = []

    factor_graph = nx.Graph()

    edges = list()
    variables = list()
    factors = list()

    variables_without_factors = list()

    variables_with_xor_constant = list()

    # Now, loop and get set of nodes for each trace
    for t in range(number_of_traces):
        t_index = '-' + str(t)

        # Cover all 10 Rounds
        for i in range(1, 177):
            i_index = pad_string_zeros(str(i), 3)

            # Plaintext Bytes
            variables.append('p{}{}'.format(i_index, t_index))

            if i > 16:
                # factors.append('_Xor_p{}{}'.format(i_index, t_index))
                # Key Bytes
                variables.append('k{}{}'.format(i_index, t_index))
                if not key_scheduling:
                    variables_without_factors.append('k{}{}'.format(i_index, t_index))
                elif ((i % 16) == 1) and (i > 1):
                    variables_with_xor_constant.append('k{}{}'.format(i_index, t_index))

            else:
                variables.append('k{}-K'.format(i_index))
                variables_without_factors.append('k{}-K'.format(i_index))
                variables_without_factors.append('p{}{}'.format(i_index, t_index))

            if i <= 160:

                # t and s
                variables.append('t{}{}'.format(i_index, t_index))
                variables.append('s{}{}'.format(i_index, t_index))
                # factors.append('_Xor_t{}{}'.format(i_index, t_index))
                # factors.append('_Sbox_s{}{}'.format(i_index, t_index))

                if i <= 144:

                    # xt, cm, xa, xb
                    variables.append('xt{}{}'.format(i_index, t_index))
                    variables.append('cm{}{}'.format(i_index, t_index))
                    variables.append('mc{}{}'.format(i_index, t_index))

                    if i <= 108:

                        # h
                        variables.append('h{}{}'.format(i_index, t_index))

                        if key_scheduling and (i <= 40):

                            # sk
                            variables.append('sk{}{}'.format(i_index, t_index))

                            if i <= 10:
                                # xk
                                variables.append('xk{}{}'.format(i_index, t_index))

    # Factors: do all
    for var in variables:
        if var in variables_with_xor_constant:
            # k017, k033, etc. (replaces rc)
            factors.append('_XorConstant_{}'.format(var))
        elif var not in variables_without_factors:
            # Add factor node according to what variable node is
            if string_starts_with(var, 's'):
                # SBOX
                factors.append('_Sbox_{}'.format(var))
            elif string_starts_with(var, 'xt'):
                # XTIMES
                factors.append('_Xtimes_{}'.format(var))
            else:
                # XOR
                factors.append('_Xor_{}'.format(var))

    # Edges forward (Factor -> Variable created by Factor Operation)
    for fac in factors:
        # find variable that matches
        for var in variables:
            if string_ends_with(fac, '_' + var):
                edges.append((var, fac))
                break

    # All other edges
    # Standard
    for t in range(number_of_traces):

        t_index = '-' + str(t)

        # Cover all 10 Rounds
        for i in range(1, 177):

            i_index = pad_string_zeros(str(i), 3)
            index = i_index + t_index

            # Non-key scheduling
            if i <= 160:

                i_index2 = pad_string_zeros(str(i + 16), 3)

                edges.append(('p{}'.format(index), '_Xor_t{}'.format(index)))
                edges.append(('t{}'.format(index), '_Sbox_s{}'.format(index)))

                if i <= 16:
                    edges.append(('k{}-K'.format(i_index), '_Xor_t{}'.format(index)))
                else:
                    edges.append(('k{}'.format(index), '_Xor_t{}'.format(index)))

                if i <= 144:
                    edges.append(('xt{}'.format(index), '_Xor_cm{}'.format(index)))

                    #  xtN = Xtimes(shift[N%16])
                    which_round = get_round_from_number(i)
                    shift_number = shift[my_mod(i, 16)]
                    s_index = pad_string_zeros(str(((get_round_from_number(i) - 1) * 16) + shift_number), 3)

                    edges.append(('s{}{}'.format(s_index, t_index), '_Xor_mc{}'.format(index)))
                    edges.append(('s{}{}'.format(s_index, t_index), '_Xor_cm{}'.format(index)))

                    edges.append(('mc{}'.format(index), '_Xtimes_xt{}'.format(index)))

                    edges.append(('xt{}'.format(index), '_Xor_cm{}'.format(index)))

                    edges.append(('cm{}'.format(index), '_Xor_p{}{}'.format(i_index2, t_index)))

            else:

                edges.append(('k{}'.format(index), '_Xor_p{}'.format(index)))

                shift_number = pad_string_zeros(str(shift[i - 160] + 144))

                edges.append(('s{}{}'.format(shift_number, t_index), '_Xor_p{}'.format(index)))

        # Non-Standard

        # Mix Columns
        for i in range(1, 145, 16):
            # i = 1, 17, 33, ...
            i_index = pad_string_zeros(str(i))
            which_round = get_round_from_number(i)

            # Each column
            for j in range(i, i + 16, 4):
                # j = 1, 5, 9, 13 (first column)
                shift1 = pad_string_zeros(str(shift[my_mod(j)] + ((which_round - 1) * 16)))
                shift2 = pad_string_zeros(str(shift[my_mod(j + 1)] + ((which_round - 1) * 16)))
                shift3 = pad_string_zeros(str(shift[my_mod(j + 2)] + ((which_round - 1) * 16)))
                shift4 = pad_string_zeros(str(shift[my_mod(j + 3)] + ((which_round - 1) * 16)))
                # h1 index
                indexh1 = pad_string_zeros(str((((j - 1) / 4) * 3) + 1))
                indexh2 = pad_string_zeros(str((((j - 1) / 4) * 3) + 2))
                indexh3 = pad_string_zeros(str((((j - 1) / 4) * 3) + 3))

                # mc1
                edges.append(
                    ('s{}{}'.format(shift2, t_index), '_Xor_mc{}{}'.format(pad_string_zeros(str(j)), t_index)))

                # mc2
                edges.append(
                    ('s{}{}'.format(shift3, t_index), '_Xor_mc{}{}'.format(pad_string_zeros(str(j + 1)), t_index)))

                # mc3
                edges.append(
                    ('s{}{}'.format(shift4, t_index), '_Xor_mc{}{}'.format(pad_string_zeros(str(j + 2)), t_index)))

                # mc4
                edges.append(
                    ('s{}{}'.format(shift1, t_index), '_Xor_mc{}{}'.format(pad_string_zeros(str(j + 3)), t_index)))

                # h1
                edges.append(('s{}{}'.format(shift1, t_index), '_Xor_h{}{}'.format(indexh1, t_index)))
                edges.append(('s{}{}'.format(shift2, t_index), '_Xor_h{}{}'.format(indexh1, t_index)))

                edges.append(('h{}{}'.format(indexh1, t_index), '_Xor_h{}{}'.format(indexh2, t_index)))

                # h2
                edges.append(('s{}{}'.format(shift3, t_index), '_Xor_h{}{}'.format(indexh2, t_index)))

                edges.append(('h{}{}'.format(indexh2, t_index), '_Xor_h{}{}'.format(indexh3, t_index)))

                # h3
                edges.append(('s{}{}'.format(shift4, t_index), '_Xor_h{}{}'.format(indexh3, t_index)))

                edges.append(
                    ('h{}{}'.format(indexh3, t_index), '_Xor_p{}{}'.format(pad_string_zeros(str(j + 16)), t_index)))
                edges.append(
                    ('h{}{}'.format(indexh3, t_index), '_Xor_p{}{}'.format(pad_string_zeros(str(j + 17)), t_index)))
                edges.append(
                    ('h{}{}'.format(indexh3, t_index), '_Xor_p{}{}'.format(pad_string_zeros(str(j + 18)), t_index)))
                edges.append(
                    ('h{}{}'.format(indexh3, t_index), '_Xor_p{}{}'.format(pad_string_zeros(str(j + 19)), t_index)))

        # Key scheduling
        if key_scheduling:

            for i in range(0, 160, 16):
                i_index = pad_string_zeros(str(i))
                # REMEMBER TO +1 FOR STRINGS

                # sk
                if i < 16:
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 14)), '-K'),
                                  '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 1)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 15)), '-K'),
                                  '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 2)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 16)), '-K'),
                                  '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 3)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 13)), '-K'),
                                  '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 4)), t_index)))
                else:
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 14)), t_index),
                                  '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 1)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 15)), t_index),
                                  '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 2)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 16)), t_index),
                                  '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 3)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 13)), t_index),
                                  '_Sbox_sk{}{}'.format(pad_string_zeros(str((i / 4) + 4)), t_index)))

                # XOR with rc
                edges.append(('sk{}{}'.format(pad_string_zeros(str((i / 4) + 1)), t_index),
                              '_Xor_xk{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index)))

                if i < 16:
                    edges.append(('k{}-K'.format(pad_string_zeros(str(i + 1))),
                                  '_Xor_xk{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index)))
                else:
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 1)), t_index),
                                  '_Xor_xk{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index)))

                # First
                edges.append(('xk{}{}'.format(pad_string_zeros(str((i / 16) + 1)), t_index),
                              '_XorConstant_k{}{}'.format(pad_string_zeros(str(i + 17)), t_index)))

                # Others
                edges.append(('sk{}{}'.format(pad_string_zeros(str((i / 4) + 2)), t_index),
                              '_Xor_k{}{}'.format(pad_string_zeros(str(i + 18)), t_index)))
                edges.append(('sk{}{}'.format(pad_string_zeros(str((i / 4) + 3)), t_index),
                              '_Xor_k{}{}'.format(pad_string_zeros(str(i + 19)), t_index)))
                edges.append(('sk{}{}'.format(pad_string_zeros(str((i / 4) + 4)), t_index),
                              '_Xor_k{}{}'.format(pad_string_zeros(str(i + 20)), t_index)))

                if i < 16:
                    edges.append(('k{}-K'.format(pad_string_zeros(str(i + 2))),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 18)), t_index)))
                    edges.append(('k{}-K'.format(pad_string_zeros(str(i + 3))),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 19)), t_index)))
                    edges.append(('k{}-K'.format(pad_string_zeros(str(i + 4))),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 20)), t_index)))
                else:
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 2)), t_index),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 18)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 3)), t_index),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 19)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(i + 4)), t_index),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(i + 20)), t_index)))

                # Loop the rest
                for j in range(i, i + 12):
                    # k[j+20] = k[j+4] ^ k[j+16];
                    if (j + 5) <= 16:
                        edges.append(('k{}-K'.format(pad_string_zeros(str(j + 5))),
                                      '_Xor_k{}{}'.format(pad_string_zeros(str(j + 21)), t_index)))
                    else:
                        edges.append(('k{}{}'.format(pad_string_zeros(str(j + 5)), t_index),
                                      '_Xor_k{}{}'.format(pad_string_zeros(str(j + 21)), t_index)))
                    edges.append(('k{}{}'.format(pad_string_zeros(str(j + 17)), t_index),
                                  '_Xor_k{}{}'.format(pad_string_zeros(str(j + 21)), t_index)))

    # Remove duplicates
    variables = list(set(variables))
    factors = list(set(factors))
    edges = list(set(edges))



    # Remove invalid nodes
    if rounds_of_aes < 10:
        edges_ = edges[:]
        factors_ = factors[:]
        variables_ = variables[:]
        for edge in edges_:
            if not check_factor_in_round(edge[1], rounds_of_aes):
                edges.remove(edge)
        for factor in factors_:
            if not check_factor_in_round(factor, rounds_of_aes):
                factors.remove(factor)
        for variable in variables_:
            if not check_variable_in_round(variable, rounds_of_aes):
                variables.remove(variable)

    if remove_cycle:
        # First, remove h1 and h2 nodes
        variables_to_merge_ = ['h' + pad_string_zeros(i) for i in range(1, 109) if ((i % 3) > 0)]
        variables_to_merge = match_nodes(variables, variables_to_merge_)
        for variable in variables_to_merge:
            variables, factors, edges = remove_node(variables, factors, edges, variable, 0)
        # Then, back factors for all mc nodes and h nodes
        variables_to_remove_parents = match_nodes(variables, ['mc', 'h'])
        for variable in variables_to_remove_parents:
            variables, factors, edges = remove_node(variables, factors, edges, variable, 2)

    # REMOVING NODES.
    # Start by getting actual list of nodes to remove:
    r_node_all = match_nodes(variables, removed_nodes)

    if len(r_node_all) > 0:

        print "* Attempting to remove {} nodes from the factor graph".format(len(r_node_all))

        # For each variable that we want to remove:
        for current_node in r_node_all:
            variables, factors, edges = remove_node(variables, factors, edges, current_node, REMOVING_VARIATION)

    # Make sure no unconnected nodes
    check_unconnected_nodes(variables, factors, edges)

    # Check all variables, factors, and edges match up properly
    check_variable_factor_edge_match(variables, factors, edges)

    # Add to graph
    factor_graph.add_nodes_from(variables, bipartite=0)
    factor_graph.add_nodes_from(factors, bipartite=1)
    factor_graph.add_edges_from(edges)


    # Print number of variables / traces / edges
    round_string = "{} ROUNDS ".format(rounds_of_aes)
    print "AES FURIOUS GRAPH {}{} Traces, Removed Nodes: {}, Key Scheduling {}".format(round_string,
                                                                                       number_of_traces, removed_nodes,
                                                                                       key_scheduling)
    print "Variables: {}, Factors: {}, Edges: {}".format(len(variables), len(factors), len(edges))

    # print "TEST: len(factor_graph.edges()): {}".format(len(factor_graph.edges()))
    # print "TEST: set of edges: {}".format(len(set(edges)))
    #
    # print "Connected: {}".format(nx.is_connected(factor_graph))
    #
    # bottom_nodes, top_nodes = bipartite.sets(factor_graph)
    #
    # print "TEST: top_nodes: {}".format(len(top_nodes))
    # print "TEST: bottom_nodes: {}".format(len(bottom_nodes))
    #
    # print sorted(bottom_nodes)
    # print_new_line()

    remove_cycle_string = ""
    if remove_cycle:
        remove_cycle_string = "RemovedCycle"
    round_string = "{}Rounds".format(rounds_of_aes)

    # Save to file
    try:
        nx.write_gexf(factor_graph,
                      'graphs/{}_trace_fullAESFurious{}{}_removednodes-{}_keysched-{}.graph'.format(number_of_traces,
                                                                                                rounds_of_aes,
                                                                                                remove_cycle_string,
                                                                                                list_to_file_string(
                                                                                                    removed_nodes),
                                                                                                key_scheduling))
        # print "{} Trace Graph Done!".format(number_of_traces)
    except IOError:
        print "IOError Encountered when writing Factor Graph!"
        raise

    if EDGES_TO_FILE:
        edge_string = ""
        for edge in edges:
            edge_string += edge[0] + ':' + edge[1] + ','
        f = open('graphs/AESFurious{}EdgeList.csv'.format(round_string), 'w+')
        f.write('{}'.format(edge_string))
        f.close()


if __name__ == "__main__":
    print "Hello World!"

    create_factor_graph_full_aes_furious(1)
