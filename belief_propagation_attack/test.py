import factorGraphAES as fG

print "Start!"

G = fG.FactorGraphAES()

print "Init!"

G.initialise_edges()

print "Edges!"

G.set_all_initial_distributions(seed=0, real_traces=True)

print "Set all Dists!"

# G.print_all_initial_distributions()

print "All done!"
