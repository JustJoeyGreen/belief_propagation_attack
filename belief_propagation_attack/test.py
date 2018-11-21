# import factorGraphAES as fG
#
# print "Start!"
#
# G = fG.FactorGraphAES()
#
# print "Init!"
#
# G.initialise_edges()
#
# print "Edges!"
#
# G.set_all_initial_distributions(seed=0, real_traces=True)
#
# print "Set all Dists!"
#
# # G.print_all_initial_distributions()
#
# print "All done!"

from utility import *

a = np.array([[1,2,3],[5,1,9],[5,2,8],[1,10,0],[-2,-7,20]])

b = normalise_neural_traces(a)

print b
print np.min(b, axis=1)
print np.max(b, axis=1)
