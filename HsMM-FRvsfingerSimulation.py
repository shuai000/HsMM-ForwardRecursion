"""
1. Applied for the HsMM forward recursion algorithm
2. 08/09/2020 Shuai Sun
This is an illustrative example of how HMM works to localization user region.
"""

from txtparse import TxtManager
from hmm import HiddenMarkov, HiddenSemiMarkov
import numpy as np
from demo import hmm_hsmm_comp
import matplotlib.pyplot as plt
import time

txt = TxtManager()
training_path = 'Data\_training'
trajecotry_path = 'Data\_testing'
# extract the training and testing data, the RSS data is generated from a ray-tracing software
trained_data = txt.wirelessinsite_path(training_path, s_num=12, offset=9, ap_num=8)
trajectory_data = txt.wirelessinsite_path(trajecotry_path, s_num=1, offset=1, ap_num=8)
time_length = trajectory_data[0].shape[0]

# the status was used to simulate data missing problem, it is not required here
# so we set the status as all 1 (not missing)
status = txt.miss_pattern_gen(8, [1, 1, 1, 1, 1, 1, 1, 1], time_length, seed_num=10)

D = 50  # predefine the maximum possible duration time
BBD_para = np.array([[.43, 5, D], [.35, 4.21, D], [.21, 2.25, D], [.46, 10, D], [.65, 8.88, D], [.53, 2.72, D],
                     [2, 3, D], [1, 40, D], [0.50, 5.53, D], [.65, 7.78, D], [.43, 6.49, D], [.43, 7.36, D]])

para_hsmm = {'state_num': 12, 'anchor_num': 8, 'initial_p': np.ones(12) / 12,
             'transition_p': np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [.68, 0, .32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, .29, 0, .3, 0, 0, .41, 0, 0, 0, 0, 0],
                                       [0, 0, .26, 0, .74, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, .71, 0, .29, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0.51, 0, 0, .49, 0, 0, 0, 0],
                                       [0, 0, .3, 0, 0, 0, 0, .31, .17, 0, 0, .21],
                                       [0, 0, 0, 0, 0, 0.48, .52, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, .46, 0, 0, .54, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 0, .53, 0, 0, 0, .47, 0]]),
             'training_type': 'not_provided', 'likelihood_from': 'not_provided',
             'max_duration': D,
             'sojourn': {'type': ['BBD' for item in range(12)], 'parameter': BBD_para}}

hsmm = HiddenSemiMarkov(para_hsmm)
mean_u, cov_u = hsmm.training(t_type='list', train_data=trained_data)
hsmm.set_gaussian(mean_u, cov_u)
# compute the likelihood (emission probability in the HsMM)
likelihood_value = hsmm.get_likelihood(trajectory_data[0], status)
hsmm.set_likelihood(likelihood_value)

start_time = time.time()
st_forward_hsmm = hsmm.forward_only_relax()[0]
end_time = time.time()

print("forward_hsmm time:", (end_time - start_time) / time_length)

st_max_likelihood = hsmm.maximum_likelihood(likelihood_value)

# This is the ground truth data, in each of the bracket is [state index, duration time]
st_ground = [[1, 8], [2, 18], [3, 23], [4, 36], [5, 46], [6, 62], [8, 78], [7, 120], [9, 129], [10, 138]]

hmm_hsmm_comp(
    [[st_max_likelihood, st_forward_hsmm]],
    [['Fingerprinting', 'HsMM-FR', 'Ground truth']], ground_truth=st_ground, fig_height=6, over_width=15, save_pdf=True)

plt.show()
