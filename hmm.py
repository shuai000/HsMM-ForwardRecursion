"""
1. Hidden Markov Based Module -- Include HMM and HsMM inherited from HMM
2. Ref: Lawrence tutorial on HMM & selected application in speech recognition
3. Author: Shuai Sun    05/09/2018
-----------------------------------------------------------------------------
Parameter needs to be specified once a HMM/HsMM model instance is initialized


"""
import numpy as np
import scipy.io as sio
import scipy.stats as stats
from math import log, exp, factorial
from scipy.special import comb
from scipy.special import beta as betafunction


class HiddenMarkov:

    def __init__(self, para_hmm, specification=False):
        """
        Called right after setting the HMM model
        Set the hidden Markov parameters
        :param para_hmm: dictionary format
        :return:
        """
        assert 'state_num' in para_hmm, " HMM state number is not provided!"
        self.Ns = para_hmm['state_num']
        self.Pi = np.ones(self.Ns) / self.Ns

        assert 'anchor_num' in para_hmm, " HMM anchor number is not provided!"
        self.Na = para_hmm['anchor_num']

        assert 'initial_p' in para_hmm, " HMM state initial probability is not provided!"
        self.Pi = para_hmm['initial_p']
        assert self.Pi.shape[0] == self.Ns, " Number of state doesn't match model initial probability size! "

        assert 'transition_p' in para_hmm, " HMM state transition probability is not provided!"
        self.Pt = para_hmm['transition_p']
        self.log_pt = np.zeros((self.Ns, self.Ns))
        for i in range(self.Ns):
            for j in range(self.Ns):
                if self.Pt[i, j] != 0:
                    self.log_pt[i, j] = log(self.Pt[i, j])
                else:
                    self.log_pt[i, j] = float('-inf')
        assert self.Pt.shape == (self.Ns, self.Ns), " Number of state doesn't match model transition probability"

        assert 'training_type' in para_hmm, " HMM training type is not provided!"
        # 1. from_loading 2. from_external 3. from_data
        training_from = para_hmm['training_type']
        if training_from == 'from_loading':
            self.training()  # load training parameter
        elif training_from == 'from_external':
            if 'mean' in para_hmm:
                self.mean = para_hmm['mean']
                assert self.mean.shape == (self.Na, self.Ns), \
                    " Number of state/observation don't match model mean size!"
            else:
                assert False, " NO mean parameter provided to the model!"
            if 'cov' in para_hmm:
                self.cov = para_hmm['cov']
                assert self.cov.shape == (self.Na, self.Na, self.Ns), \
                    "Number of state/observation don't match model cov size!"
            else:
                assert False, "NO COV parameter provided to the model!"
        elif training_from == 'not_provided':
            pass
        else:
            raise Exception('training_type is in wrong format: from_loading, from_external or not_provided!')

        assert 'likelihood_from' in para_hmm, "HMM observation filename is not provided!"
        if para_hmm['likelihood_from'] == 'from_external':  # filename in None means likelihood is provided outside
            assert 'likelihood' in para_hmm, "HMM observation likelihood is not provided"
            self.likelihood = para_hmm['likelihood']
            assert self.likelihood.shape[1] == self.Ns, "Number of state doesn't match likelihood data size"
        elif para_hmm['likelihood_from'] == 'from_file':
            assert 'filename' in para_hmm, "HMM evaluation data filename is not provided!"
            data, status = self.data_extraction(para_hmm['filename'], 0)
            self.likelihood = self.get_likelihood(data, status)
        elif para_hmm['likelihood_from'] == 'not_provided':
            pass
        else:
            raise Exception("Likelihood from is not in the library!")

        if specification:  # Print out the HMM model set up in the console
            self.model_specification(para_hmm)

    def set_likelihood(self, likelihood_value):
        self.likelihood = likelihood_value
        assert self.likelihood.shape[1] == self.Ns, "Number of state doesn't match likelihood data size"

    def set_gaussian(self, mu, cov):
        self.mean = mu
        self.cov = cov

        assert self.mean.shape == (self.Na, self.Ns), "Number of state and observation don't match model mean size!"
        assert self.cov.shape == (self.Na, self.Na, self.Ns), "Number of state/observation don't match model cov size!"

    def model_specification(self, para):
        print("\n\n-------- HIDDEN MARKOV MODEL ---------")
        print("State num: {}".format(self.Ns))
        if para['training_type'] == 'not_provided':
            print("Training data is not specified initially, provided later!")
        else:
            print("Training data is from: {}".format(para['training_type']))

        if para['likelihood_from'] == 'not_provided':
            print("Likelihood data is not specified initially, provided later!")
        else:
            print("Likelihood is from: {}".format(para['likelihood_from']))
        print("-------- HIDDEN MARKOV MODEL ---------\n")

    @staticmethod
    def match_error(st_truth, st_est, state_num, c_type='absolute'):
        """
        Function: statistically compute the matching error with different user-defined criteria
                  c_type: specify the criteria,
                          'absolute',      any types of error are not allowed
                          'lag_include',   lagging error is tolerated
        :return: correct_record: length of the total time index, if correct matched, set to 1, otherwise 0
                 error_matching: square matrix format, with diagonally zero, and other entries be the number
                 of wrong matches, where the truth is its row index, and the wrong decision is its column index
        """
        correct_record = np.zeros(len(st_truth))
        error_matching = np.zeros((state_num, state_num))

        if c_type == 'absolute':
            for i in range(len(st_truth)):
                if st_truth[i] == st_est[i]:
                    correct_record[i] = 1
        elif c_type == 'lag_include':
            previous_truth = 0
            current_truth = st_truth[0]
            for i in range(len(st_truth)):
                if current_truth != st_truth[i]:  # state change happens in ground truth
                    previous_truth = current_truth
                    current_truth = st_truth[i]
                if st_est[i] == current_truth or st_est[i] == previous_truth:
                    correct_record[i] = 1
        elif c_type == 'adjacent_include':
            pass
        else:
            raise Exception(
                'c_type should be one of the following: original, lag_include, adjacent_include; {}'.format(c_type),
                'is not defined')

        for i in range(len(st_truth)):
            if st_truth[i] != st_est[i]:
                error_matching[st_truth[i] - 1, st_est[i] - 1] += 1

        return correct_record, error_matching

    def forward_process(self, p_type='none', interval_index=None, prior=None):
        """
        Hidden Markov Model standard forward process
        func: Recursively compute forward variable
              alpha_t(i) = p(O_[1:t], q_t = S_i|lambda)
        :param p_type: propagation type: 'none', 'posterior', 'logform' (logform is not used, has to be approximated)
        :return: forward variable and estimation based on purely forward process
        """
        if interval_index is None:
            like_value = self.likelihood
        else:
            assert len(interval_index) == 2, "interval format wrong!"
            assert interval_index[1] > interval_index[0], "interval format wrong!"
            like_value = self.likelihood[interval_index[0]:interval_index[1] + 1, :]
        alpha = np.zeros((like_value.shape[0], self.Ns))

        # Forward Algorithm
        if p_type == 'none':
            for i_time in range(like_value.shape[0]):
                if i_time == 0:  # initialization
                    alpha[i_time, :] = self.Pi * like_value[i_time, :]  # Eq.(19)
                else:  # induction
                    alpha[i_time, :] = self.forward_propagation(alpha[i_time - 1, :]) \
                                       * like_value[i_time, :]  # Eq. (20)
            s_forward = self.state_estimation(alpha)
            return s_forward, alpha
        elif p_type == 'posterior':
            ct = np.zeros(like_value.shape[0])
            # initialization, if no pripr provided, then it is assumed as uniform
            if prior is None:
                alpha[0, :] = self.Pi * like_value[0, :]  # Eq.(19)
            else:
                alpha[0, :] = prior * like_value[0, :]
            ct[0] = sum(alpha[0, :])
            alpha[0, :] = alpha[0, :] / ct[0]
            # induction
            for i_time in range(1, like_value.shape[0]):
                alpha[i_time, :] = self.forward_propagation(alpha[i_time - 1, :]) \
                                   * like_value[i_time, :]  # Eq. (20)
                ct[i_time] = sum(alpha[i_time, :])
                alpha[i_time, :] = alpha[i_time, :] / ct[i_time]  # scaling p(si,o_1:t) -> p(si|o_1:t)
            log_likelihood = sum(np.log(ct))
            s_forward = self.state_estimation(alpha)
            return s_forward, alpha, log_likelihood, ct
        elif p_type == 'logform':
            for i_time in range(like_value.shape[0]):
                if i_time == 0:  # initialization
                    for i in range(self.Ns):
                        alpha[i_time, i] = np.log(self.Pi[i]) + np.log(like_value[i_time, i])
                else:
                    alpha[i_time, :] = self.forward_propagation(alpha[i_time - 1, :], p_type='log')
                    for i in range(self.Ns):
                        alpha[i_time, i] = alpha[i_time, i] + np.log(like_value[i_time, i])
            log_likelihood = 0
            for i in range(self.Ns):
                log_likelihood = log_likelihood + exp(alpha[-1, i])
            log_likelihood = log(log_likelihood)
            s_forward = self.state_estimation(alpha)
            return s_forward, alpha, log_likelihood

    @staticmethod
    def state_estimation(alpha):
        st = alpha.argmax(axis=1)
        st = st[:] + 1  # physically map to state index
        return st

    @staticmethod
    def maximum_likelihood(likelihood_value):
        st = likelihood_value.argmax(axis=1)
        st = st[:] + 1
        return st

    def forward_propagation(self, alpha_p, p_type='none'):

        if p_type == 'none':
            alpha_n = np.sum(alpha_p * self.Pt.T, axis=1)
        elif p_type == 'log':
            alpha_n = np.zeros(len(alpha_p))
            for j in range(self.Ns):
                for i in range(self.Ns):
                    alpha_n[j] = alpha_n[j] + exp(alpha_p[i] + np.log(self.Pt[i, j]))
                alpha_n[j] = np.log(alpha_n[j])
        return alpha_n

    def likelihood_com(self, data, status):
        """
        Function: compute likelihood based on multivariate Gaussian distribution
        :param data: RSS for all anchors, note that
        :param status: indicator of which anchor is active, the probability evaluation
                       will change size based on status's configuration
        :return:  likelihood probability, normalized to one
        """
        likelihood = np.zeros(self.Ns)
        if data.any():
            for i in range(self.Ns):
                mean_i = self.mean[status, i]
                cov_i = self.cov[:, :, i][status][:, status]  # take the sub-covariance
                likelihood[i] = stats.multivariate_normal.pdf(
                    data, mean_i, cov_i)
            norm_sum = likelihood.sum()
            likelihood = likelihood / norm_sum
        else:
            likelihood[:] = 1 / self.Ns
        return likelihood

    def get_likelihood(self, data, status=None):

        likelihood = np.zeros((data.shape[0], self.Ns))
        if status is None:
            status = np.ones((data.shape[0], self.Na))

        for i_time in range(data.shape[0]):
            tempstatus = np.where(status[i_time, :] == 1)[0]
            tempdata = data[i_time, :][tempstatus]

            # Following is due to requirement from stats library
            # maybe not efficient
            reformat_data = np.zeros((1, len(tempstatus)))
            reformat_data[0, :] = tempdata

            likelihood[i_time, :] = self.likelihood_com(reformat_data, tempstatus)

        return likelihood

    def training(self, t_type='load', train_data=[]):
        if t_type == 'load':
            self.mean = sio.loadmat('data/mean_x.mat')['mean_x']
            self.cov = sio.loadmat('data/cov_x.mat')['cov_x']
        elif t_type == 'list':  # I don't understand the meaning of list nowadays
            rx_num = train_data[0].shape[1]
            state_num = len(train_data)
            train_mean = np.zeros((rx_num, state_num))
            train_cov = np.zeros((rx_num, rx_num, state_num))
            for idx, val in enumerate(train_data):
                tmp_mean = val.mean(axis=0)
                tmp_error = val - tmp_mean
                tmp_cov = np.dot(tmp_error.T, tmp_error) / (val.shape[0] - 1)
                train_mean[:, idx] = tmp_mean
                train_cov[:, :, idx] = tmp_cov
            return train_mean, train_cov


class HiddenSemiMarkov(HiddenMarkov):

    def __init__(self, para_hsmm, specification=False):
        assert 'state_num' in para_hsmm, "HsMM state number is not provided!"
        self.Ns = para_hsmm['state_num']

        assert 'anchor_num' in para_hsmm, "HsMM anchor number is not provided!"
        self.Na = para_hsmm['anchor_num']

        assert 'max_duration' in para_hsmm, "HsMM maximum duration is not provided!"
        self.D = para_hsmm['max_duration']
        self.Pd = np.zeros((self.Ns, self.D))

        assert 'initial_p' in para_hsmm, "HsMM state initial probability is not provided!"
        self.Pi = para_hsmm['initial_p']
        assert self.Pi.shape[0] == self.Ns, "Number of state doesn't match model initial probability size! "

        assert 'transition_p' in para_hsmm, "HsMM state transition probability is not provided!"
        self.Pt = para_hsmm['transition_p']
        self.log_pt = np.zeros((self.Ns, self.Ns))
        for i in range(self.Ns):
            for j in range(self.Ns):
                if self.Pt[i, j] != 0:
                    self.log_pt[i, j] = log(self.Pt[i, j])
                else:
                    self.log_pt[i, j] = float('-inf')
        assert self.Pt.shape == (self.Ns, self.Ns), "Number of state doesn't match model transition probability"

        assert 'training_type' in para_hsmm, "HsMM training type is not provided!"
        # 1. from_loading 2. from_external 3. from_data
        training_from = para_hsmm['training_type']
        if training_from == 'from_loading':
            self.training()  # load training parameter
        elif training_from == 'from_external':
            if 'mean' in para_hsmm:
                self.mean = para_hsmm['mean']
                # assert self.mean.shape == (self.Na, self.Ns), \
                #     " Number of state/observation don't match model mean size!"
            else:
                assert False, " NO mean parameter provided to the model!"
            if 'cov' in para_hsmm:
                self.cov = para_hsmm['cov']
                # assert self.cov.shape == (self.Na, self.Na, self.Ns), \
                #     "Number of state/observation don't match model cov size!"
            else:
                assert False, "NO COV parameter provided to the model!"
        elif training_from == 'not_provided':
            pass
        else:
            raise Exception("training from is not from the library!")

        assert 'likelihood_from' in para_hsmm, "HMM observation filename is not provided!"
        if para_hsmm['likelihood_from'] == 'from_external':  # filename in None means likelihood is provided outside
            assert 'likelihood' in para_hsmm, "HMM observation likelihood is not provided"
            self.likelihood = para_hsmm['likelihood']
            assert self.likelihood.shape[1] == self.Ns, "Number of state doesn't match likelihood data size"
        elif para_hsmm['likelihood_from'] == 'from_file':
            assert 'filename' in para_hsmm, "HMM evaluation data filename is not provided!"
            data, status = self.data_extraction(para_hsmm['filename'], 0)
            self.likelihood = self.get_likelihood(data, status)
        elif para_hsmm['likelihood_from'] == 'not_provided':
            pass
        else:
            raise Exception("The likelihood_from data is not in right format!")

        if 'duration_p' in para_hsmm:  # in case an outside duration probability matrix is required
            self.Pd = para_hsmm['duration_p']

        assert 'sojourn' in para_hsmm, "HsMM sojourn parameter is not provided"
        if para_hsmm['sojourn'] is not None:
            self.sojourn_gen(para_hsmm['sojourn'])

        if specification:
            self.model_specification(para_hsmm)

    def set_durationMax(self, d_max):
        self.D = d_max

    def model_specification(self, para):
        print("\n-------- HIDDEN Semi-Markov MODEL ---------")
        print("State num: {}".format(self.Ns))
        print("The duration type for each state is:")
        for i in range(self.Ns):
            print("State {} ".format(i + 1), para['sojourn']['type'][i],
                  "with parameter: {}".format(para['sojourn']['parameter'][i]))
        if para['training_type'] == 'not_provided':
            print("Training data is not provided when initializing the model!")
        else:
            print("Training data is from: {}".format(para['training_type']))

        if para['likelihood_from'] == 'not_provided':
            print("Likelihood data is not provided when initializing the model!")
        else:
            print("Likelihood is from: {}".format(para['likelihood_from']))

        print("-------- HIDDEN Semi-Markov MODEL ---------\n\n")

    def forward_only_relax(self):
        """
        This is the original implementation of the HsMM-FR algorithm, scaling is not added
        See forward_only_relax_scaling for the practical implementation
        :return: st (state estimates), bar_alpha (forward variable)
        """
        like_value = self.likelihood
        alpha, alpha_star = self.forward_process(p_type='none')[0:2]
        bar_alpha = np.zeros(alpha.shape)
        for i_time in range(alpha.shape[0]):
            if i_time == 0:
                for i in range(self.Ns):
                    bar_alpha[i_time, i] = alpha_star[i_time, i] * (1 - self.Pd[i, 0]) * like_value[i_time, i]
            else:
                for i in range(self.Ns):
                    for d in range(min(i_time + 1, self.D)):
                        like_d = self.combined_likelihood(like_value, i_time, d)
                        bar_alpha[i_time, i] += alpha_star[i_time - d, i] * sum(self.Pd[i, d:self.D]) * like_d[i]

        st = self.state_estimation(bar_alpha)
        return st, bar_alpha

    def forward_only_relax_scaling(self):
        """
        For practical implementation of the HsMM-FR algorithm
        The bar_alpha is defined as state i is active not ends
        see Equations (17-21) in the paper
        :return: st (state estimates), bar_alpha (scaled forward variable), ct (scaling coefficient)
        """
        like_value = self.likelihood
        alpha = np.zeros(like_value.shape)
        alpha_star = np.zeros(like_value.shape)
        bar_alpha = np.zeros(like_value.shape)

        ct = np.ones(like_value.shape[0])
        # initialization
        alpha_star[0, :] = self.Pi
        alpha[0, :], ct[0] = self.forward_semi_propagation(alpha_star, like_value, 0,
                                                               p_type='scale', scaling_coefficent=ct)
        bar_alpha[0, :] = self.forward_semi_propagation_relax(alpha_star, like_value, 0, ct)

        for i_time in range(1, like_value.shape[0]):
            # step 1: prediction
            alpha_star[i_time, :] = self.forward_propagation(alpha[i_time - 1, :])
            # step 2: update
            alpha[i_time, :], ct[i_time] = self.forward_semi_propagation(alpha_star, like_value, i_time,
                                                                         p_type='scale', scaling_coefficent=ct)
            bar_alpha[i_time, :] = self.forward_semi_propagation_relax(alpha_star, like_value, i_time, ct)

        st = self.state_estimation(bar_alpha)

        return st, bar_alpha, ct

    def forward_process(self, p_type='none', interval_index=None, prior=None):
        """
        func: rewrited to conduct forward process in hidden Semi-Markov model
        :param filename: txt file that contains the data
        :param p_type: define the propagation type
        :param interval_index: evaluation interval
        :param prior: prior information for the forward propagation, if None, uniform is assumed
        :return:
        """
        if interval_index is None:
            like_value = self.likelihood
        else:
            assert len(interval_index) == 2, "interval format wrong!"
            assert interval_index[1] > interval_index[0], "interval format wrong!"
            like_value = self.likelihood[interval_index[0]:interval_index[1] + 1, :]

        alpha = np.zeros((like_value.shape[0], self.Ns))
        alpha_star = np.zeros((like_value.shape[0], self.Ns))
        # Forward Algorithm
        if p_type == 'none':
            for i_time in range(like_value.shape[0]):
                if i_time == 0:  # initialization
                    alpha_star[i_time, :] = self.Pi
                    alpha[i_time, :] = self.forward_semi_propagation(alpha_star, like_value, i_time)
                else:  # induction
                    # step 1: prediction
                    alpha_star[i_time, :] = self.forward_propagation(alpha[i_time - 1, :])
                    # step 2: update
                    alpha[i_time, :] = self.forward_semi_propagation(alpha_star, like_value, i_time)
            likelihood = sum(alpha[-1, :])
            st = self.state_estimation(alpha)
            return alpha, alpha_star, st, likelihood

        elif p_type == 'posterior':
            ct = np.ones(like_value.shape[0])
            # initialization
            if prior is None:
                alpha_star[0, :] = self.Pi
            else:
                alpha_star[0, :] = prior

            alpha[0, :], ct[0] = self.forward_semi_propagation(alpha_star, like_value, 0,
                                                               p_type='scale', scaling_coefficent=ct)

            for i_time in range(1, like_value.shape[0]):
                # step 1: prediction
                alpha_star[i_time, :] = self.forward_propagation(alpha[i_time - 1, :])
                # step 2: update
                alpha[i_time, :], ct[i_time] = self.forward_semi_propagation(alpha_star, like_value, i_time,
                                                                             p_type='scale', scaling_coefficent=ct)
            # likelihood = -1 * self.hmm_log(ct.prod())
            st = self.state_estimation(alpha)
            return alpha, alpha_star, st, ct

    def forward_semi_propagation_relax(self, alpha_star, like_value, i_time, scaling_coefficient):
        """
        func: Recursively compute "active" forward variable -- alpha
        :param alpha_star: sub forward variable
        :param like_value: likelihood matrix (needs to be reevaluated)
        :param i_time: current time instance
        :param scaling_coefficient: scaling factors array
        """
        alpha = np.zeros(self.Ns)
        d_max = min(self.D, i_time + 1)

        for i in range(self.Ns):
            for d in range(d_max):
                like_d = self.combined_likelihood(like_value, i_time, d)
                scaling = 1
                for k in range(d):
                    scaling = scaling * scaling_coefficient[i_time - 1 - k]
                alpha[i] = alpha[i] + alpha_star[i_time - d, i] * sum(self.Pd[i, d:self.D]) * like_d[i] * scaling
        alpha = alpha * scaling_coefficient[i_time]
        return alpha

    def forward_semi_propagation(self, alpha_star, like_value, i_time, p_type='none', scaling_coefficent=[]):
        """
        func: Recursively compute forward variable -- alpha
        :param alpha_star: sub forward variable
        :param like_value: likelihood matrix (needs to be reevaluated)
        :param i_time: current time instance
        :return:
        """
        alpha = np.zeros(self.Ns)
        # the key of the formula
        d_max = min(self.D, i_time + 1)
        if p_type == 'none':
            for i in range(self.Ns):
                for d in range(d_max):
                    like_d = self.combined_likelihood(like_value, i_time, d)
                    alpha[i] = alpha[i] + alpha_star[i_time - d, i] * self.Pd[i, d] * like_d[i]
            return alpha
        elif p_type == 'scale':
            for i in range(self.Ns):
                for d in range(d_max):
                    like_d = self.combined_likelihood(like_value, i_time, d)
                    scaling = 1
                    for k in range(d):
                        scaling = scaling * scaling_coefficent[i_time - 1 - k]
                    alpha[i] = alpha[i] + alpha_star[i_time - d, i] * self.Pd[i, d] * like_d[i] * scaling
            ct = 1 / sum(alpha)
            alpha = alpha * ct
            return alpha, ct

    @staticmethod
    def combined_likelihood(like_value, t, d, norm_indicator=0):
        """
        To compute b_j(O_{t-d+1:t}) by assuming conditional independence
        :param like_value: likelihood matrix
        :param t: end time instance (likelihood will include t)
        :param d: the duration of the end state
        :param norm_indicator: indication normalization (hasn't solved the theory)
        :return: the sequence based observation likelihood for all states
                 vector based value
        """
        packed_likelihood = like_value[t - d:t + 1, :]  # extract corresponding likelihood value (matrix form)
        like = np.prod(packed_likelihood, axis=0)  # conditional independence, multiplied vertically (along t)
        if norm_indicator:
            like = like / like.sum()  # normalization
        return like

    def sojourn_gen(self, para):
        """
        :Function, generate sojourn time distribution matrix
        :param para: dict format to specify para in the distribution
        :return: Pd in a matrix format, Row: Ns, Column: D

        ########## ------ Gamma distribution ------ ############
        There are two ways for gamma distribution parametrization,
        we chose the shape and scale in sojourn generation, however,
        for the re-estimation assuming sojourn time is from gamma distribution,
        we chose the shape and inverse scale(beta) parametrization

        """

        def state_sojurn(parameter, s_type):
            Pd = np.zeros(self.D)
            if s_type == 'Geometric':
                assert len(parameter) == 1, "The stay probability dimension is not one!"
                stay_pb = parameter[0]
                for d in range(0, self.D):
                    Pd[d] = stay_pb * (1 - stay_pb) ** d
                Pd = Pd / sum(Pd)
            elif s_type == 'Gaussian':
                assert len(parameter) == 2, "The stay probability dimension is not two: mean and sd!"
                mean_d = parameter[0]
                sd = parameter[1]  # standard deviation
                for d in range(self.D):
                    Pd[d] = stats.norm(mean_d, sd).pdf(d)
                Pd = Pd / sum(Pd)
            elif s_type == 'Gamma':
                assert len(parameter) == 2, "The Gamma probability dimension is not two: alpha and beta!"
                a = parameter[0]  # shape parameter
                b = parameter[1]  # scale
                # mean is a*b (shape * scale)
                for d in range(self.D):
                    # when d is 0, physically it maps to d=1, but we compute d=0 in the
                    # probability setup, which means p(d) = p(d-1), this is to exclude the 0
                    Pd[d] = stats.gamma.pdf(d + 1, a, loc=0, scale=b)
                Pd = Pd / sum(Pd)
            elif s_type == 'BBD':
                alpha_d = parameter[0]
                beta_d = parameter[1]
                n = self.D - 1  # pay attention to this
                for d in range(self.D):  # support k in {0, 1, ..., D-1} maps to -> {1, 2, ..., D}
                    Pd[d] = comb(n, d) * betafunction(d + alpha_d, n - d + beta_d) / betafunction(
                        alpha_d, beta_d)
            elif s_type == 'Uniform':
                assert len(parameter) == 1, "The parameter dimension is not one!"
                assert parameter[0] is None, "The parameter for uniform distribution is not None!"
                Pd = 1 / self.D * np.ones(self.D)
            elif s_type == 'Possion':
                assert len(parameter) == 1, "The lambda para dimension is not one!"
                lambda_p = parameter[0]
                for d in range(self.D):
                    Pd[d] = lambda_p ** d * exp(-lambda_p) / factorial(d)
                Pd = Pd / sum(Pd)
            else:
                raise Exception("The format {} has not been defined!".format(s_type))
            return Pd

        for i in range(0, self.Ns):
            self.Pd[i, :] = state_sojurn(para['parameter'][i], s_type=para['type'][i])
        return self.Pd


