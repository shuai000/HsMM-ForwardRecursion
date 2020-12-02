import numpy as np
import os
from scipy.stats import bernoulli
"""
The functions here are used to parse the .p2m RSS data file and extract the corresponding RSS data
"""


class TxtManager:
    """
    Purpose: To manage txt-based data extraction with specific requirements
     **.txt data is from Bosch vehicle based measurements campaign
    There are five anchors involved
    """

    def __init__(self, filename=[]):
        self.filename = filename

    def wirelessinsite_path(self, pathname, s_num, offset=1, ap_num=5):
        """
         extract RSS data from the .p2m file (obtained from ray-tracing
        :param pathname: directory path
        :param s_num: state number in an HMM
        :param offset: manually remove the description non-numbered data at the beginning
        :param ap_num: Anchor number
        :return: Extracted RSS data, list, each item corresponds to an HMM state
        """
        state_num = s_num
        rx_data_all = [None for i in range(state_num)]
        rx_num = ap_num
        data_length = 1000

        fs = os.listdir(pathname)
        for i_state in range(state_num):
            rss_data = np.zeros((data_length, rx_num))
            for item in fs:
                rx_index = int(item.split('_')[-1].split('.')[0]) - 1
                state_index = int(item.split('_')[-1].split('.')[1][2:4]) - offset
                if state_index == i_state:
                    tmp_path = os.path.join(pathname, item)
                    data = self.wirelessinsite_file(tmp_path)
                    rss_data[0:len(data), rx_index] = data
            rx_data_all[i_state] = rss_data[0:len(data), :]
        return rx_data_all

    @staticmethod
    def wirelessinsite_file(tmp_filename):
        tmp_data = []
        f = open(tmp_filename, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            if line[0] != '#':
                data = float(line.split()[5])  # by default it is separated by space
                tmp_data.append(data)
        return tmp_data

    @staticmethod
    def miss_pattern_gen(anchor_num, successful_rate, time_length, seed_num=1):
        status = np.zeros((time_length, anchor_num))
        np.random.seed(seed_num)
        for i in range(anchor_num):
            status[:, i] = bernoulli.rvs(successful_rate[i], size=time_length)
        return status


