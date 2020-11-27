import numpy as np
import os
from scipy.stats import bernoulli
import pickle


class TxtManager:
    """
    Purpose: To manage txt-based data extraction with specific requirements
     **.txt data is from Bosch vehicle based measurements campaign
    There are five anchors involved
    """

    def __init__(self, filename=[]):
        self.filename = filename

    def txtname_set(self, filename):
        self.filename = filename

    def line_parse(self, lines):
        """
        To parse a single reading of the Bluetooth piconet data
        :param lines: single line as a input
        :return: data, status indicating if the anchor gets measurement update,
                glitch_indication: 0-(normal), 1-(glitch) 2-(blank lines)
                blank lines are artificially created when I save the data in matlab
        """
        segment_line = lines.split()  # by default it is separated by space
        status_position = 0
        auth = 0  # initialized as 0 each time
        data = []
        extracted_status = []
        count = 0
        for item in segment_line:
            status_position += 1  # the postion of update status, after Auth it will be the status
            if item == 'Auth':
                auth = 1
                break
        if auth:
            for item in lines:  # loop through the data one by one
                count += 1
                if item == '{':
                    item_from = count
                if item == '}':
                    item_to = count - 1
                    break
            glitch_indicator = 0
            data = list(map(float, lines[item_from:item_to].split()))
            # parse the status
            for i in segment_line[status_position]:
                if i == 'X' or i == 'L':
                    extracted_status.append(1)
                else:
                    extracted_status.append(0)
        else:
            if lines == '\n':
                glitch_indicator = 2
            else:
                glitch_indicator = 1
        return data, extracted_status, glitch_indicator

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


