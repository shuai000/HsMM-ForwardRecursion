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

    def combined_parse(self, compacted_num, pack_type='takeout', is_reverse=False):
        """
        To parse several lines altogether and merge their patterns
        This is due to the txt printing rate is double of the phone advertising rate
        in the hardware setup, currently, the tag is broadcasting at 10Hz, while each
        anchor might not be able to measure the RSS during a certain broadcasting.
        RSS collected at the master for all the slave anchors is printed in the logfile
        at 20Hz(double) rate.

        :param compacted_num: 2 for current setup as the printing rate is doubled
        :param pack_type: Because we merge data according to the compacted number in the buffer,
                          we specify two strategies: 1. take out the latest one ('takeout')
                                                     2. take the average        ('average')
        :return:
        """
        anchor_num = 5
        data_all = np.zeros((10000, anchor_num))
        data_status = np.zeros((10000, anchor_num))

        buffer_RSS = np.zeros((compacted_num, anchor_num))
        buffer_status = np.zeros((compacted_num, anchor_num))

        f = open(self.filename, 'r')
        lines = f.readlines()
        if is_reverse:
            lines.reverse()
        count_local = 0
        count_global = 0
        for item in lines:
            current_index = count_local % compacted_num
            if current_index == 0 and count_local != 0:  # condition that data needs to be packed
                data_all[count_global, :], data_status[count_global, :] = self.data_pack(buffer_RSS, buffer_status,
                                                                                         pack_type)
                count_global += 1
            count_local += 1
            temp_data, temp_status, glitch_indicator = self.line_parse(item)
            if glitch_indicator == 1:     # a physical glitch
                count_local = 0
            elif glitch_indicator == 2:   # this line is a '/n'
                pass
            else:
                buffer_RSS[current_index, :] = temp_data
                buffer_status[current_index, :] = temp_status
        data_all = data_all[0:count_global, :]
        data_status = data_status[0:count_global, :]
        return data_all, data_status

    def data_pack(self, RSS, status, pack_type='takeout'):
        """
        Function: merge RSS data from RSS array based on its status
        :param RSS: T * anchor_num, T is the time length to be packed
        :param status: T * anchor_num, corresponding status (0 or 1)
        :param pack_type: takeout--merged as 0 if no status in the length
                          is 1 (by default)
                          average--average the data value (kind of using
                          old data to make up the new data)
        :return:
        """
        merge_value = np.zeros(RSS.shape[1])
        merge_status = np.zeros(RSS.shape[1])
        # merge the data
        if pack_type == 'takeout':
            for i in range(RSS.shape[1]):
                effective_index = []
                for j in range(RSS.shape[0]):
                    if status[j, i] == 1:
                        effective_index.append(j)
                if effective_index:
                    merge_status[i] = 1
                    merge_value[i] = RSS[
                        effective_index[-1], i]  # take the value that is updated, if more than one time based data
                    # for a certain anchor is updated, take the last one
        elif pack_type == 'average':
            time_length = RSS.shape[0]
            for i in range(RSS.shape[1]):
                merge_value[i] = sum(RSS[:, i]) / time_length
                if sum(status[:, i]) > 0:
                    merge_status[i] = 1
        return merge_value, merge_status

    @staticmethod
    def fliter_txtspace(readname, writename):
        """
        To remove the space line in the log data file
        :param readname:
        :param writename:
        :return:
        """
        f2 = open(writename, 'w')
        with open(readname) as f:
            for line in f:
                if line != '\n':
                    f2.writelines(line)
        f2.close()

    @staticmethod
    def concatenate_file(input_name, writename):
        fw = open(writename, 'w')
        for filename in input_name:
            fr = open(filename, 'r')
            fw.writelines(fr.readlines())
            fr.close()
        fr.close()
        fw.close()

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

    def rx_oriented_extraction(self, pathname, rx_aimed, reshapesize=(11, 22)):
        """
        Extract data from a specified txt file in a given path
        :param pathname:
        :param rx_index:
        :return:
        """
        def data_read(tmp_filename):
            tmp_data = []
            x = []
            y = []
            f = open(tmp_filename, 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                if line[0] != '#':
                    temp = line.split()
                    data = float(temp[5])  # by default it is separated by space
                    temp_x = float('%.1f' % float(temp[1]))
                    temp_y = float('%.1f' % float(temp[2]))
                    tmp_data.append(data)
                    if temp_x not in x:
                        x.append(temp_x)
                    if temp_y not in y:
                        y.append(temp_y)
            return tmp_data, x, y

        fs = os.listdir(pathname)
        for item in fs:
            rx_index = int(item.split('_')[-1].split('.')[0])
            if rx_index == rx_aimed:
                tmp_path = os.path.join(pathname, item)
                data, x, y = data_read(tmp_path)
        data = np.array(data).reshape((len(y), len(x)))
        return data, x, y

    def wirelessinsite_file(self, tmp_filename):
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

    @staticmethod
    def save_variable(v, filename):
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename

    @staticmethod
    def load_variable(filename):
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r

