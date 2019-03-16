# StationId, StationName, StationMap
import time
import json
from pymongo import MongoClient
import os
import pandas as pd
import numpy as np
from scipy import stats

HOST = '127.0.0.1'
PORT = 27017
DB = 'XRNN'

class DataService:
    def __init__(self, configPath):
        pass

        self.client = MongoClient(HOST, PORT)
        self.config = self.read_config()
        self.io_stats_df = self.read_state_merge('GRU_1')
        """
        Hard code
        """

        # self.db = self.client[DB]
        # if configPath == None:
        #     return
        # self.config_path = configPath
        # self.init_config()


    def read_config(self):
        with open('./config/config.json', 'r') as input_file:
            config_json = json.load(input_file)
            return config_json

    def get_units_stats(self, m_id, db = False):
        """
        Read units status, filepath:./data/GRU_1-units_stats.json
        :param m_id: the id of model
        :param db: if collect data from database(MongoDB)
        :return: all the statistics
        """
        file_path = './data/{}-{}.json'.format(m_id, 'units_stats')
        if (not os.path.exists(file_path)) or db == True:
            print('d')
            stats_c_name = self.config[m_id]['unit_stats_collection']
            db_name = self.config[m_id]['database']
            collection = MongoClient('127.0.0.1', 27017)[db_name][stats_c_name]
            stats_c_records = list(collection.find({},{'_id': 0}))
            return stats_c_records
        else:
            print('f')
            with open(file_path, 'r') as input_file:
                return json.load(input_file)

    def get_and_save_units_stats(self, m_id):
        """
        save units statistics to file, to accelerate reading data, filepath:./data/GRU_1-units_stats.json
        :param m_id: model id
        :return: no return, save data to file
        """
        stats_c_name = self.config[m_id]['unit_stats_collection']
        db_name = self.config[m_id]['database']
        collection = MongoClient('127.0.0.1', 27017)[db_name][stats_c_name]
        stats_c_records = list(collection.find({}, {'_id': 0}))
        with open('./data/{}-{}.json'.format(m_id, 'units_stats'),'w') as output_file:
            json.dump(stats_c_records, output_file)


    def get_feature_stats(self, m_id, db = False):
        """z
        Read units status, filepath:./data/GRU_1-units_stats.json
        :param m_id: the id of model
        :param db: if collect data from database(MongoDB)
        :return: all the statistics
        """
        file_path = './data/{}-{}.json'.format(m_id, 'feature_stats')
        if (not os.path.exists(file_path)) or db == True:
            print('d')
            stats_c_name = self.config[m_id]['feature_stats_collection']
            db_name = self.config[m_id]['database']
            collection = MongoClient('127.0.0.1', 27017)[db_name][stats_c_name]
            stats_c_records = list(collection.find({},{'_id': 0}))
            return stats_c_records
        else:
            print('f')
            with open(file_path, 'r') as input_file:
                return json.load(input_file)


    def get_bi_cluster(self,m_id, nc):
        bi_cluster_json = self.config[m_id]['bi_cluster_file'][str(nc)]
        with open(bi_cluster_json, 'r') as input_file:
            return json.load(input_file)




    def get_and_save_feature_stats(self, m_id):
        """
        save units statistics to file, to accelerate reading data, filepath:./data/GRU_1-units_stats.json
        :param m_id: model id
        :return: no return, save data to file
        """

        stats_c_name = self.config[m_id]['feature_stats_collection']
        db_name = self.config[m_id]['database']
        collection = MongoClient('127.0.0.1', 27017)[db_name][stats_c_name]
        stats_c_records = list(collection.find({}, {'_id': 0}))

        with open('./data/{}-{}.json'.format(m_id, 'feature_stats'),'w') as output_file:
            json.dump(stats_c_records, output_file)


    def get_gradient_and_io_data(self, m_id, t_ids):
        """
        save units statistics to file, to accelerate reading data, filepath:./data/GRU_1-units_stats.json
        :param m_id: model id
        :return: no return, save data to file
        """
        def read_gradient_stats(t_id):
            with open("{}{}.json".format(gradient_folder, t_id), 'r') as output_file:
                return json.load(output_file)

        print('get_gradient_and_io_data')
        gradient_folder = self.config[m_id]['gradient_stats_folder']

        input_output_folder = self.config[m_id]['input_out_folder']

        io_columns = pd.read_csv("{}column.csv".format(input_output_folder)).columns
        input_output_list = []
        gradient_stats_list = []
        for t_id in t_ids:
            io_data = np.load("{}{}.npy".format(input_output_folder, t_id))
            input_output_list.append(io_data.tolist())
            gradient_stats_list.append(read_gradient_stats(t_id))


        return {
            "input_output_list": input_output_list,
            "gradient_stats_list": gradient_stats_list,
            "column": list(io_columns)
        }

    def get_feature_values(self, m_id, features):

        def df2dict(df):
            index_2_dict = df.T.to_dict()
            return [index_2_dict[index] for index in index_2_dict]


        feature_values_csv = self.config[m_id]['feature_value_file']
        df = pd.read_csv(feature_values_csv)
        sub_df = df[features + ['time','seconds']]
        dict_list = df2dict(sub_df)
        return dict_list

    def read_state_merge(self, m_id):
        start_time = time.time()
        data_file = np.load(self.config[m_id]['io_state_merge_data'])
        data_column = pd.read_csv(self.config[m_id]['io_state_merge_column']).columns
        all_seq_df = pd.DataFrame(data_file, columns = data_column)
        print(all_seq_df.shape, time.time() - start_time)
        return all_seq_df



    # ------------------------------------------------------------------#
    def get_stats_of_subgroup_data_mulit_features(self, all_seq_df, feature_scales, r_len=50, dif_type='ks'):
        def get_sample_points(value_range, num):
            gap = (value_range[-1] - value_range[0]) / num
            return [(r * gap + value_range[0]) for r in range(num + 1)]

        def form_feature_stats_dict(df, describe_df, column, min_v=-1, max_v=1, gap_n=50):
            try:
                column_stats = describe_df[column]
                feature_value_list = df[column]
                kernel = stats.gaussian_kde(feature_value_list)
                column_stats = dict(column_stats)
                column_stats['kde_sample_points'] = get_sample_points([min_v, max_v], gap_n)
                column_stats['kde_point'] = list(kernel(column_stats['kde_sample_points']).astype(float))
                column_stats['uid'] = column.split('_')[-1]

                num_freq = dict(
                    ((feature_value_list - min_v) / ((max_v - min_v) / gap_n)).astype(int).value_counts().sort_index())
                column_stats['distribution'] = [int(num_freq[i]) if i in num_freq else 0 for i in range(gap_n)]
                return column_stats
            except:
                return None

        def calculate_diff(seq1, seq2, dif_type='ks', seq1_des=None, seq2_des=None):
            """
            calculate the distribution difference of seq1 and seq2
            dif_type = ks, mean, 50%
            seq2 contains all data, seq1 contains filtering data
            """
            if dif_type == 'ks':
                ks1 = stats.ks_2samp(seq1, seq2)
                result = ks1.statistic
            elif dif_type in ['mean', '50%', '25%', 'min', 'max', '75%']:
                seq1_des = seq1.describe() if seq1_des is None else seq1_des
                seq2_des = seq2.describe() if seq2_des is None else seq2_des
                result = seq1_des[dif_type] - seq2_des[dif_type]

            return result, dif_type

        if set(all_seq_df.columns) <= set(feature_scales) != True:
            print('Some feature not existed!')
        output_df = all_seq_df.iloc[:, -100:]
        condition = None
        for i, f in enumerate(feature_scales):
            min_val, max_val = feature_scales[f][0] / r_len, feature_scales[f][1] / r_len
            if i == 0:
                condition = (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)
            else:
                condition = condition & (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)

        if condition is None:
            return []
        condition_output = output_df[condition]
        # hard code
        print(condition_output.shape, all_seq_df.shape)
        lim_n = 10000
        sub_df = output_df[condition].sample(n = lim_n if lim_n < condition_output.shape[0] else condition_output.shape[0])
        # sub_df = output_df[condition]
        sub_describe_df = sub_df.describe()

        if sub_df.shape[0] == 0:
            return []
        # sub stats for all hidden units
        sub_stats_list = []
        start_time = time.time()

        for column in sub_df.columns:
            sub_stats = form_feature_stats_dict(sub_df, sub_describe_df, column=column)

            all_se = output_df.sample(n = lim_n)[column]
            # all_se = output_df[column]
            sub_se = sub_df[column]
            if sub_stats is not None:
                dif, dif_type = calculate_diff(seq1=all_se, seq2=sub_se, dif_type=dif_type,
                                               seq2_des=sub_describe_df[column])
                sub_stats['dif'] = dif
                sub_stats['dif_type'] = dif_type

            sub_stats_list.append(sub_stats)
            # calculate ks_test

        print('inner', time.time() - start_time)

        return sub_stats_list

    def get_subgroup_statistics(self, feature_scales, r_len, dif_type = 'ks'):
        result = self.get_stats_of_subgroup_data_mulit_features(self.io_stats_df, feature_scales, r_len = r_len, dif_type = dif_type)

        return result
    # def get_map(self, station_id):
    #     map_path = None
    #     for obj in self.station_config:
    #         if obj['StationId'] == station_id:
    #             map_path = obj['StationMap']
    #
    #     if map_path == None:
    #         print('No station_id', station_id, 'is found')
    #         return None
    #
    #     with open(map_path, 'r') as map_file:
    #         map = json.load(map_file)
    #         map['stationId'] = station_id
    #         return map
    #
    # def get_legend_config(self, station_id):
    #     config_path = None
    #     for obj in self.station_config:
    #         if obj['StationId'] == station_id:
    #             config_path = obj['LegendConfig']
    #
    #     if config_path == None:
    #         print('No station_id', station_id, 'is found')
    #         return None
    #
    #     with open(config_path, 'r') as map_file:
    #         legend_config = json.load(map_file)
    #         return {
    #             'stationId': station_id,
    #             'legendConfig': legend_config}
    #
    # def get_recent_records_single_collection(self, c_name, start, time_range):
    #     collection = self.db[c_name] # people_activity / posts
    #     num = 0
    #     recent_arr = []
    #     start_time = time.time()
    #     for record in collection.find({
    #         'time_stamp':{
    #             '$gte': start,
    #             '$lt': (start + time_range)
    #         }
    #     }).sort('time_stamp', pymongo.ASCENDING):
    #         if "_id" in record:
    #             del record['_id']
    #         if "_id" in record:
    #             del record['map_data']
    #         recent_arr.append(record)
    #
    #     return recent_arr
    #
    # def get_recent_records(self, start, time_range):
    #     people_activity = self.get_recent_records_single_collection('people_activity', start, time_range)
    #     ticket_record = self.get_recent_records_single_collection('tickets_ADM', start, time_range)
    #
    #     return {
    #         'people_activity': people_activity,
    #         'ticket_record': ticket_record
    #     }
    #
    #
    # def get_people_count(self, day, ttt):
    #     """
    #     This function is used to retrieve the people count collection from MongoDB.
    #     Created by Qing Du (q.du@ust.hk)
    #     """
    #     collection = self.db['people_count']
    #     max_count = collection.find().sort('count', pymongo.DESCENDING).limit(1)[0]['count']
    #     result = {}
    #     result['max_count'] = max_count
    #     for record in collection.find({'day': day, 'time': ttt}):
    #         result[record['station_ID']] = record['count']
    #     return result

    def get_temporal_data(self):
        with open('./data/PM25_2018.json', 'r') as input_file:
            data = json.load(input_file)
            return json.dumps(data)

    def get_scatter_data(self):
        with open('./data/MDS_input24_1.json', 'r') as input_file:
            data = json.load(input_file)
            return json.dumps(data)

    # def get_distribution(self, c_name):

if __name__ == '__main__':
    dataService = DataService(None)
    dataService.get_recent_records(0, 100)
