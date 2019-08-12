# StationId, StationName, StationMap
import time
import json
from pymongo import MongoClient
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.manifold import TSNE


HOST = '127.0.0.1'
PORT = 27017
DB = 'XRNN'

class DataService:
    def __init__(self, configPath):

        print('start')
        self.client = MongoClient(HOST, PORT)
        self.config = self.read_config()
        # self.io_stats_df = self.read_state_merge('GRU_1')
        # self.observation_feature = pd.read_csv(self.config['observation_feature'])

        # ------------------timestamps-----------------


        """
        Hard code
        """

        # self.db = self.client[DB]
        # if configPath == None:
        #     return
        # self.config_path = configPath
        # self.init_config()
        self.current_feature_gradient_to_end = None
        self.current_m_id = None
        self.statistics_name = ['min', 'max', 'mean', 'std', '25', '50', '75']

        self.test()

    def read_model_list(self):
        model_list = self.config['model_list']
        return model_list

    def read_config(self):
        with open('./config/test.json', 'r') as input_file:
            config_json = json.load(input_file)
            return config_json



    def read_selected_model(self, mid):
        print('Load io merge state!')
        self.io_stats_df = self.read_state_merge(mid)
        print('ssss', self.config[mid]['observation']);
        self.observation_feature = pd.read_csv(self.config[mid]['observation'])


    # Get projection data

    def get_gradient_projection(self, mid, target_feature):

        columns = np.load(self.config[mid]['projection_columns'], allow_pickle=True)
        projections = np.load(self.config[mid]['projection_data'], allow_pickle=True)

        feature_index = 3

        target_index = None
        target_column = None
        for index, column in enumerate(columns):
            if column[feature_index] == target_feature:
                target_column = column
                target_index = index
                break
        if target_column is None:
            print('No feature {} existed in the data'.format(target_feature))
        df = pd.DataFrame(projections[target_index], columns=target_column)
        print('projection_df', df.shape)
        data_list = df.to_dict('index').values()
        return list(data_list)



    # -------------------------------update--------------------------------------------------------
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
            stats_c_records = list(collection.find({}, {'_id': 0}))
            return stats_c_records
        else:
            print('f')
            with open(file_path, 'r') as input_file:
                return json.load(input_file)


    def get_bi_cluster(self,m_id, nc):
        """
        Not used!
        :param m_id:
        :param nc:
        :return:
        """
        bi_cluster_json = self.config[m_id]['bi_cluster_file'][str(nc)]
        with open(bi_cluster_json, 'r') as input_file:
            return json.load(input_file)


    def get_cluster(self, m_id, n_unit_cluster = 10, n_feature_cluster = 12):
        cluster_file = self.config[m_id]['cluster_file'][str("{}_{}".format(n_unit_cluster, n_feature_cluster))]
        with open(cluster_file, 'r') as input_file:
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



    def get_gradient_and_io_data_by_cluster(self, m_id, t_ids, n_unit_cluster=10, n_feature_cluster=12):
        """
        save units statistics to file, to accelerate reading data, filepath:./data/GRU_1-units_stats.json
        :param m_id: model id
        :return: no return, save data to file
        """

        def get_cluster(m_id, n_unit_cluster=n_unit_cluster, n_feature_cluster=n_feature_cluster):
            cluster_file = self.config[m_id]['cluster_file'][str("{}_{}".format(n_unit_cluster, n_feature_cluster))]
            with open(cluster_file, 'r') as input_file:
                return json.load(input_file)

        def get_unit_clusters_gradient(sequence_gradient, cluster, output_u_id, input_u_id):
            output_uids = cluster['unit_map'][str(output_u_id)]
            input_uids = cluster['unit_map'][str(input_u_id)]
            # hard code
            update_input_ids = [265 + _id for _id in input_uids]

            # the gradient of previous unit
            unit_cluster_gradient = np.absolute(sequence_gradient[:, output_uids, :][:, :, update_input_ids])
            unit_cluster_gradient_mean = unit_cluster_gradient.mean(axis=(1, 2))

            return unit_cluster_gradient_mean

        def get_feature_clusters_gradient(sequence_gradient, cluster, output_u_id, input_f_id, feature_index_map):

            output_uids = cluster['unit_map'][str(output_u_id)]
            input_fids = cluster['feature_map'][str(input_f_id)]

            input_findeces = [feature_index_map[feature_name] for feature_name in input_fids]

            # the gradient of current feature
            feature_cluster_gradient = np.absolute(sequence_gradient[:, output_uids, :][:, :, input_findeces])
            feature_cluster_gradient_mean = feature_cluster_gradient.mean(axis=(1, 2))

            return feature_cluster_gradient_mean

        def get_all_unit_clusters_gradient(sequence_gradient, cluster, feature_index_map):
            u_cluster_ids = sorted([int(i) for i in list(cluster['unit_map'])])
            f_cluster_ids = sorted([int(i) for i in list(cluster['feature_map'])])

            unit_gradient = np.ndarray((len(u_cluster_ids), len(u_cluster_ids), sequence_gradient.shape[0]))
            feature_gradient = np.ndarray((len(u_cluster_ids), len(f_cluster_ids), sequence_gradient.shape[0]))

            for output_id in u_cluster_ids:
                for input_id in u_cluster_ids:
                    unit_cluster_gradient = get_unit_clusters_gradient(sequence_gradient, cluster, output_id, input_id)
                    unit_gradient[output_id, input_id] = unit_cluster_gradient

            for output_id in u_cluster_ids:
                for input_id in f_cluster_ids:
                    feature_cluster_gradient = get_feature_clusters_gradient(sequence_gradient, cluster,
                                                                             output_id,
                                                                             input_id, feature_index_map)
                    feature_gradient[output_id, input_id] = feature_cluster_gradient

            return unit_gradient, feature_gradient

        def get_io_data(io_ndarray, cluster):
            # hard code
            unit_output = io_ndarray[:, 365: 365 + 100]
            cluster_ids = sorted([int(unit_cluster['uc_id']) for unit_cluster in cluster['unit_clusters']])
            result_array = np.ndarray((len(cluster_ids), io_ndarray.shape[0], 4))

            for cluster_id in cluster_ids:
                u_ids = cluster['unit_map'][str(cluster_id)]
                cluster_output = unit_output[:, u_ids]

                cluster_output_above = cluster_output.copy()
                cluster_output_below = cluster_output.copy()
                cluster_output_above[cluster_output_above < 0] = 0
                cluster_output_below[cluster_output_below > 0] = 0
                above_sum = cluster_output_above.sum(axis=1).reshape(cluster_output.shape[0], 1)
                below_sum = cluster_output_below.sum(axis=1).reshape(cluster_output.shape[0], 1)
                all_sum = np.absolute(cluster_output).sum(axis=1).reshape(cluster_output.shape[0], 1)
                all_mean = np.absolute(cluster_output).mean(axis=1).reshape(cluster_output.shape[0], 1)

                result_arr = np.concatenate((above_sum, below_sum, all_sum, all_mean), axis=1)
                # sum of above 0, sum of below 0, sum of absolute, mean of absolute
                result_array[cluster_id] = result_arr

            return result_array

        def read_gradient(t_id):
            file_name = "{}{}.npy".format(gradient_folder, t_id);
            arr = np.load(file_name)
            return arr

        # cluster_json = self.get_bi_cluster(m_id, 15)
        # bi_cluster = cluster_json['bi_clusters']

        cluster_json = get_cluster(m_id)
        cluster = cluster_json

        gradient_folder = self.config[m_id]['gradient_folder']
        input_output_folder = self.config[m_id]['input_out_folder']

        io_columns = pd.read_csv("{}column.csv".format(input_output_folder)).columns
        feature_index_map = {}
        # hard code
        for _i, column in enumerate(list(io_columns[:265])):
            feature_index_map[column] = _i

        cluster_io_list = []
        unit_cluster_gradient_list = []
        feature_cluster_gradient_list = []
        for t_id in t_ids:
            io_data = np.load("{}{}.npy".format(input_output_folder, t_id))
            sequence_gradient = read_gradient(t_id)
            all_cluster_gradient = get_all_unit_clusters_gradient(sequence_gradient,
                                                                  cluster,
                                                                  feature_index_map=feature_index_map)

            all_cluster_io = get_io_data(io_data, cluster)
            cluster_io_list.append(all_cluster_io.tolist())
            unit_cluster_gradient_list.append(all_cluster_gradient[0].tolist())
            feature_cluster_gradient_list.append(all_cluster_gradient[1].tolist())


        return {
            "cluster_io_list": cluster_io_list,
            "unit_cluster_gradient_list": unit_cluster_gradient_list,
            "feature_cluster_gradient_list": feature_cluster_gradient_list
        }

    def get_feature_gradient_to_end(self, m_id, t_ids, n_unit_cluster=10, n_feature_cluster=12):
        """
        !!!!! unit cluster and feature cluster is not fixed!
        save units statistics to file, to accelerate reading data, filepath:./data/GRU_1-units_stats.json
        :param m_id: model id
        :return: no return, save data to file
        """
        def get_cluster(m_id, n_unit_cluster=n_unit_cluster, n_feature_cluster=n_feature_cluster):
            cluster_file = self.config[m_id]['cluster_file'][str("{}_{}".format(n_unit_cluster, n_feature_cluster))]
            with open(cluster_file, 'r') as input_file:
                return json.load(input_file)

        def get_observation_features(t_ids):
            data_array = []
            for tid in t_ids:
                time_list = [tid - (23 - i)*3600 for i in range(0, 24)]
                feature_list = self.observation_feature[self.observation_feature['seconds'].isin(time_list)].values[:,:-1]
                data_array.append({
                    'time_stamp': tid,
                    'value': feature_list.tolist()
                })
            return data_array

        def get_feature_gradient_to_end(time_sequence, timestamps, feature_gradient, cluster, columns):
            """
            :return:[{
                feature: PM25,
                timestamp: 1515168000,
                feature_gradient: 2dList,
                feature_cluster_gradient: 2dList
            }]
            """

            def get_cluster_gradient(feature_gradient, feature_index_clusters, feature_index):

                clusters_gradient = None

                for index, cluster in enumerate(feature_index_clusters):
                    sub_gradient = feature_gradient[:, cluster]
                    shape = sub_gradient.shape

                    cluster_sum_gradient = np.mean(sub_gradient, axis=1).reshape((shape[0], 1))
                    if clusters_gradient is None:
                        clusters_gradient = cluster_sum_gradient
                    else:
                        clusters_gradient = np.concatenate((clusters_gradient, cluster_sum_gradient), axis=1)

                return clusters_gradient


            features = self.features_columns[-5:]

            time_indices = [int((time_sequence[i] - timestamps[0]) / 3600) for i in range(len(time_sequence))]

            selected_gradient = feature_gradient[:, time_indices, :, :]
            feature_gradient_map = {}

            feature_index = {}
            for index, column in enumerate(columns):
                feature_index[column] = index

            feature_clusters = cluster['feature_clusters']
            feature_index_clusters = []

            for cluster in feature_clusters:
                cluster['f_indices'] = [feature_index[i] for i in cluster['f_ids']]
                feature_index_clusters.append(cluster['f_indices'])


            for i in range(selected_gradient.shape[0]):
                feature_gradient_map[features[i]] = []
                for time_index in range(len(selected_gradient[i])):
                    _gradient = np.absolute(selected_gradient[i][time_index])
                    _group_gradient = get_cluster_gradient(_gradient, feature_index_clusters, feature_index)

                    feature_gradient_map[features[i]].append({
                        'feature': features[i],
                        'timestamp': time_sequence[time_index],
                        'feature_gradient': _gradient.T.tolist(),
                        'feature_cluster_gradient': _group_gradient.tolist(),
                        'feature_gradient_mean': _gradient.mean(axis = 1).tolist()
                    })

            return feature_gradient_map, feature_index_clusters

        cluster_json = get_cluster(m_id)

        feature_gradient, gradient_timestamps = self.read_feature_gradient_and_time_to_end(m_id)
        feature_gradient_result, feature_cluster_list = get_feature_gradient_to_end(t_ids, gradient_timestamps, feature_gradient, cluster_json, self.features_columns)
        feature_value = get_observation_features(t_ids)
        print('m_id, t_ids, n_unit_cluster, n_feature_cluster', m_id, t_ids, n_unit_cluster, n_feature_cluster)

        return {'feature_gradient_to_end':feature_gradient_result, 'cluster': feature_cluster_list, 'all_features': self.features_columns, 'feature_value': feature_value}




    def read_feature_gradient_and_time_to_end(self, m_id):
        if self.current_m_id is not None and self.current_m_id == m_id:
            return self.current_feature_gradient_to_end, self.current_gradient_time
        self.current_feature_gradient_to_end = np.load(self.config[m_id]['feature_gradient_to_end'])
        self.current_m_id = m_id
        self.current_gradient_time = np.load(self.config[m_id]['feature_gradient_timestamps'])
        return self.current_feature_gradient_to_end, self.current_gradient_time

    def get_feature_values(self, m_id, features = None):
        def df2dict(df):
            index_2_dict = df.T.to_dict()
            return [index_2_dict[index] for index in index_2_dict]


        feature_values_csv = self.config[m_id]['feature_value_file']
        df = pd.read_csv(feature_values_csv)

        sub_df = df[features + ['time','seconds']] if features is not None else df
        dict_list = df2dict(sub_df)
        return dict_list

    def get_feature_values_scaled(self, features = None):

        def df2dict(df):
            index_2_dict = df.T.to_dict()
            return [index_2_dict[index] for index in index_2_dict]


        feature_values_csv = self.config['observation_feature']
        df = pd.read_csv(feature_values_csv)
        print(df.columns)
        sub_df = df[features + ['seconds']] if features is not None else df
        dict_list = df2dict(sub_df)
        return dict_list


    def read_state_merge(self, m_id):
        print('Read state merge')
        start_time = time.time()
        data_file = np.load(self.config[m_id]['io_state_merge_data'])
        data_column = pd.read_csv(self.config[m_id]['io_state_merge_column']).columns
        """
        Hard code
        """
        features_columns = []
        for c in data_column:
            if c == '0' or c == 0:
                break
            features_columns.append(c)
        self.features_columns = features_columns
        all_seq_df = pd.DataFrame(data_file, columns = data_column)
        print("all_seq_df", all_seq_df.shape, time.time() - start_time)
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

        condition = None
        for i, f in enumerate(feature_scales):
            min_val, max_val = feature_scales[f][0] / r_len, feature_scales[f][1] / r_len
            if i == 0:
                condition = (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)
            else:
                condition = condition & (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)

        if condition is None:
            return []

        all_sub_df = all_seq_df[condition]
        #
        # output_df = all_seq_df.iloc[:, 365: 365 + 100]
        # condition_output = output_df[condition]

        condition_output = all_sub_df.iloc[:, 365: 365 + 100]
        # hard code
        print(condition_output.shape, all_seq_df.shape)

        lim_n = 10000

        sub_df = condition_output.sample(n = lim_n if lim_n < condition_output.shape[0] else condition_output.shape[0])

        # sub_df = output_df[condition]
        sub_describe_df = sub_df.describe()

        if sub_df.shape[0] == 0:
            return []
        # sub stats for all hidden units
        sub_stats_list = []
        start_time = time.time()

        for column in sub_df.columns:
            sub_stats = form_feature_stats_dict(sub_df, sub_describe_df, column=column)

            all_se = all_seq_df.sample(n = lim_n)[column]
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


    def get_subgroup_scatter_plot(self, feature_scales, r_len=50):
        start_time = time.time()
        all_seq_df = self.io_stats_df
        if set(all_seq_df.columns) <= set(feature_scales) != True:
            print('Some feature not existed!')

        condition = None
        for i, f in enumerate(feature_scales):
            min_val, max_val = feature_scales[f][0] / r_len, feature_scales[f][1] / r_len
            if i == 0:
                condition = (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)
            else:
                condition = condition & (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)

        if condition is None:
            return []

        all_sub_df = all_seq_df[condition]
        #


        condition_output = all_sub_df.iloc[:, 365: 365 + 100]
        # hard code
        print(condition_output.shape, all_seq_df.shape)
        print('timetime', time.time() - start_time)

############
    def get_scatter_plot_by_sub_groups(self, sub_groups):
        def get_subgroup_scatter_plot(feature_scales, r_len=50):
            start_time = time.time()
            all_seq_df = self.io_stats_df
            if set(all_seq_df.columns) <= set(feature_scales) != True:
                print('Some feature not existed!')

            condition = None
            for i, f in enumerate(feature_scales):
                min_val, max_val = feature_scales[f][0] / r_len, feature_scales[f][1] / r_len
                if i == 0:
                    condition = (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)
                else:
                    condition = condition & (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)

            if condition is None:
                return []
            all_sub_df = all_seq_df[condition]
            condition_output = all_sub_df.iloc[:, 365:]

            return condition_output

        def set_class(df, class_range):
            _sum_class = 0
            for i, group_size in enumerate(class_range):
                df.loc[(df.index < _sum_class + group_size) & (df.index >= _sum_class), 'class'] = i
                _sum_class += group_size
            df['class'] = df['class'].astype(int)
            return df

        def return_unconnected_df(df, column='sequence_time', hour_gap=6):
            time_series = df[column]
            pd_indeces = []
            for index, t in enumerate(time_series):
                pd_i = time_series.index[index]
                if index == 0:
                    pd_indeces.append(pd_i)
                    current_time = t
                    continue

                if (t - current_time) >= hour_gap * 3600:
                    pd_indeces.append(pd_i)
                    current_time = t
            return df.iloc[pd_indeces]


        group_configs = sub_groups
        unit_dfs = []
        identify_features = []
        all_selected_features = []
        for i, feature_scales in enumerate(group_configs):
            sub_df = get_subgroup_scatter_plot(feature_scales, r_len=50)

            sub_df['sequence_time'] = sub_df['sequence_time'].astype(int, inplace=True)
            sub_df['unit_time'] = sub_df['unit_time'].astype(int, inplace=True)
            units_df = sub_df[sub_df['unit_time'] == sub_df['sequence_time']]

            all_selected_features.append(units_df.iloc[:, -3:].values.tolist())
            units_df = units_df.sort_values(['sequence_time'])
            units_df = units_df.reset_index(drop=True)
            units_df = return_unconnected_df(units_df, column='sequence_time', hour_gap=6)


            n = 500 if units_df.shape[0]>500 else units_df.shape[0]
            units_df = units_df.sample(n = n)
            unit_dfs.append(units_df)
            identify_features.append(units_df.iloc[:, -3:].values)



        all_values = np.concatenate(tuple([sub_df.values[:, :100] for sub_df in unit_dfs]))
        identify_features = np.concatenate(tuple([i_df for i_df in identify_features]))
        class_range = [sub_df.shape[0] for sub_df in unit_dfs]

        print('Start generating TSNE plot', all_values.shape)
        start_time = time.time()
        embedded = TSNE(n_components=2, perplexity = 20).fit_transform(all_values)
        print("Using time", time.time() - start_time);

        df = pd.DataFrame(embedded, columns=['x', 'y'])

        df = set_class(df, class_range)
        df['unit_time'] = identify_features[:, 0]
        df['sequence_time'] = identify_features[:, 1]
        df['_id'] = identify_features[:, 2]

        return df, all_selected_features

    def get_scatter_plot_by_sub_groups_sequence_pattern(self, sub_groups):
        """
        Currenly
        :param sub_groups:
        :return:
        """
        print('sequence')
        def get_subgroup_scatter_plot(feature_scales, r_len=50):
            start_time = time.time()
            all_seq_df = self.io_stats_df
            if set(all_seq_df.columns) <= set(feature_scales) != True:
                print('Some feature not existed!')

            condition = None
            for i, f in enumerate(feature_scales):
                min_val, max_val = feature_scales[f][0] / r_len, feature_scales[f][1] / r_len
                if i == 0:
                    condition = (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)
                else:
                    condition = condition & (all_seq_df[f] > min_val) & (all_seq_df[f] < max_val)

            if condition is None:
                return []
            all_sub_df = all_seq_df[condition]
            condition_output = all_sub_df.iloc[:, 365:]

            return condition_output

        def set_class(df, class_range):
            _sum_class = 0
            for i, group_size in enumerate(class_range):
                df.loc[(df.index < _sum_class + group_size) & (df.index >= _sum_class), 'class'] = i
                _sum_class += group_size
            df['class'] = df['class'].astype(int)
            return df

        def return_unconnected_df(df, column='sequence_time', hour_gap=6):
            time_series = df[column]
            pd_indeces = []
            for index, t in enumerate(time_series):
                pd_i = time_series.index[index]
                if index == 0:
                    pd_indeces.append(pd_i)
                    current_time = t
                    continue

                if (t - current_time) >= hour_gap * 3600:
                    pd_indeces.append(pd_i)
                    current_time = t
            return df.iloc[pd_indeces]


        group_configs = sub_groups
        unit_dfs = []
        identify_features = []
        all_selected_features = []
        for i, feature_scales in enumerate(group_configs):
            sub_df = get_subgroup_scatter_plot(feature_scales, r_len=50)

            sub_df['sequence_time'] = sub_df['sequence_time'].astype(int, inplace=True)
            sub_df['unit_time'] = sub_df['unit_time'].astype(int, inplace=True)
            units_df = sub_df[sub_df['unit_time'] == sub_df['sequence_time']]

            all_selected_features.append(units_df.iloc[:, -3:].values.tolist())
            units_df = units_df.sort_values(['sequence_time'])
            units_df = units_df.reset_index(drop=True)
            units_df = return_unconnected_df(units_df, column='sequence_time', hour_gap=6)


            n = 500 if units_df.shape[0]>500 else units_df.shape[0]
            units_df = units_df.sample(n = n)
            unit_dfs.append(units_df)
            identify_features.append(units_df.iloc[:, -3:].values)



        all_values = np.concatenate(tuple([sub_df.values[:, :100] for sub_df in unit_dfs]))
        identify_features = np.concatenate(tuple([i_df for i_df in identify_features]))
        class_range = [sub_df.shape[0] for sub_df in unit_dfs]

        print('Start generating TSNE plot', all_values.shape)
        start_time = time.time()
        embedded = TSNE(n_components=2, perplexity = 20).fit_transform(all_values)
        print("Using time", time.time() - start_time);

        df = pd.DataFrame(embedded, columns=['x', 'y'])

        df = set_class(df, class_range)
        df['unit_time'] = identify_features[:, 0]
        df['sequence_time'] = identify_features[:, 1]
        df['_id'] = identify_features[:, 2]

        return df, all_selected_features


    # def get_temporal_data(self):
    #     with open('./data/PM25_2018.json', 'r') as input_file:
    #         data = json.load(input_file)
    #         return json.dumps(data)

    def get_temporal_data(self):
        with open('./data/sequence_cluster_PM25.json', 'r') as input_file:
            data = json.load(input_file)
            return json.dumps(data)

    def get_scatter_data(self):
        with open('./data/MDS_input24_1.json', 'r') as input_file:
            data = json.load(input_file)
            return json.dumps(data)

    # def get_distribution(self, c_name):

    def get_feature_gradient_statistics(self, m_id, target_feature):
        """

        :param m_id: the id of model
        :param target_feature: the prediction feature name , one of ['min', 'max', 'mean', 'std', '25', '50', '75']
        :return: {
                    'statistics_name': ['min', 'max', 'mean', 'std', '25', '50', '75']，
                    'temporal_statistics': feature_gradient
                    }
                The feature gradient(temporal_statistics)is 5 * 265 * 24 * 7,
                indicates the target features, input features, timestamp, statistics name
        """
        input_feature_gradient_data_path = self.config[m_id]['result_gradient_statistics']
        input_feature_gradient = np.load(input_feature_gradient_data_path)
        input_features = list(self.features_columns)
        target_features = list(self.features_columns[-5:])
        statistics_name = self.statistics_name
        def get_json_gradient_importance(feature, result_gradient, features_columns, target_columns, statistics_name):
            feature_index = target_columns.index(feature)
            print('feature_index', target_columns, feature_index)
            features_gradient = result_gradient[feature_index]
            feature_statics_object = []
            for feature_id, feature in enumerate(features_columns):
                feature_gradient = features_gradient[feature_id]
                feature_statics_object.append({
                    'feature_name': feature,
                    'temporal_statistics': feature_gradient.tolist()
                })

            return {
                'statistics_name': statistics_name,
                'feature_statics': feature_statics_object
            }

        result = get_json_gradient_importance(target_feature, input_feature_gradient, input_features, target_features, statistics_name)
        return result


    def test(self):
        return
        self.get_feature_gradient_statistics("GRU_1", "PM25")


if __name__ == '__main__':
    dataService = DataService(None)
    dataService.get_recent_records(0, 100)
