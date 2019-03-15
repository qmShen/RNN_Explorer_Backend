# StationId, StationName, StationMap
import time
import json
from pymongo import MongoClient
import os
import pandas as pd
import numpy as np

HOST = '127.0.0.1'
PORT = 27017
DB = 'XRNN'

class DataService:
    def __init__(self, configPath):
        pass

        self.client = MongoClient(HOST, PORT)
        self.config = self.read_config()

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
        """
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

    def get_feature_values(self,m_id, features):

        def df2dict(df):
            index_2_dict = df.T.to_dict()
            return [index_2_dict[index] for index in index_2_dict]


        feature_values_csv = self.config[m_id]['feature_value_file']
        df = pd.read_csv(feature_values_csv)
        sub_df = df[features + ['time','seconds']]
        dict_list = df2dict(sub_df)
        return dict_list

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
