# -*- coding: UTF-8 -*-
from app import app
import json
from app.DataService.DataService import DataService
from flask import request

import pandas as pd

import os
cwd = os.getcwd()
print("Test root path", cwd)

dataService = DataService('config.txt')
print('here')


# @app.route('/testscatterplot')
# def get_test_scatter_plot():
#     print('here')
#     return json.dumps('test_scatter_plot')

@app.route('/')
def index():
    print('ere')
    return app.send_static_file('index.html')

@app.route('/test')
def getStationConfig():
    return json.dumps("test")

@app.route('/test2')
def getStationConfig2():
    return json.dumps("test2")

@app.route('/testscatterplot')
def get_test_scatter_plot():
    with open('./data/test_scatter_plot.json','r') as input_file:
        data = json.load(input_file)
        return json.dumps(data)



@app.route('/temporal_trend')
def get_temporal_Trend():
    return dataService.get_temporal_data()

@app.route('/scatter_data')
def get_scatter_data():
    return dataService.get_scatter_data()


@app.route('/all_units_stats',  methods = ['POST'])
def get_units_stats_data():
    post_data = json.loads(request.data.decode())
    print('Get unit statistics', post_data)
    return json.dumps(dataService.get_units_stats(post_data['mid']))

@app.route('/all_feature_stats',  methods = ['POST'])
def get_feature_stats_data():
    post_data = json.loads(request.data.decode())
    print('Get feature statistics',post_data)
    return json.dumps(dataService.get_feature_stats(post_data['mid']))


@app.route('/all_stats',  methods = ['POST'])
def get_stats_data():
    post_data = json.loads(request.data.decode())
    print('Get all statistics',post_data)
    return json.dumps({
        'features': dataService.get_feature_stats(post_data['mid']),
        'units': dataService.get_units_stats(post_data['mid']),
        'bicluster':dataService.get_bi_cluster(post_data['mid'],post_data['nc'])
    })



@app.route('/cell_input_output',  methods = ['POST'])
def get_gradient_and_io():
    post_data = json.loads(request.data.decode())
    print('Get cell input out ',post_data)
    return json.dumps(dataService.get_gradient_and_io_data(post_data['mid'],post_data['tid']))

@app.route('/feature_values',  methods = ['POST'])
def get_feature_values():

    post_data = json.loads(request.data.decode())
    print('Get feature values ',post_data)
    dicts = dataService.get_feature_values(post_data['mid'],post_data['features'])
    return json.dumps(dicts)

@app.route('/subgroup_stats',  methods = ['POST'])
def get_subgroup_stats():

    post_data = json.loads(request.data.decode())
    print('Get feature values ',post_data)
    # feature_scales, r_len, dif_type
    feature_scales = post_data['feature_scales']
    r_len = 50
    dif_type =  post_data['dif_type']
    results = dataService.get_subgroup_statistics(feature_scales = feature_scales, r_len = r_len, dif_type = dif_type)
    return json.dumps(results)



@app.route('/scatter_plot_subgroup',  methods = ['POST'])
def get_subgroup_scatter_plot():

    post_data = json.loads(request.data.decode())
    print('Get scatter_plot_subgroup values ',post_data)
    # feature_scales, r_len, dif_type
    feature_scales = post_data['feature_scales']
    r_len = 50

    # results = dataService.get_subgroup_scatter_plot(feature_scales = feature_scales, r_len = r_len)
    # print("scatter_plot_subgroup", results)

    df = pd.read_csv('./data/test_scatter_plot.csv');
    return json.dumps(df.values.tolist())


# @app.route('/getLegendConfiguration',  methods = ['POST'])
# def get_legend_config():
#     post_data = json.loads(request.data.decode())
#     station_id = post_data['StationId']
#     config = dataService.get_legend_config(station_id)
#     return json.dumps(config)


if __name__ == '__main__':
    pass
