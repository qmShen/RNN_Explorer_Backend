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



# @app.route('/testscatterplot')
# def get_test_scatter_plot():
#     print('here')
#     return json.dumps('test_scatter_plot')
n_unit_cluster = 10
n_feature_cluster = 12

@app.route('/')
def index():
    print('ere')
    return app.send_static_file('index.html')

@app.route('/test')
def getStationConfig():
    print('testx')
    return json.dumps("test")

@app.route('/test2')
def getStationConfig2():
    return json.dumps("test2")


@app.route('/model_list',  methods = ['POST'])
def read_model_list():
    print('Read Model List')
    model_list = dataService.read_model_list()
    return json.dumps(model_list)

# load_selected_model


@app.route('/load_selected_model',  methods = ['POST'])
def read_selected_model():
    post_data = json.loads(request.data.decode())
    print('Read Selected Model', post_data)
    dataService.read_selected_model(post_data['mid'])


    return json.dumps({
        'features': dataService.get_feature_stats(post_data['mid']),
        'units': dataService.get_units_stats(post_data['mid']),
        'cluster': dataService.get_cluster(post_data['mid'], n_unit_cluster=n_unit_cluster, n_feature_cluster=n_feature_cluster),


    })


@app.route('/all_stats',  methods = ['POST'])
def get_stats_data():
    post_data = json.loads(request.data.decode())
    print('Get all statistics', post_data)
    return json.dumps({
        'features': dataService.get_feature_stats(post_data['mid']),
        'units': dataService.get_units_stats(post_data['mid']),
        'cluster': dataService.get_cluster(post_data['mid'], n_unit_cluster = n_unit_cluster,  n_feature_cluster = n_feature_cluster)
    })


@app.route('/get_gradient_projection',  methods = ['POST'])
def get_gradient_projection():
    post_data = json.loads(request.data.decode())
    mid = post_data['mid']
    target_feature = post_data['target_feature']
    data = dataService.get_gradient_projection(mid, target_feature)
    print('Get gradient projection', post_data)
    return json.dumps(data)



# ------------------------------------------------------------------------------------
@app.route('/testscatterplot')
def get_test_scatter_plot():
    with open('./data/test_scatter_plot.json', 'r') as input_file:
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


@app.route('/feature_values',  methods = ['POST'])
def get_feature_values():

    post_data = json.loads(request.data.decode())
    print('Get feature values ',post_data)
    if 'features' not in post_data:
        dicts = dataService.get_feature_values(post_data['mid'])
    else:
        dicts = dataService.get_feature_values(post_data['mid'],post_data['features'])
    return json.dumps(dicts)


@app.route('/feature_values_scaled',  methods = ['POST'])
def get_feature_values_scaled():
    post_data = json.loads(request.data.decode())
    print('Get feature values ', post_data)
    if 'features' not in post_data:
        dicts = dataService.get_feature_values_scaled()
    else:
        dicts = dataService.get_feature_values_scaled(post_data['features'])
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
    print('Get scatter_plot_subgroup values xx', post_data)
    # feature_scales, r_len, dif_type
    # feature_scales = post_data['feature_scales']
    # r_len = 50
    sub_groups = post_data['sub_groups']


    # results = dataService.get_subgroup_scatter_plot(feature_scales = feature_scales, r_len = r_len)
    # print("scatter_plot_subgroup", results)

    # df = pd.read_csv('./data/test_scatter_plot_wind3class.csv');
    # df , all_selected_features = dataService.get_scatter_plot_by_sub_groups(sub_groups)
    df, all_selected_features = dataService.get_scatter_plot_by_sub_groups_sequence_pattern(sub_groups)

    return json.dumps({
        'data': df.values.tolist(),
        'selected_timestamps': all_selected_features})


# Sequence data
@app.route('/cell_input_output',  methods = ['POST'])
def get_gradient_and_io():
    post_data = json.loads(request.data.decode())
    print('Get cell input out ', post_data)
    return json.dumps(dataService.get_gradient_and_io_data(post_data['mid'],post_data['tid']))


@app.route('/sequence_cluster',  methods = ['POST'])
def get_selected_sequence_cluster():
    post_data = json.loads(request.data.decode())
    print('Get selected sequence cluster ', post_data)
    return json.dumps(dataService.get_gradient_and_io_data_by_cluster(post_data['mid'],post_data['tid'],
                      n_unit_cluster=n_unit_cluster, n_feature_cluster=n_feature_cluster ))
    # return json.dumps(dataService.get_gradient_and_io_data(post_data['mid'],post_data['tid']))

@app.route('/feature_gradient_cluster_to_end',  methods = ['POST'])
def get_feature_gradient_sequence_cluster_to_end():

    post_data = json.loads(request.data.decode())
    print('Get selected sequence cluster ',post_data)
    return json.dumps(dataService.get_feature_gradient_to_end(post_data['mid'], post_data['tid'],
                      n_unit_cluster=n_unit_cluster, n_feature_cluster=n_feature_cluster))


# @app.route('/getLegendConfiguration',  methods = ['POST'])
# def get_legend_config():
#     post_data = json.loads(request.data.decode())
#     station_id = post_data['StationId']
#     config = dataService.get_legend_config(station_id)
#     return json.dumps(config)

@app.route('/input_feature_gradient_statistics',  methods = ['POST'])
def get_input_feature_gradient_statistics():
    post_data = json.loads(request.data.decode())
    print('Get selected sequence cluster ', post_data)
    return json.dumps(dataService.get_feature_gradient_statistics(post_data['mid'], post_data['target_feature']))

if __name__ == '__main__':
    pass
