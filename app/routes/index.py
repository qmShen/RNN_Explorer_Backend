# -*- coding: UTF-8 -*-
from app import app
import json
from app.DataService.DataService import DataService
from flask import request


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




if __name__ == '__main__':
    pass
