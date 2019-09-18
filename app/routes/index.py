# -*- coding: UTF-8 -*-
from app import app
import json
from app.DataService.DataService import DataService
from flask import request
import pandas as pd
import numpy as np
import time

dataService = DataService()
print('here')

@app.route('/')
def index():

    return app.send_static_file('index.html')

@app.route('/cmaq_region')
def getregion():
    print('run here')
    return json.dumps(dataService.get_regions())

@app.route('/feature_data')
def get_feature_data():
    start_time = time.time()
    print('run herex')
    data = dataService.read_feature_data()

    print("Use timex", time.time() - start_time)
    # print(data)
    return json.dumps(data)
if __name__ == '__main__':
    pass
