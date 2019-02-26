# -*- coding: UTF-8 -*-
from app import app
import json
from app.DataService.DataService import DataService
from flask import request


dataService = DataService('config.txt')
print('here')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/test')
def getStationConfig():
    return json.dumps("test")

@app.route('/getStationMap',  methods = ['POST'])
def get_map():
    print('test')
    post_data = json.loads(request.data.decode())
    station_id = post_data['StationId']
    map = dataService.get_map(station_id)
    return json.dumps(map)

@app.route('/getLegendConfiguration',  methods = ['POST'])
def get_legend_config():
    post_data = json.loads(request.data.decode())
    station_id = post_data['StationId']
    config = dataService.get_legend_config(station_id)
    return json.dumps(config)

# getRecordWithTimeRange
@app.route('/getRecordWithTimeRange',  methods = ['POST'])
def get_realtime_data():
    post_data = json.loads(request.data.decode())
    station_id = post_data['StationId']
    start_time = post_data['starttime']
    time_range = post_data['timerange']
    data = dataService.get_recent_records(start_time / 1000, time_range / 1000)
    return json.dumps(data)

# getRecordWithTimeRange
@app.route('/getStationRecord',  methods = ['GET'])
def get_station_record():
    with open('config/point_positions2.csv', 'r') as input:
        line = input.readline()
        schemas = line.split(' ')
        schemas = [schema.strip() for schema in schemas]

        line = input.readline()
        station_records = []

        while line:
            segs = line.split()
            segs = [seg.strip() for seg in segs]
            stationObj = {}
            for i in range(0, len(schemas)):
                stationObj[schemas[i]] = segs[i]
            station_records.append(stationObj)
            line = input.readline()
    return json.dumps(station_records)

# getPeopleCount (Added by Qing Du)
@app.route('/getPeopleCount', methods = ['POST'])
def get_people_count():
    get_data = json.loads(request.data.decode())
    day = get_data['day']
    time = get_data['time']
    data = dataService.get_people_count(day, time)
    return json.dumps(data)


if __name__ == '__main__':
    pass
