from app import app
import time

# The port number should be the same as the front end
# try:
# app.run(use_reloader=True, debug=True, port = 9930)
# except:
#     print("Some thing wrong!")

app.run(use_reloader=True, debug=True, port = 9930)


# from app.DataService.DataService import DataService
#
# if __name__ == '__main__':
#     import json
#     dataService = DataService(None)
#     start_time = time.time()
#     # unit_stats = dataService.get_and_save_feature_stats('GRU_1')
#     d = dataService.get_gradient_and_io_data_by_cluster('GRU_1', ['1514736000'])
#     # print(d)
#     result_json = json.dumps(d)
#
#     end_time = time.time()
#     print(end_time - start_time)
#
#     print(result_json)



