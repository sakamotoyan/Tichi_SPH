import pandas as pd
import json
import os

if __name__ == '__main__':
    df = pd.read_csv('./table.csv', header=None)
    sample_json = json.load(open(r'../scenario_example/two_phases.json'))
    config_json = json.load(open(r'../config_example/config_3d.json'))
    for i in range(len(df)):
        _data = df.iloc[i]
        _json = sample_json
        _json['fluid'][0]['start_pos'][0] = _data.iloc[6]
        _json['fluid'][0]['start_pos'][2] = _data.iloc[7]
        _json['fluid'][0]['end_pos'][0] = _data.iloc[4]
        _json['fluid'][0]['end_pos'][2] = _data.iloc[5]
        _json['fluid'][1]['start_pos'][0] = _data.iloc[2]
        _json['fluid'][1]['start_pos'][2] = _data.iloc[3]
        _json['fluid'][1]['end_pos'][0] = _data.iloc[0]
        _json['fluid'][1]['end_pos'][2] = _data.iloc[1]
        json.dump(_json, open(r'scene_temp.json', 'w'))

        _json = config_json
        _json['auto_start'] = True
        _json['auto_stop'] = True
        _json['save_csv'] = True
        _json['save_csv_id'] = i+200
        json.dump(_json, open(r'config_temp.json', 'w'))

        os.system(r'python ..\main.py -c config_temp.json -s scene_temp.json')

