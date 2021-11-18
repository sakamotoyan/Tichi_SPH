import shutil

import pandas as pd
import numpy as np
import os

SAMPLE_FOLDER = r'./sample_csv'
SCENE_ID = 0
SAVE_FOLDER = os.path.join(SAMPLE_FOLDER, str(SCENE_ID))
if os.path.exists(SAVE_FOLDER):
    print('SAVE_FOLDER exists!')
os.makedirs(SAVE_FOLDER, exist_ok=True)

TABLE_TITLE = ['pos0', 'pos1', 'pos2', 'vel0', 'vel1', 'vel2', 'mass']


def save_csv(fps, fluid, solid):
    save_path = os.path.join(SAVE_FOLDER, str(fps) + r'.csv')
    fluid_num = fluid.part_num[None]
    fluid_data = np.hstack((fluid.pos.to_numpy()[:fluid_num], fluid.vel.to_numpy()[:fluid_num],
                            fluid.mass.to_numpy()[:fluid_num, np.newaxis]))

    solid_num = solid.part_num[None]
    solid_data = np.hstack((solid.pos.to_numpy()[:solid_num], solid.vel.to_numpy()[:solid_num],
                            0 - solid.mass.to_numpy()[:solid_num, np.newaxis]))

    df = pd.DataFrame(np.vstack([fluid_data, solid_data]), columns=TABLE_TITLE)
    df.to_csv(save_path, index=False)


def save_scene_config(config_file, scenario_file):
    shutil.copy(config_file, os.path.join(SAVE_FOLDER, r'config.json'))
    shutil.copy(scenario_file, os.path.join(SAVE_FOLDER, r'scenario.json'))
    print('save scene config!')
