import json
from types import SimpleNamespace
import os
import shutil
import pickle
from datetime import datetime
import numpy as np


class Utilities:
    def __init__(self):
        self.res_folder = None
        with open('./Parameters.json', 'r') as json_file:
            self.params = json.load(json_file,
                                    object_hook=lambda d: SimpleNamespace(**d))

    def make_res_folder(self, sub_folder=''):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = 'tr{0}'.format(now)
        dirname = os.path.join(folder, sub_folder)

        if os.path.exists(folder) and not os.path.exists(dirname):
            os.mkdir(dirname)
        elif not os.path.exists(dirname):
            os.makedirs(dirname)
        self.res_folder = dirname
        shutil.copy('./Parameters.json', self.res_folder)
        return dirname

    def get_environment_probability_map(self, style, params):  # style: 'equal', or 'edges'
        if style == 'equal':
            prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
        elif style == 'edges':
            prob_map = np.ones((params.HEIGHT, params.WIDTH))
            prob_map[[0, params.WIDTH - 1], :] *= 3
            prob_map[1:-1, [0, params.HEIGHT - 1]] *= 3

    def get_start_episode(self):
        if self.params.CHECKPOINTS_DIR != "":
            with open(os.path.join(self.params.CHECKPOINTS_DIR, 'train.pkl'), 'rb') as f:
                train_dict = pickle.load(f)
                return train_dict['episode']
        return 0
