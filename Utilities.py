import json
from types import SimpleNamespace
import os
import shutil
from datetime import datetime
import numpy as np


class Utilities:
    def __init__(self):
        self.res_folder = None
        with open('./Parameters.json', 'r') as json_file:
            self.params = json.load(json_file,
                                    object_hook=lambda d: SimpleNamespace(**d))

    # def initialize_constants(self):
    #     self.params.CONV1_KERNEL = min(4, self.params.WIDTH)
    #     self.params.CONV1_OUT = 32
    #     self.params.LINEAR_IN = self.params.CONV1_OUT

    def make_res_folder(self, sub_folder=''):
        now = datetime.now().strftime("%d-%m-%Y_%H-%M")
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

    def save_training_config(self):
        config = {'EPISODE_NUM': self.params.META_CONTROLLER_EPISODE_NUM,
                  'META_CONTROLLER_TARGET_UPDATE': self.params.META_CONTROLLER_TARGET_UPDATE,
                  'CONTROLLER_BATCH_SIZE': self.params.CONTROLLER_BATCH_SIZE,
                  'META_CONTROLLER_BATCH_SIZE': self.params.META_CONTROLLER_BATCH_SIZE,
                  'CONTROLLER_MEMORY_CAPACITY': self.params.CONTROLLER_MEMORY_CAPACITY,
                  'META_CONTROLLER_MEMORY_CAPACITY': self.params.META_CONTROLLER_MEMORY_CAPACITY,
                  'GAMMA': self.params.GAMMA,
                  'REWARD_OF_OBJECT': self.params.REWARD_OF_OBJECT,
                  'PROB_OF_FAR_OBJECTS_FOR_TWO': self.params.PROB_OF_FAR_OBJECTS_FOR_TWO,
                  'PROB_OF_INIT_NEEDS_EQUAL': self.params.PROB_OF_INIT_NEEDS_EQUAL,
                  'Additional comments:': ('Getting the goal map at each step in the while loop. Saving the '
                                           'experiences, at each step.\nCost is the sum of needs.\n'
                                           'All memory experiences have the same probability.\n'
                                           '\nNo (1-done)')
                  }
        json_object = json.dumps(config, indent=4)
        with open(os.path.join(self.res_folder, 'config.json'), "w") as outfile:
            outfile.write(json_object)
