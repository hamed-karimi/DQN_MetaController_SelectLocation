import matplotlib.pyplot as plt
from MetaControllerVisualizer import MetaControllerVisualizer
from ObjectFactory import ObjectFactory
from Utilities import Utilities
import torch
import numpy as np
import os
from MetaControllerVisualizer import MetaControllerVisualizer


def test_meta_controller(meta_controller_dir, test_num=3):
    utilities = Utilities()
    params = utilities.params
    factory = ObjectFactory(utilities)

    environment_initialization_prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
    object_amount_options = ['few', 'many']

    meta_controller = factory.get_meta_controller(trained_path=os.path.join(meta_controller_dir,
                                                                            'meta_controller_model.pt'))
    meta_controller_visualizer = MetaControllerVisualizer(utilities)

    for test_trial in range(test_num):
        episode_object_amount = [np.random.choice(object_amount_options) for _ in range(params.OBJECT_TYPE_NUM)]
        pre_located_objects_location = [[[]]] * params.OBJECT_TYPE_NUM
        pre_located_objects_num = torch.zeros((params.OBJECT_TYPE_NUM,), dtype=torch.int32)
        pre_located_agent = [[]]
        pre_assigned_needs = [[]]
        test_agent = factory.get_agent(pre_located_agent,
                                       pre_assigned_needs)
        test_environment = factory.get_environment(episode_object_amount,
                                                   environment_initialization_prob_map,
                                                   pre_located_objects_num,
                                                   pre_located_objects_location)

        for fig, ax, name in meta_controller_visualizer.policynet_values(test_environment, meta_controller):
            fig.savefig('{0}/Trial_{1}_{2}.png'.format(meta_controller_dir, test_trial, name))
            plt.close()
