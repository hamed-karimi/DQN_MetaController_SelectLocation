import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
from MetaControllerVisualizer import MetaControllerVisualizer
from Visualizer import get_reward_plot, get_loss_plot
from ObjectFactory import ObjectFactory
from Utilities import Utilities
from AgentExplorationFunctions import *


def training_meta_controller():
    utility = Utilities()
    params = utility.params

    res_folder = utility.make_res_folder(sub_folder='MetaController')
    # utility.save_training_config()
    writer = SummaryWriter()

    factory = ObjectFactory(utility)
    controller = factory.get_controller()
    meta_controller = factory.get_meta_controller(trained_path='')
    meta_controller_visualizer = MetaControllerVisualizer(utility)
    environment_initialization_prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)

    for episode in range(params.META_CONTROLLER_EPISODE_NUM):
        episode_begin = True
        episode_meta_controller_reward = 0
        episode_meta_controller_loss = 0
        # all_actions = 0
        pre_located_objects_location = [[[]]] * params.OBJECT_TYPE_NUM
        pre_located_objects_num = torch.zeros((params.OBJECT_TYPE_NUM,), dtype=torch.int32)
        pre_located_agent = [[]]
        pre_assigned_needs = [[]]
        object_amount_options = ['few', 'many']
        episode_object_amount = [np.random.choice(object_amount_options) for _ in range(params.OBJECT_TYPE_NUM)]
        # Initialized later in the reached if-statement
        environment = None
        agent = None

        for goal_selecting_step in range(params.EPISODE_LEN):
            steps = 0
            # steps_rho = []
            step_satisfactions = []
            step_moving_costs = []
            step_needs_costs = []
            if episode_begin:
                agent = factory.get_agent(pre_located_agent,
                                          pre_assigned_needs)
                environment = factory.get_environment(episode_object_amount,
                                                      environment_initialization_prob_map,
                                                      pre_located_objects_num,
                                                      pre_located_objects_location)
                episode_begin = False

            # env_map_0 = environment.env_map.clone()
            # need_0 = agent.need.clone()
            goal_map, goal_location = meta_controller.get_goal_map(environment,
                                                                   agent,
                                                                   episode)  # goal type is either 0 or 1
            done = torch.tensor([0])
            while True:
                env_map_0 = environment.env_map.clone()
                need_0 = agent.need.clone()
                agent_goal_map_0 = torch.stack([environment.env_map[:, 0, :, :], goal_map], dim=1)

                action_id = controller.get_action(agent_goal_map_0).clone()
                satisfaction, moving_cost, needs_cost, dt = agent.take_action(environment, action_id)
                # step_satisfactions.append(satisfaction)
                # step_moving_costs.append(moving_cost)
                # step_needs_costs.append(needs_cost)

                # steps_rho.append(rho)

                goal_reached = agent_reached_goal(environment, goal_map)

                steps += 1

                # if goal_reached or steps == 1:  # taking only the 1st step
                pre_located_objects_location = update_pre_located_objects(environment.object_locations,
                                                                          agent.location,
                                                                          goal_reached)
                pre_located_objects_num = environment.each_type_object_num
                pre_located_agent = agent.location.tolist()
                pre_assigned_needs = agent.need.tolist()

                agent = factory.get_agent(pre_located_agent,
                                          pre_assigned_needs)

                environment = factory.get_environment(episode_object_amount,
                                                      environment_initialization_prob_map,
                                                      pre_located_objects_num,
                                                      pre_located_objects_location)
                # satisfaction_tensor = torch.tensor(step_satisfactions)
                # moving_cost_tensor = torch.tensor(step_moving_costs)
                #
                # needs_cost_tensor = torch.tensor(step_needs_costs)

                steps_reward = (satisfaction - needs_cost - moving_cost).unsqueeze(dim=0)
                dt = torch.tensor([dt])
                steps_discounts = torch.pow(torch.ones(dt.shape[0]) * params.GAMMA,
                                            torch.arange(dt.shape[0]) * dt).flip(dims=(0,))
                discounted_reward = (steps_reward * steps_discounts).sum().unsqueeze(dim=0)

                meta_controller.save_experience(env_map_0, need_0, goal_map, discounted_reward, done,
                                                dt, environment.env_map.clone(), agent.need.clone())

                if goal_reached or steps == params.EPISODE_STEPS:
                    break

            episode_meta_controller_reward += discounted_reward
            at_loss = meta_controller.optimize()
            episode_meta_controller_loss += get_meta_controller_loss(at_loss)
        if episode_meta_controller_loss > 0:
            meta_controller.lr_scheduler.step()
        writer.add_scalar("Meta Controller/Loss", episode_meta_controller_loss / params.EPISODE_LEN, episode)
        writer.add_scalar("Meta Controller/Reward", episode_meta_controller_reward / params.EPISODE_LEN, episode)

        if (episode + 1) % params.PRINT_OUTPUT == 0:
            pre_located_objects_location = [[[]]] * params.OBJECT_TYPE_NUM
            pre_located_objects_num = torch.zeros((params.OBJECT_TYPE_NUM,), dtype=torch.int32)
            test_environment = factory.get_environment(episode_object_amount,
                                                       environment_initialization_prob_map,
                                                       pre_located_objects_num,
                                                       pre_located_objects_location)

            fig, ax = meta_controller_visualizer.get_goal_directed_actions(test_environment,
                                                                           meta_controller,
                                                                           controller)
            fig.savefig('{0}/episode_{1}.png'.format(res_folder, episode + 1))
            plt.close()

            # for fig, ax, name in meta_controller_visualizer.policynet_values(test_environment, meta_controller):
            #     fig.savefig('{0}/{1}.png'.format(res_folder, name))
            #     plt.close()

        if (episode + 1) % params.META_CONTROLLER_TARGET_UPDATE == 0:
            meta_controller.update_target_net()
            print('META CONTROLLER TARGET NET UPDATED')

    return meta_controller, res_folder
