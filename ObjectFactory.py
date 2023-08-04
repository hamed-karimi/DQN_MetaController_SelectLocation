from Agent import Agent
from Environment import Environment
from Controller import Controller
from MetaController import MetaController
import inspect
from copy import deepcopy


class ObjectFactory:
    def __init__(self, utility):
        self.agent = None
        self.environment = None
        self.controller = None
        self.meta_controller = None
        self.params = utility.params

    def get_agent(self, pre_location, preassigned_needs):
        agent = Agent(self.params.HEIGHT, self.params.WIDTH, n=self.params.OBJECT_TYPE_NUM,
                      prob_init_needs_equal=self.params.PROB_OF_INIT_NEEDS_EQUAL, predefined_location=pre_location,
                      rho_function=self.params.RHO_FUNCTION,
                      epsilon_function=self.params.EPSILON_FUNCTION,
                      preassigned_needs=preassigned_needs)
        self.agent = deepcopy(agent)
        return agent

    def get_environment(self, few_many, probability_map, pre_located_objects_num, pre_located_objects_location):  # pre_located_objects is a 2D list
        curr_frame = inspect.currentframe()
        call_frame = inspect.getouterframes(curr_frame, 2)
        if 'meta_controller' in call_frame[1][3]:
            num_objects = self.params.OBJECT_TYPE_NUM
        else:
            num_objects = 1
        env = Environment(few_many, self.params.HEIGHT, self.params.WIDTH, self.agent, probability_map,
                          reward_of_object=self.params.REWARD_OF_OBJECT,
                          far_objects_prob=self.params.PROB_OF_FAR_OBJECTS_FOR_TWO,
                          num_object=num_objects,
                          pre_located_objects_num=pre_located_objects_num,
                          pre_located_objects_location=pre_located_objects_location)
        self.environment = deepcopy(env)
        return env

    def get_controller(self):
        controller = Controller(self.params.HEIGHT,
                                self.params.WIDTH)

        self.controller = deepcopy(controller)
        return controller

    def get_meta_controller(self):
        meta_controller = MetaController(self.params.META_CONTROLLER_BATCH_SIZE,
                                         self.params.OBJECT_TYPE_NUM,
                                         self.params.GAMMA,
                                         self.params.META_CONTROLLER_EPISODE_NUM,
                                         self.params.EPISODE_LEN,
                                         self.params.META_CONTROLLER_MEMORY_CAPACITY,
                                         self.params.REWARDED_ACTION_SAMPLING_PROBABILITY_RATIO)
        self.meta_controller = deepcopy(meta_controller)
        return meta_controller

    def get_saved_objects(self):
        return deepcopy(self.agent), deepcopy(self.environment), \
               deepcopy(self.controller), deepcopy(self.meta_controller)
