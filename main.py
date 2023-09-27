import os.path
from datetime import datetime

from AgentExplorationFunctions import *
from MetaControllerTraining import training_meta_controller
from Utilities import Utilities
from TestMetaController import test_meta_controller

utility = Utilities()
params = utility.params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta_controller_dir = params.PRETRAINED_META_CONTROLLER
if params.TRAIN_META_CONTROLLER:
    start_time = datetime.now()
    meta_controller, meta_controller_dir = training_meta_controller(utility)
    meta_controller_model_path = os.path.join(meta_controller_dir, 'meta_controller_model.pt')
    torch.save(meta_controller.target_net.state_dict(), meta_controller_model_path)

if params.TEST_Q_VALUES:
    test_meta_controller(utility, meta_controller_dir)
