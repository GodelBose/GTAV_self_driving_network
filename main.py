from gta_driver import GTA_driver
from models import get_vis_comb_2
from data_generator import Data_generator
from models import two_input_model, get_model, get_3d_model
import os
import sys
import shutil

experiment_path = sys.argv[1]
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)
shutil.copy('main.py', experiment_path+'/'+'main.py')
shutil.copy('models.py', experiment_path+'/'+'models.py')
# regions to grab for input
screen_region = (0,20, 800, 620)
map_region = (10,500,160,615)
speed_region = (730,320,760,350)
seq_len = 1
# eventual shapes that the model gets
screen_shape = (240,360,3) #2d input
#screen_shape = (seq_len, 120,180,3) #3d input
map_shape = (116,151,3)
speed_shape = (31,31,1)

epochs = 50
load_model_name = ''
save_model_name = experiment_path+'/comb_model'
batch_size = 32
frame_rate = 18
train = True
driving = False
#model = get_vis_comb_2(map_shape, screen_shape, speed_shape)
model = two_input_model(screen_shape, speed_shape)
#model = get_model(screen_shape)
#model = get_3d_model(screen_shape)

gen = Data_generator(15, 776, '', cam_view=True,
                     map_view=False, speed_view=True,
                     view_resize=None, return_axis=True,
                     return_buttons=False, axis_indices =[0,1,4,5],
                      seq_len=seq_len, use_sampling=True)

driver = GTA_driver(gen, epochs, load_model_name,
                    save_model_name, batch_size, model,
                    screen_shape, frame_rate, cam_region=screen_region,
                    map_region=(), speed_region=speed_region)

if load_model_name:
    driver.load_model()
if train:
    driver.train_model()
if driving:
    driver.live_driving()
