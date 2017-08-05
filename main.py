from gta_driver import GTA_driver
from models import get_vis_comb_2
from data_generator import Data_generator
from models import steering_model, get_model, get_3d_model
# regions to grab for input
screen_region = (0,20, 800, 620)
map_region = (10,500,160,615)
speed_region = (730,320,760,350)
seq_len = 5
# eventual shapes that the model gets
#screen_shape = (120,180,3) #2d input
screen_shape = (seq_len, 120,180,3)
map_shape = (116,151,3)
speed_shape = (31,31,1)

epochs = 50
load_model_name = ''
save_model_name = '3d_input.model'
batch_size = 4
frame_rate = 15
train = True
driving = False
#model = get_vis_comb_2(map_shape, screen_shape, speed_shape)
#model = steering_model(screen_shape, speed_shape)
#model = get_model(screen_shape)
model = get_3d_model(screen_shape)

gen = Data_generator(10, 776, '', cam_view=True,
                     map_view=False, speed_view=False,
                     view_resize=(180,120), return_axis=True,
                     return_buttons=False, axis_indices =[0,1], seq_len=seq_len, use_sampling=True)

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
