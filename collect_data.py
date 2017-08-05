import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
import matplotlib.pyplot as plt
from speed_predictor import get_speed_predictor
import pygame
from getkeys import key_check
from aim_predictor import get_aim_predictor



pygame.init()

done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()

starting_value =0
while True:
    file_name = 'training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)

        break



def main(file_name, starting_value):
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    name = joystick.get_name()
    axes = joystick.get_numaxes()
    file_name = file_name
    starting_value = starting_value
    training_data = []

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    buttons = joystick.get_numbuttons()
    axes = joystick.get_numaxes()
    hats = joystick.get_numhats()

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    model = get_speed_predictor()
    aim_model = get_aim_predictor()
    while(True):
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        if 'Q' in keys:
            break
        if not paused:

            map_screen = grab_screen(region=(10,500,160,615))

            map_screen = cv2.cvtColor(map_screen, cv2.COLOR_BGR2RGB)
            # network that checks if the mini-map contains a target if not frame is skipped
            play_scene = aim_model.predict(map_screen[None,:,:,:])
            if play_scene[0] >= 0.5:
                temp_axis= []
                # grabbing axis values
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done=True


                for i in range(axes):
                    axis = joystick.get_axis(i)
                    temp_axis.append(axis)
                temp_button= []

                buttons = joystick.get_numbuttons()
                for i in range(buttons):
                    button = joystick.get_button(i)
                    temp_button.append(button)
                temp_hats = []

                hats = joystick.get_numhats()
                for i in range(hats):
                    hat = joystick.get_hat(i)
                    temp_hats.append(hat)

                # saving desktop starting right on the left top of the screen
                cam_screen = grab_screen(region=(0,20, 800, 620))
                cam_screen = cv2.cvtColor(cam_screen, cv2.COLOR_BGR2RGB)
                # grabbing input of the speedometer because after resizing the game window its not easily readable
                speed_screen =grab_screen(region=(730,320,760,350))
                speed_screen = cv2.cvtColor(speed_screen, cv2.COLOR_BGR2RGB)
                speed_screen = cv2.cvtColor(speed_screen,cv2.COLOR_RGB2GRAY)

                last_time = time.time()
                cam_screen = cv2.resize(cam_screen, (360,240))
                training_data.append([cam_screen, map_screen, speed_screen, np.array(temp_axis), np.array(temp_button), np.array(temp_hats)])

                last_time = time.time()

                if len(training_data) % 50 == 0:
                    print(len(training_data))
                    if len(training_data) == 550:
                        np.save(file_name,training_data)
                        plt.imshow(screen2)
                        # save update images of the last file to check for consistency
                        plt.savefig("temp_imgs/image %d"%0)
                        plt.close()
                        print('SAVED')
                        training_data = []
                        starting_value += 1
                        file_name = 'training_data-{}.npy'.format(starting_value)
                # record data at 15 frames a second
                clock.tick(15)


main(file_name, starting_value)
