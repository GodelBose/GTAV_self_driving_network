import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os
import pyxinput
import matplotlib.pyplot as plt
import pygame
from getkeys import key_check
from directkeys import PressKey,ReleaseKey, W, A, S, D

class GTA_driver:

	def __init__(self, data_gen, epochs, load_model_name, save_model_name, batch_size, compiled_model, cam_resolution, frame_rate, cam_region=None, map_region=None, speed_region=None):
		self.data_gen = data_gen
		self.epochs = epochs
		self.load_model_name = load_model_name
		self.save_model_name = save_model_name
		self.batch_size = batch_size
		self.model = compiled_model
		self.save_per_iteration = 10
		self.cam_resolution = cam_resolution
		self.inputs = {'map_view':data_gen.map_view, 'cam_view':data_gen.cam_view, 'speed_view':data_gen.speed_view}
		self.clock = pygame.time.Clock()
		self.frame_rate = frame_rate
		self.cam_region = cam_region
		self.speed_region = speed_region
		self.map_region = map_region

	def train_model(self):
		iteration = 0
		for epoch in range(self.epochs):
			instances = 0
			print("Starting epoch %d"%epoch)
			while True:
				X,y = self.data_gen.yield_data()
				try:
					self.model.fit(X, y, epochs=1, batch_size=self.batch_size, shuffle=True)
					instances += self.data_gen.files_per_batch * 550
					if iteration % self.save_per_iteration == 0:
						print('SAVING MODEL!')
						np.save(self.save_model_name, self.model.get_weights())
					iteration += 1
				except Exception as e:
					print(str(e))
				# epoch end
				if instances >= 550 * self.data_gen.total_files:
					break

	def load_model(self):
		if self.load_model_name:
			self.model.set_weights(np.load(self.load_model_name))

	def predict(self,X):
		return self.model.predict(X)

	def make_input(self, ax_predictions, speed_predictions, controller):
		ax_value = {0:'AxisLx', 1:'AxisLy', 2:'AxisRx', 3:'AxisRy', 4:'TriggerR', 5:'TriggerL'}
		for j,i in enumerate(self.data_gen.axis_indices):
			controller.set_value(ax_value[i], ax_predictions[j])


	def live_driving(self):
		controller = pyxinput.vController()
		paused = False
		while(True):
			if not paused:
				map_screen = grab_screen(region=self.map_region)
				map_screen = cv2.cvtColor(map_screen, cv2.COLOR_BGR2RGB)
				cam_screen = grab_screen(region=self.cam_region)
				cam_screen = cv2.cvtColor(cam_screen, cv2.COLOR_BGR2RGB)
				cam_screen = cv2.resize(cam_screen, self.cam_resolution[:2])
				if self.data_gen.view_resize:
					cam_screen = cv2.resize(cam_screen, self.data_gen.view_resize)
				speed_screen = grab_screen(region=self.speed_region)
				speed_screen = cv2.cvtColor(speed_screen, cv2.COLOR_BGR2RGB)
				speed_screen = cv2.cvtColor(speed_screen,cv2.COLOR_RGB2GRAY)[:,:,None]
				X = [x for name,x in zip(['map_view', 'cam_view', 'speed_view'],[map_screen[None,:,:,:], cam_screen[None,:,:,:], speed_screen[None,:,:,:]]) if self.inputs[name]]
				if self.data_gen.return_buttons:
					ax_predictions, button_predictions = self.predict(X)
				else:
					ax_predictions = self.predict(X)
					button_predictions = []
				print(ax_predictions, X[0].shape, X[1].shape)
				self.make_input(ax_predictions[0], button_predictions, controller)
				self.clock.tick(self.frame_rate)
				keys = key_check()

				# p pauses game and can get annoying.
				if 'T' in keys:
					if paused:
						paused = False
						time.sleep(1)
					else:
						c = 0
						paused = True
						ReleaseKey(A)
						ReleaseKey(W)
						ReleaseKey(D)
						time.sleep(1)
				if 'Q' in keys:
					break
