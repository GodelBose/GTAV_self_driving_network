import numpy as np
import cv2
import matplotlib.pyplot as plt

class Data_generator:


	def __init__(self, files_per_batch, total_files, files_directory, cam_view=True, map_view=True, speed_view=True, view_resize=None, return_axis=True, return_buttons=True, axis_indices =[], button_indices=[], seq_len=1, use_sampling=False):
		self.files_per_batch = files_per_batch
		self.total_files = total_files
		self.files_directory = files_directory
		self.cam_view = cam_view
		self.map_view = map_view
		self.speed_view = speed_view
		self.view_resize = view_resize
		self.return_axis = return_axis
		self.return_buttons = return_buttons
		self.axis_indices = axis_indices
		self.button_indices = button_indices
		self.seq_len = seq_len
		self.use_sampling = use_sampling

	def yield_data(self):
		indices = np.random.randint(0,self.total_files,self.files_per_batch)
		X = []
		Y = []
		num_inputs = self.map_view + self.cam_view + self.speed_view
		num_outputs = self.return_axis + self.return_buttons
		random_radius = np.random.uniform(0.02,0.1)
		normalizer = np.pi * random_radius**2
		for i in indices:
			file_name = self.files_directory+'/'+'training_data-{}.npy'.format(i) if self.files_directory else 'training_data-{}.npy'.format(i)
			train_data = np.load(file_name)
			for j, x in enumerate(train_data):
				temp_inputs = []
				temp_labels = []
				# randomly discard samples with no steering to remove bias towards not steering
				if self.use_sampling:
					x_i,y_i = x[3][0], x[3][1]
					radius = np.sqrt(x_i**2 + y_i**2)
					volume = np.pi * radius**2
					ratio = volume / normalizer
					rand_f = np.random.uniform(0,1)
					if rand_f > ratio:
						continue

				if self.map_view:
					temp_inputs.append(x[1])
				if self.cam_view:
					# only consider 1 frame at a time
					if self.seq_len == 1:
						if self.view_resize:
							screen = cv2.resize(x[0], self.view_resize)
							temp_inputs.append(screen)
						else:
							temp_inputs.append(x[0])
					else:
						# consider multiframe scenario
						if j >= self.seq_len:
							views = [xx[0] for xx in train_data[j-self.seq_len:j]]
							if self.view_resize:
								views = [cv2.resize(xx, self.view_resize) for xx in views]
							temp_inputs.append(np.array(views))
						else:
							continue
				if self.speed_view:
					temp_inputs.append(x[2][:,:,None])

				if self.return_axis:
					temp_labels.append([x[3][k] for k in self.axis_indices])
				if self.return_buttons:
					temp_labels.append([x[4][k] for k in self.button_indices])
				X.append(temp_inputs)
				Y.append(temp_labels)
		#print(len(X[0]), len(X))
		return [np.array([x[i] for x in X]) for i in range(num_inputs)],  [np.array([y[i] for y in Y]) for i in range(num_outputs)]
