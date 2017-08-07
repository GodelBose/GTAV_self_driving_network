from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D
from keras.layers.pooling import MaxPooling3D
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend
from keras.layers.merge import Concatenate
import numpy as np
from keras.layers.recurrent import GRU
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from data_generator import Data_generator

class GAN_driver:

	def __init__(self, cam_shape, data_gen):
		self.cam_shape = cam_shape
		self.data_gen = data_gen
		print("init")

	def generator_model(self):
		b = Input(shape=self.cam_shape)
		c = Convolution2D(36, 5, strides=(2, 2), padding="same", activation='relu')(b)
		c = BatchNormalization()(c)
		c = Convolution2D(48, 5, strides=(2, 2), padding="same", activation='relu')(c)
		c = BatchNormalization()(c)
		c = Convolution2D(64, 5, strides=(2, 2), activation='relu',padding="same")(c)
		c = BatchNormalization()(c)
		c = Convolution2D(96, 3, activation='relu',padding="same")(c)
		c = BatchNormalization()(c)
		c = Convolution2D(96, 3, activation='relu',padding="same")(c)
		c = BatchNormalization()(c)
		c = Flatten()(c)
		c = Dense(1524,activation='elu')(c)
		c = BatchNormalization()(c)
		c = Dense(750,activation='elu')(c)
		c = BatchNormalization()(c)
		c = Dense(370,activation='elu')(c)
		c = BatchNormalization()(c)
		c = Dense(180,activation='elu')(c)
		c = BatchNormalization()(c)
		c = Dense(50,activation='elu')(c)
		c = BatchNormalization()(c)
		c = Dense(2,activation='linear')(c)
		model = Model(input=b, output=c)
		return model


	def discriminator_model(self):
		b = Input(shape=self.cam_shape)
		c = Input(shape=(2,))

		c1 = Convolution2D(64, 5, strides=(2, 2), padding="same", activation ='elu')(b)
		c1 = Convolution2D(96, 3, strides=(1, 1), padding="same", activation ='elu')(c1)
		c1 = MaxPooling2D(pool_size=(3, 3))(c1)
		c1 = BatchNormalization()(c1)
		flatten1 = Flatten()(c1)

		merged_vector = Concatenate(axis=1)([flatten1, c])
		fc1 = Dense(1250, activation ='elu')(merged_vector)
		fc1 = Dense(1, activation='sigmoid')(fc1)

		model = Model(input=[b,c], output=fc1)
		return model


	def generator_containing_discriminator(self, g, d3):
		b = Input(shape=self.cam_shape)
		#c = Input(shape=(2,))
		x = g(b)
		d3.trainable = False
		x = d3([b,x])
		model2 = Model(input=b, output=x)
		return model2

	def fit(self):
		# init the graph
		g = gd.generator_model()
		d_optim = Adam(lr=0.001, clipvalue=5)
		d = gd.discriminator_model()
		d2 = gd.discriminator_model()
		gan = gd.generator_containing_discriminator(g,d2)
		g_optim = Adam(lr=0.0005, clipvalue=3)
		g.compile(loss='mean_squared_error', optimizer=g_optim)
		g.set_weights(np.load('steering_model_1.0.npy'))
		d.trainable = True
		d.compile(loss='binary_crossentropy', optimizer=d_optim)
		gan.compile(loss='binary_crossentropy', optimizer=g_optim)
		d2.compile(loss='binary_crossentropy', optimizer=d_optim)
		d2.trainable = False
		self.g = g
		self.d = d
		train_d = True
		train_g = True
		iteration = 0
		for epoch in range(1):
			instances = 0
			print("Starting epoch %d"%epoch)
			while True:
				X,y = self.data_gen.yield_data()
				X2,y2 = self.data_gen.yield_data()
				try:

					generated_commands = g.predict(X, verbose=0)

					X3 = np.concatenate([X[0],X2[0]],axis=0)
					X4 = np.concatenate([generated_commands, y2[0]],axis=0)
					y_disc = [0] * len(X[0]) + [1] * len(X2[0])

					d.fit([X3,X4], y_disc, batch_size=32, shuffle=True)
					d_predicts = d.predict([X3,X4])
					print(sum(d_predicts))
					# set the new learned weights for backpropagation
					d2.set_weights(d.get_weights())
					d2.trainable = False
					gan.fit(X, [1] * len(X[0]), batch_size=32, shuffle=True, epochs=1)
					g_predicts = gan.predict(X)
					print(sum(g_predicts))
					instances += 10 * 550
					if iteration % 10 == 0:
						print('SAVING MODEL!')
						np.save('gan_test'+'_d', self.d.get_weights())
						np.save('gan_test'+'_g', self.g.get_weights())
						plt.plot(generated_commands)
						plt.savefig("temp_imgs/predictions %d"%0)
						plt.close()
						plt.plot(y[0][:,0])
						plt.plot(y[0][:,1])
						plt.savefig("temp_imgs/labels %d"%0)
						plt.close()
					iteration += 1
				except Exception as e:
					print(str(e))
				# epoch end
				if instances >= 550 * 771:
					break

if __name__=='__main__':
	gen = Data_generator(5, 776, '', cam_view=True,
	                     map_view=False, speed_view=False,
	                     view_resize=(180,120), return_axis=True,
	                     return_buttons=False, axis_indices =[0,1],
	                      seq_len=1, use_sampling=True)
	gd = GAN_driver((120,180,3), gen)
	gd.fit()
	'''
	gd = GAN_driver((180,120,3))
	g = gd.generator_model()
	d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
	d = gd.discriminator_model()
	d2 = gd.discriminator_model()
	gan = gd.generator_containing_discriminator(g,d2)
	g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
	g.compile(loss='mean_squared_error', optimizer=g_optim)
	d.trainable = True
	d.compile(loss='binary_crossentropy', optimizer=d_optim)
	gan.compile(loss='binary_crossentropy', optimizer=g_optim)
	d2.compile(loss='binary_crossentropy', optimizer=d_optim)
	d2.trainable = False

	X = np.random.randn(25,180,120,3)
	X3 = np.random.randn(25,180,120,3)
	X2 = np.random.randn(25,2)
	generated_commands = g.predict(X, verbose=0)
	X4 = np.concatenate([X,X3],axis=0)
	X5 = np.concatenate([generated_commands, X2],axis=0)

	y_disc = [1] * 25 + [0] * 25
	print(X4.shape, X5.shape, len(y_disc))
	dloss = d.train_on_batch([X4,X5], y_disc)
	print(dloss)
	d2.set_weights(d.get_weights())
	print("weights set")
	g_loss = gan.train_on_batch(X, [1] * 25)
	'''
