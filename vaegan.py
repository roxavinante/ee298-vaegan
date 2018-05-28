from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Flatten, Conv2DTranspose, Lambda, Reshape
from keras.callbacks import ModelCheckpoint
from keras.losses import mse, binary_crossentropy
import numpy as np
from keras import backend as K
import utilities
import imageio


def sampling(args):
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon


#making the encoder model
encoder_input = Input(shape = (64,64,3),
	name = 'encoder_input')
enc = Conv2D(filters = 64,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'encoder_first_conv')(encoder_input)
enc = BatchNormalization()(enc)
enc = Conv2D(filters = 128,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'encoder_second_conv')(enc)
enc = BatchNormalization()(enc)
enc = Conv2D(filters = 256,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'encoder_third_conv')(enc)
enc = BatchNormalization()(enc)
enc = Flatten()(enc)
z_mean = Dense(units = 2048,
	activation = 'relu',
	name = 'encoder_output_mean')(enc)
z_mean = BatchNormalization()(z_mean)
z_log_var = Dense(units = 2048,
	name = 'encoder_output_log_var')(enc)
z_log_var = BatchNormalization()(z_log_var)
z_sampled = Lambda(sampling,
	name = 'sampler')([z_mean,z_log_var])
encoder = Model(encoder_input,[z_mean,z_log_var,z_sampled],name='encoder')
encoder.summary()

#making the decoder model
decoder_input = Input(shape = (2048,),
	name = 'decoder_input')
dec = Dense(units = 8*8*256,
	activation = 'relu',
	name = 'decoder_first_layer')(decoder_input)
dec = BatchNormalization()(dec)
dec = Reshape((8,8,256))(dec)
dec = Conv2DTranspose(filters = 256,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'decoder_first_conv')(dec)
dec = BatchNormalization()(dec)
dec = Conv2DTranspose(filters = 128,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'decoder_second_conv')(dec)
dec = BatchNormalization()(dec)
dec = Conv2DTranspose(filters = 32,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'decoder_third_conv')(dec)
dec = BatchNormalization()(dec)
reconstruction = Conv2DTranspose(filters = 3,
	kernel_size = 5,
	padding = 'same',
	activation = 'tanh',
	name = 'decoder_fourth_conv')(dec)
decoder = Model(decoder_input,reconstruction,name='decoder')
decoder.summary()

#making the discriminator model
discriminator_input = Input(shape = (64,64,3),
	name = 'discriminator_input')
dis = Conv2D(filters = 32,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'discriminator_first_conv')(discriminator_input)
dis = BatchNormalization()(dis)
dis = Conv2D(filters = 128,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'discriminator_second_conv')(dis)
dis = BatchNormalization()(dis)
dis = Conv2D(filters = 256,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'discriminator_third_conv')(dis)
Disl = BatchNormalization()(dis)
discrim = Conv2D(filters = 256,
	kernel_size = 5,
	strides = (2,2),
	padding = 'same',
	activation = 'relu',
	name = 'discriminator_fourth_conv')(Disl)
discrim = BatchNormalization()(discrim)
discrim = Flatten()(discrim)
discrim = Dense(units = 512,
	activation = 'relu',
	name = 'discriminator_first_output_layer')(discrim)
discrim = BatchNormalization()(discrim)
judgement = Dense(units = 1,
	activation = 'sigmoid',
	name = 'discriminator_second_output_layer')(discrim)
discriminator = Model(discriminator_input,[Disl,judgement],name='discriminator')
discriminator.summary()

#making the loss functions
def encoder_loss(y,ypred):
	Ldisl = 0.5*mse(y,ypred)
	Lprior = - 0.5 * K.mean( 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
	return Ldisl+Lprior

def decoderLdisl(y,ypred):
	gamma = 1e-6
	Ldisl = 0.5*mse(y,ypred)
	return gamma*Ldisl

discriminator_only = Model(discriminator_input,discriminator(discriminator_input)[1],name='discriminator_only')
discriminator_only.compile(loss=binary_crossentropy, optimizer = 'rmsprop')
discriminator.trainable = False
decoder_only = Model(decoder_input,[discriminator(decoder(decoder_input))[0],discriminator(decoder(decoder_input))[1]],name='decoder_only')
decoder_only.compile(loss = [decoderLdisl,binary_crossentropy], optimizer = 'rmsprop')
generator_only = Model(decoder_input,discriminator(decoder(decoder_input))[1],name='generator_only')
generator_only.compile(loss = binary_crossentropy , optimizer = 'rmsprop')
decoder.trainable = False
encoder_only = Model(encoder_input,discriminator(decoder(encoder(encoder_input)[2]))[0],name='encoder_only')
encoder_only.compile(loss = encoder_loss, optimizer = 'rmsprop')
total = Model(encoder_input,discriminator(decoder(encoder(encoder_input)[2])),name='total')

#loading the images
print("loading the images for training:")
images_to_load = 202000
t = 1
x = []
while t <= images_to_load :
	x += [imageio.imread('64x64_celeba\%06d.jpg'%t)]
	utilities.show_loop_progress(t,images_to_load)
	t += 1
x_train = np.array(x)

batch_size = 64
train_steps = 50000
save_interval = 20

for i in range(train_steps):
	random_indices = np.random.randint(0, x_train.shape[0], size=batch_size)
	x_batch = x_train[random_indices]
	z_from_batch = encoder.predict(x_batch)[2]
	z_from_noise = np.random.normal(size = (batch_size,2048))
	reconstructed = decoder.predict(z_from_batch)
	new_images = decoder.predict(z_from_noise)
	Disl_from_batch = discriminator.predict(x_batch)[0]
	metrics = encoder_only.train_on_batch(x_batch,Disl_from_batch)
	log = "%d: [encoder loss: %f]" % (i, metrics)
	metrics = decoder_only.train_on_batch(z_from_batch,[Disl_from_batch,np.ones(batch_size)])
	log = "%s: [decoder content loss: %f, style loss: %f]" % (log, metrics[0],metrics[1])
	metrics = generator_only.train_on_batch(z_from_noise,np.ones(batch_size))
	log = "%s: [decoder loss from noise: %f]" % (log, metrics)
	discriminator_training_set = np.concatenate((reconstructed,new_images,x_batch))
	discriminator_answer_key = np.concatenate((np.zeros(2*batch_size),np.ones(batch_size)))
	metrics = discriminator_only.train_on_batch(discriminator_training_set,discriminator_answer_key)
	log = "%s: [discriminator loss: %f]" % (log, metrics)
	print(log)
	if (i+1)%save_interval == 0:
		total.save("model.h5")