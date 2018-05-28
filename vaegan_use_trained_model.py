from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Flatten, Conv2DTranspose, Lambda, Reshape
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

total = Model(encoder_input,discriminator(decoder(encoder(encoder_input)[2])),name='total')
total.load_weights('model.h5')

x = []
random_indices = np.random.randint(202001, 202599, size=8)
for i in random_indices:
	x += [imageio.imread('64x64_celeba\%06d.jpg'%i)]
x_test = np.array(x)

z_test = encoder.predict(x_test)[2]
reconstructed = decoder.predict(z_test)
z_from_noise = np.random.normal(size = (8,2048))
new_images = decoder.predict(z_from_noise)

reconstructed = list(((reconstructed+1)*128).astype(np.uint8))
new_images = list(((new_images+1)*128).astype(np.uint8))
x_test = list(x_test)
random_indices = random_indices.tolist()

t = 0
while t<8:
	imageio.imwrite('test\%06d.jpg'%random_indices[t],np.concatenate((x_test[t],reconstructed[t]),1))
	imageio.imwrite('test\created%d.jpg'%t,new_images[t])
	t+=1