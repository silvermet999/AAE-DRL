"""_________________________________________________import libraries_________________________________________________"""
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise,
                                     BatchNormalization, Activation, Embedding, ZeroPadding2D, MaxPooling2D,
                                     Lambda, LeakyReLU, UpSampling1D, Conv1D)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras import Input
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import main
import numpy as np
import datetime



tf.debugging.set_log_device_placement(True)
"""______________________________________________________archi______________________________________________________"""
class AdversarialAutoencoder():
    def __init__(self):
        self.input_vrt = 10
        self.input_hzt = 53439
        self.input_dim = (self.input_vrt, self.input_hzt)
        self.nn_dim = 128
        self.z_dim = 10

        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['loss'])

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        input = Input(shape=self.input_dim)
        encoded_repr = self.encoder(input)
        reconstructed_input = self.decoder(encoded_repr)

        self.discriminator.trainable = False
        validity = self.discriminator(encoded_repr)

        self.adversarial_autoencoder = Model(input, [reconstructed_input, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


    def build_encoder(self):
        input = Input(shape=self.input_dim)
        h = Flatten()(input)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        mu = Dense(self.z_dim)(h)
        log_var = Dense(self.z_dim)(h)
        def sampling(args):
            mu, log_var = args
            batch = tf.shape(mu)[0]
            dim = tf.shape(mu)[1]
            epsilon = tf.random_normal(shape=(batch, dim))
            return mu + K.exp(log_var / 2) * epsilon
        latent_repr = Lambda(sampling, output_shape=(self.z_dim,))([h, mu, log_var])
        enc_model = Model(input, latent_repr, name="encoder")
        return enc_model

    def build_decoder(self):
        dec_input = Input(shape=(self.z_dim,))
        h = Dense(512)(dec_input)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(np.prod(self.input_dim), activation='tanh')(h)
        h = Reshape(self.input_dim)(h)

        dec_model = Model(dec_input, h, name="decoder")
        dec_model.summary()
        return dec_model

    def build_discriminator(self):
        encoded_repr = Input(shape=(self.z_dim, ))
        h = Dense(512, input_dim=self.z_dim)(encoded_repr)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(256)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(1, activation="sigmoid")(h)
        disc_model = Model(encoded_repr, h, name="discriminator")
        disc_model.summary()


        return disc_model



"""_______________________________________________________vars_______________________________________________________"""
# lr = 0.02
# batch_size = 16
# epochs = 10
# input_dim = main.X_pca_rs.shape[0] * main.X_pca_rs.shape[1]
# nn_dim = 128
# z_dim = 10
#
# z_input = Input([None, z_dim], dtype=tf.float32, name="input_noise")
# x_input = Input([None, input_dim], dtype=tf.float32, name="real_noise")

# ae = ae_model(z_output, final_output)
# ae.compile(optimizer=Adam(), loss='mse', metrics=['loss'])
# ae.fit(main.x_train, main.x_train, epoch=10, batch_size=6)
AdversarialAutoencoder()



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
