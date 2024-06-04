from __future__ import print_function, division
from sklearn.model_selection import train_test_split
import main
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise,
                                     BatchNormalization, Activation, Embedding, ZeroPadding2D, MaxPooling2D,
                                     Lambda, LeakyReLU, UpSampling1D, Conv1D)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

class AdversarialAutoencoder():
    def __init__(self):
        self.input_rows = 28
        self.input_cols = 28
        self.channels = 1
        self.input_shape = (self.input_rows, self.input_cols, self.channels)
        self.latent_dim = 10

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        input = Input(shape=self.input_shape)
        # The generator takes the input, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(input)
        reconstructed_input = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(input, [reconstructed_input, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder

        input = Input(shape=self.input_shape)

        h = Flatten()(input)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        def sampling(args):
            mu, log_var = args
            batch = tf.shape(mu)[0]
            dim = tf.shape(mu)[1]
            epsilon = tf.random_normal(shape=(batch, dim))
            return mu + K.exp(log_var / 2) * epsilon
        latent_repr = Lambda(sampling, output_shape=(self.latent_dim,))([h, mu, log_var])

        return Model(input, latent_repr)

    def build_decoder(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.input_shape), activation='tanh'))
        model.add(Reshape(self.input_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        input = model(z)

        return Model(z, input)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def train(self, epochs, batch_size=1, sample_interval=5):

        # Load the dataset
        X_train, X_test, y_train, y_test = train_test_split(main.X_pca_rs, main.y, test_size=0.2, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(main.X_pca_mas, main.y, test_size=0.2, random_state=42)

        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of samples
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            inputs = X_train[idx]

            latent_fake = self.encoder.predict(inputs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(inputs, [inputs, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated samples
            if epoch % sample_interval == 0:
                self.gen_samples(epoch)

    def gen_samples(self, epoch):
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_inputs = self.decoder.predict(z)

        gen_inputs = 0.5 * gen_inputs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_inputs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("runs/_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "aae_generator")
        save(self.discriminator, "aae_discriminator")


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=10, batch_size=6, sample_interval=1)