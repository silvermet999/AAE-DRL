"""_________________________________________________import libraries_________________________________________________"""
from tensorflow import Variable
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.nn import sigmoid, relu
from tensorflow.keras.losses import BinaryCrossentropy, Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.math import log
from keras import Input
import tensorflow as tf
import matplotlib.pyplot as plt
import main




"""______________________________________________________params______________________________________________________"""
# normal dist
def init_function(shape):
    return tf.random.normal(shape, stddev=1 / tf.sqrt(shape[0] / 2))


def weights(input_var, output_var):
    w = Variable(init_function([input_var, output_var]))
    return w

def biases(input_var):
    b = Variable(init_function([input_var]))
    return b


"""______________________________________________________archi______________________________________________________"""
class Encoder(Layer):
    def __init__(self, w1, b1, w2, b2):
        super(Encoder, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
    def call(self, x):
        hidden_layer = relu(tf.add(tf.matmul(x, self.w1), self.b1))
        enc_output = tf.add(tf.matmul(hidden_layer, self.w2), self.b2)
        return enc_output

class Decoder(Layer):
    def __init__(self, w1, b1, w2, b2):
        super(Decoder, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
    def call(self, x):
        hidden_layer = relu(tf.add(tf.matmul(x, self.w1), self.b1))
        dec_output = (tf.add(tf.matmul(hidden_layer, self.w2), self.b2))
        prob = sigmoid(dec_output)
        return prob, dec_output

class Discriminator(Layer):
    def __init__(self, w1, b1, w2, b2):
        super(Discriminator, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
    def call(self, x):
        hidden_layer = relu(tf.add(tf.matmul(x, self.w1), self.b1))
        final_output = (tf.add(tf.matmul(hidden_layer, self.w2), self.b2))
        disc_output = sigmoid(final_output)
        return disc_output



"""___________________________________________________loss and opt___________________________________________________"""


def opt(n_loss, sel_var):
    opt = Adam(learning_rate = lr)
    with tf.GradientTape() as tape:
        loss_value = n_loss()
    gradients = tape.gradient(loss_value, sel_var)
    opt.apply_gradients(zip(gradients, sel_var))
    return opt



"""_______________________________________________________vars_______________________________________________________"""
lr = 0.02
batch_size = 16
epochs = 10
input_dim = main.X_pca_rs.shape[0] * main.X_pca_rs.shape[1]
nn_dim = 128
z_dim = 10

z_input = Input([None, z_dim], dtype=tf.float32, name="input_noise")
x_input = Input([None, input_dim], dtype=tf.float32, name="real_noise")

disc_W1 = weights(z_dim, nn_dim)
disc_W2 = weights(nn_dim, 1)
disc_B1 = biases(nn_dim)
disc_B2 = biases(1)
enc_W1 = weights(input_dim, nn_dim)
enc_W2 = weights(nn_dim, z_dim)
dec_W1 = weights(z_dim, nn_dim)
dec_W2 = weights(nn_dim, input_dim)
enc_B1 = biases(nn_dim)
enc_B2 = biases(z_dim)
dec_B1 = biases(nn_dim)
dec_B2 = biases(input_dim)

z_output = Encoder(enc_W1, enc_B1, enc_W2, enc_B2)(x_input)
_, final_output = Decoder(dec_W1, dec_B1, dec_W2, dec_B2)(z_output)
real_output_disc = Discriminator(disc_W1, disc_B2, disc_W2, disc_B2)(z_input)
fake_output_disc = Discriminator(disc_W1, disc_B2, disc_W2, disc_B2)(z_output)

# ae_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# disc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# gen_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

enc_var = [enc_W1, enc_B1, enc_W2, enc_B2]
dec_var = [dec_W1, dec_B1, dec_W2, dec_B2]
disc_var = [disc_W1, disc_W2, disc_B1, disc_B2]
enc_dec_var = enc_var + dec_var



class disc_l(Loss):
    def __init__(self, name="custom_gan_loss"):
        super().__init__(name=name)
    def call(self, real, fake):
        real_loss = tf.math.log(tf.clip_by_value(real, 1e-10, 1.0))
        fake_loss = tf.math.log(tf.clip_by_value(1.0 - fake, 1e-10, 1.0))
        return -tf.reduce_mean(real_loss + fake_loss)

ae_loss = BinaryCrossentropy(from_logits=True)
# disc_loss = -tf.reduce_mean(log(tf.clip_by_value(real_output_disc, 1e-10, 1.0)) + log(tf.clip_by_value(1 - fake_output_disc, 1e-10, 1.0)))
disc_loss = disc_l()(real_output_disc, fake_output_disc)




# gen_loss = -tf.reduce_mean(log(tf.clip_by_value(fake_output_disc, 1e-10, 1.0)))
#
# ae_opt = opt(ae_loss, enc_dec_var)
# disc_opt = opt(disc_loss, disc_var)
# gen_opt = opt(gen_loss, enc_var)

