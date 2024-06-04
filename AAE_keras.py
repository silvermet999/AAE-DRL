"""_________________________________________________import libraries_________________________________________________"""
import numpy as np
from tensorflow import Variable
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.nn import sigmoid, relu, sigmoid_cross_entropy_with_logits
from tensorflow.keras.optimizers import Adam
from keras import Input
import tensorflow as tf
import matplotlib.pyplot as plt
import main




"""______________________________________________________params______________________________________________________"""
# normal dist
def init_function(shape):
    return tf.random.normal(shape, stddev=1 / tf.sqrt(shape[0] / 2))


class params:
    def __init__(self, input_dim, nn_dim, z_dim):
        self.input_dim = input_dim
        self.nn_dim = nn_dim
        self.z_dim = z_dim

        self.disc_W = self.init_disc_W()
        self.disc_B = self.init_disc_B()
        self.ae_W = self.init_ae_W()
        self.ae_B = self.init_ae_B()

    def init_disc_W(self):
        return {
            "disc_1": Variable(init_function([self.z_dim, self.nn_dim])),
            "disc_2": Variable(init_function([self.nn_dim, 1]))
        }

    def init_disc_B(self):
        return {
            "disc_1": Variable(init_function([self.nn_dim])),
            "disc_2": Variable(init_function([1]))
        }

    def init_ae_W(self):
        return {
            "enc_W_1" : Variable(init_function([self.input_dim, self.nn_dim])),
            "enc_W_2" : Variable(init_function([self.nn_dim, self.z_dim])),
            "dec_W_1" : Variable(init_function([self.z_dim, self.nn_dim])),
            "dec_W_2" : Variable(init_function([self.nn_dim, self.input_dim])),
        }

    def init_ae_B(self):
        return {
            "enc_B_1" : Variable(init_function([self.nn_dim])),
            "enc_B_2" : Variable(init_function([self.z_dim])),
            "dec_B_1" : Variable(init_function([self.nn_dim])),
            "dec_B_2" : Variable(init_function([self.input_dim])),
        }

    def get_disc_W(self):
        return self.disc_W

    def get_disc_B(self):
        return self.disc_B

    def get_ae_W(self):
        return self.ae_W

    def get_ae_B(self):
        return self.ae_B



"""______________________________________________________archi______________________________________________________"""
class Encoder(Layer):
    def __init__(self, ae_W, ae_B, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.ae_W = ae_W
        self.ae_B = ae_B

    def call(self, x):
        hidden_layer = relu(tf.add(tf.matmul(x, self.ae_W), self.ae_B))
        enc_output = tf.add(tf.matmul(hidden_layer, self.ae_W), self.ae_B)
        return enc_output

    # def compute_output_spec(self, input_spec):
    #     input_shape = input_spec.as_list()
    #     hidden_dim = self.ae_W.shape[0]
    #     return tf.TensorSpec(shape=[input_shape[0], hidden_dim], dtype=tf.float32)

def Decoder(x, ae_W, ae_B):
    hidden_layer = relu(tf.add(tf.matmul(x, ae_W["dec_W_1"]), ae_B["dec_B_1"]))
    dec_output = tf.add(tf.matmul(hidden_layer, ae_W["dec_W_2"]), ae_B["dec_B_2"])
    prob = sigmoid(dec_output)
    return prob, dec_output

def Discriminator(x, disc_W, disc_B):
    hidden_layer = relu(tf.add(tf.matmul(x, disc_W["disc_1"]), disc_B["disc_1"]))
    final_output = tf.add(tf.matmul(hidden_layer, disc_W["disc_2"]), disc_B["disc_2"])
    disc_output = sigmoid(final_output)
    return disc_output



"""___________________________________________________loss and opt___________________________________________________"""
def ae_loss():
    ae_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(logits=final_output, labels=x_input))
    return ae_loss

def disc_loss():
    disc_loss = -tf.reduce_mean(tf.log(real_output_disc) + tf.log(1 - fake_output_disc))
    return disc_loss

def gen_loss():
    gen_loss = -tf.reduce_mean(tf.log(fake_output_disc))
    return gen_loss

def opt(loss, var1, var2):
    opt = Adam(learning_rate = lr).minimize(loss, var_list = var1 + var2)
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
initializer = params(input_dim, nn_dim, z_dim)
disc_W = initializer.get_disc_W()
disc_B = initializer.get_disc_B()
ae_W = initializer.get_ae_W()
ae_B = initializer.get_ae_B()
init_enc = Encoder(ae_W["enc_W_1"], ae_B["enc_B_1"])
z_output = Model(inputs = x_input, outputs = init_enc(x_input))
_, final_output = Decoder(z_output, ae_W["enc_W_1"], ae_B["enc_B_1"])
real_output_disc = Discriminator(z_input, disc_W, disc_B)
fake_output_disc = Discriminator(z_output, disc_W, disc_B)
ae_loss = ae_loss
disc_loss = disc_loss
gen_loss = gen_loss
enc_var = [ae_W["enc_W_1"], ae_B["enc_B_1"], ae_W["enc_W_2"], ae_B["enc_B_2"]]
dec_var = [ae_W["dec_W_1"], ae_B["dec_B_1"], ae_W["dec_W_2"], ae_B["dec_B_2"]]
disc_var = [disc_W["disc_1"], disc_W["disc_2"], disc_B["disc_1"], disc_B["disc_2"]]
ae_opt = opt(ae_loss, enc_var, dec_var)
disc_opt = opt(disc_loss, disc_var, 0)
gen_opt = opt(gen_loss, enc_var, 0)

