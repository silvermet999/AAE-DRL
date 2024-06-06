"""_________________________________________________import libraries_________________________________________________"""
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

ae_loss = ae_loss
disc_loss = disc_loss
gen_loss = gen_loss

enc_var = [enc_W1, enc_B1, enc_W2, enc_B2]
dec_var = [dec_W1, dec_B1, dec_W2, dec_B2]
disc_var = [disc_W1, disc_W2, disc_B1, disc_B2]

ae_opt = opt(ae_loss, enc_var, dec_var)
disc_opt = opt(disc_loss, disc_var, 0)
gen_opt = opt(gen_loss, enc_var, 0)

