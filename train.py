from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation, UpSampling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import datetime
from time import localtime, strftime
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf

from dataloader import DataLoader 

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def time():
    return strftime('%m%d_%H%M', localtime())

class DCGAN():
    def __init__(self, dataset_name = "celeba", img_size = 28, channels = 3, backup_dir = "backup"):
        
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.channels = channels
        
        self.backup_dir = backup_dir
        self.time = time()

        # Input shape
        self.latent_dim = 100
        self.learning_rate = 1e-3

        self.gf = 64 # filter size of generator's last layer
        self.df = 64 # filter size of discriminator's first layer

        optimizer_disc = Adam(self.learning_rate / 10, beta_1=0.5, decay=0.00005)
        optimizer_gen = Adam(self.learning_rate, beta_1=0.5, decay=0.00005)

        # Configure data loader
        self.dl = DataLoader(dataset_name = self.dataset_name, img_res=(self.img_size,self.img_size), mem_load=True)
        self.n_data = self.dl.get_n_data()

        # Build generator
        self.generator = self.build_generator()
        print("---------------------generator summary----------------------------")
        self.generator.summary()


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        print("\n---------------------discriminator summary----------------------------")
        self.discriminator.summary()

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_disc,
                                   metrics=['accuracy'])

        z = Input(shape=(self.latent_dim,))
        fake_img = self.generator(z)

        # for the combined model, we only train ganerator
        self.discriminator.trainable = False

        validity = self.discriminator(fake_img)

        # Build combined model
        self.combined = Model(z, validity)
        print("\n---------------------combined summary----------------------------")
        self.combined.summary()
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer_gen)

    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))

        def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(2, 2), bn_relu=True):
            """Layers used during upsampling"""
            u = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(layer_input)
            if bn_relu:
                u = BatchNormalization()(u)
                u = Activation("relu")(u)
                
            return u

        generator = Dense(4 * self.gf * self.img_size // 4 * self.img_size // 4, use_bias=False)(noise)
        generator = BatchNormalization()(generator)
        generator = Activation('relu')(generator)
        generator = Reshape((self.img_size // 4, self.img_size // 4, self.gf * 4))(generator)
        generator = deconv2d(generator, filters=self.gf * 2)
        generator = deconv2d(generator, filters=self.gf    )
        generator = deconv2d(generator, filters=self.channels, kernel_size=(3,3), strides=(1,1), bn_relu=False)

        generator = Activation('tanh')(generator)

        return Model(noise, generator)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)(layer_input)
            if bn:
                d = Dropout(rate=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            

            return d

        # Input img = generated image
        d0 = Input(shape=(self.img_size, self.img_size, self.channels))

        d = d_block(d0, self.df, strides=2, bn=False)
        d = d_block(d, self.df*2, strides=2)

        d = Flatten()(d)
        validity = Dense(1, activation='sigmoid')(d)

        return Model(d0, validity)

    def train(self, epochs, batch_size, sample_interval):
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        max_iter = int(self.n_data/batch_size)
        os.makedirs(f"{self.backup_dir}/logs/{self.time}", exist_ok=True)
        tensorboard = TensorBoard(f"{self.backup_dir}/logs/{self.time}")
        tensorboard.set_model(self.generator)

        os.makedirs(f"{self.backup_dir}/models/{self.time}/", exist_ok=True)
        with open(f"{self.backup_dir}/models/{self.time}/generator_architecture.json", "w") as f:
            f.write(self.generator.to_json())
        print(f"\nbatch size : {batch_size} | num_data : {self.n_data} | max iteration : {max_iter} | time : {self.time} \n")
        for epoch in range(1, epochs+1):
            for iter in range(max_iter):
                # ------------------
                #  Train Generator
                # ------------------
                ref_imgs = self.dl.load_data(batch_size)

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                make_trainable(self.discriminator, True)
                d_loss_real = self.discriminator.train_on_batch(ref_imgs, valid * 0.9)  # label smoothing *0.9
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                make_trainable(self.discriminator, False)

                logs = self.combined.train_on_batch([noise], [valid])
                tensorboard.on_epoch_end(iter, named_logs(self.combined, [logs]))

                if iter % (sample_interval // 10) == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    print(f"epoch:{epoch} | iter : {iter} / {max_iter} | time : {elapsed_time} | g_loss : {logs} | d_loss : {d_loss} ")

                if (iter+1) % sample_interval == 0:
                    self.sample_images(epoch, iter+1)

            # save weights after every epoch
            self.generator.save_weights(f"{self.backup_dir}/models/{self.time}/generator_epoch{epoch}_weights.h5")

    def sample_images(self, epoch, iter):
        os.makedirs(f'{self.backup_dir}/samples/{self.time}', exist_ok=True)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        gen_img = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_img = 0.5 * gen_img + 0.5

        # Save generated images and the high resolution originals
        fig, axs = plt.subplots(r, c)
        for row in range(r):
            for col in range(c):
                axs[row, col].imshow(gen_img[5*row+col,:,:,:])
                axs[row, col].axis('off')
        fig.savefig(f"{self.backup_dir}/samples/{self.time}/e{epoch}-i{iter}.png", bbox_inches='tight', dpi=100)
        # plt.show() # only when running in ipython, otherwise halts the execution
        plt.close()

if __name__ == "__main__":
    gan = DCGAN(dataset_name = "celeba", img_size = 28, channels = 3, backup_dir = "backup")
    DEBUG = 0

    if DEBUG == 1:
        gan.n_data = 50
        gan.train(epochs=2, batch_size=1, sample_interval=10)
    else:
        gan.train(epochs=20, batch_size=64, sample_interval=100)