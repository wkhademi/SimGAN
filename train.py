import os
import utils
import random
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from model import SimGAN
from ImageBuffer import ImageBuffer
from keras.optimizers import Adam, SGD


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--synthetic_path', required=True)
parser.add_argument('--real_path', required=True)
parser.add_argument('--refiner_model_path', type=str, default=None)
parser.add_argument('--discriminator_model_path', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--lambda_reg', type=float, default=10.0)
parser.add_argument('--refine_steps', type=int, default=500)
parser.add_argument('--discrim_steps', type=int, default=200)
parser.add_argument('--max_steps', type=int, default=1000)
parser.add_argument('--K_g', type=int, default=2)
parser.add_argument('--K_d', type=int, default=1)
parser.add_argument('--buffersize', type=int, default=512)

args = parser.parse_args()

expdir = 'experiments/%s'%args.name
os.makedirs(expdir,exist_ok=True)

# load data in from tif files
synthetic_images, test_images = utils.load_data(args.synthetic_path, include_val=True)
real_images, _ = utils.load_data(args.real_path)

train_mean = np.mean(synthetic_images)
train_std = np.std(synthetic_images)
mean_std_path = '%s/meanstd.npz'%(expdir)
np.savez(mean_std_path,train_mean=train_mean,train_std=train_std)

# standardize simulated data for input to refiner
synthetic_images = (synthetic_images - train_mean) / train_std

# normalize real images to be between -1 and 1 for discriminator
MAXVAL = 1.
MINVAL = 0.
real_images = 2*((real_images - MINVAL) / (MAXVAL - MINVAL)) - 1

if args.augment:  # generate more training data through augmentation
    synthetic_images = utils.augment_images(synthetic_images)
    real_images = utils.augment_images(real_images)

synth_crops = utils.random_crop_generator(synthetic_images, args.crop_size, args.batch_size)
real_crops = utils.random_crop_generator(real_images, args.crop_size, args.batch_size)

# select optimizer to use during training
if args.optimizer.lower() == 'sgd':
    optimizer = SGD(args.lr, momentum=0.9)
elif args.optimizer.lower() == 'adam':
    optimizer = Adam(args.lr, amsgrad=True)
else:
    raise ValueError('Not a valid optimizer. Choose between SGD or Adam.')

# build SimGAN model
sim_gan = SimGAN(input_shape=(args.crop_size, args.crop_size, 1),
                 optimizer=optimizer,
                 lambda_reg=args.lambda_reg)

# create image buffer for storing a history of refined images
image_buffer = ImageBuffer(args.buffersize)

# path name to save trained model weights to
weights_path = '%s/weights.SimGAN.'%expdir

# create ground truth labels for discriminator
labels_shape = tuple([args.batch_size] + list(sim_gan.discriminator.output_shape[1:]))
synth_labels = np.zeros(labels_shape, dtype=np.float32)
synth_labels[:,:,:,1] = 1.0
real_labels = np.zeros(labels_shape, dtype=np.float32)
real_labels[:,:,:,0] = 1.0


def train():
    """
    Adversarial training for refiner network.

    Following procedure taken from Algorithm 1 in: https://arxiv.org/pdf/1612.07828.pdf
    """
    if not args.refiner_model_path:
        # train refiner for 500 steps
        for i in range(args.refine_steps):
            refiner_inputs, _ = next(synth_crops)
            refiner_loss = sim_gan.refiner.train_on_batch(refiner_inputs, y=refiner_inputs)

            print('Step {} - Refiner loss: {}'.format(i, refiner_loss))

            # save refiner model weights every 25 steps
            if i % 25 == 0:
                print('Saving weights of refiner model...')
                sim_gan.refiner.save_weights(weights_path+'refiner.latest.hdf5')
    else:
        print('Loading in weights for refiner model...')
        sim_gan.refiner.load_weights(args.refiner_model_path)

    if not args.discriminator_model_path:
        # train discriminator for 200 steps (100 steps on real and 100 steps on simulated)
        for i in range(args.discrim_steps//2):
            # train discriminator on batch of real images
            real_inputs, _ = next(real_crops)
            discrim_real_loss = sim_gan.discriminator.train_on_batch(real_inputs, y=real_labels)

            print('Step {} - Discriminator loss w/ real images: {}'.format(i, discrim_real_loss))

            # train discriminator on batch of refined images
            refiner_inputs, _ = next(synth_crops)
            refined_inputs = sim_gan.refiner.predict_on_batch(refiner_inputs)
            discrim_refined_loss = sim_gan.discriminator.train_on_batch(refined_inputs, y=synth_labels)

            print('Step {} - Discriminator loss w/ refined images: {}'.format(i, discrim_refined_loss))

            # save discriminator model weights every 25 steps
            if i % 25 == 0:
                print('Saving weights of discriminator model...')
                sim_gan.discriminator.save_weights(weights_path+'discriminator.latest.hdf5')
    else:
        print('Loading in weights for discriminator model...')
        sim_gan.discriminator.load_weights(args.discriminator_model_path)

    # training procedure defined in Algorithm 1
    for step in range(args.max_steps):
        for i in range(args.K_g):
            refiner_inputs, _ = next(synth_crops)
            adversarial_loss = sim_gan.adversarial.train_on_batch(refiner_inputs, y=[refiner_inputs, real_labels])

            print('Step {} - Adversarial loss: {}'.format(args.K_g*step+i, adversarial_loss))

        for i in range(args.K_d):
            # train discriminator on batch of real images
            real_inputs, _ = next(real_crops)
            discrim_real_loss = sim_gan.discriminator.train_on_batch(real_inputs, y=real_labels)

            print('Step {} - Discriminator loss w/ real images: {}'.format(args.K_d*step+i, discrim_real_loss))

            # train discriminator on batch of refined images
            refiner_inputs, _ = next(synth_crops)
            refined_inputs = sim_gan.refiner.predict_on_batch(refiner_inputs)

            # only mix in old refined images when buffer has filled up
            if len(image_buffer) == args.buffersize:
                old_refined_inputs = image_buffer.fetch(args.batch_size//2)
                indices = random.sample(range(0, args.batch_size), args.batch_size//2)
                refined_inputs = np.concatenate([old_refined_inputs, refined_inputs[indices]])

            discrim_refined_loss = sim_gan.discriminator.train_on_batch(refined_inputs, y=synth_labels)

            print('Step {} - Discriminator loss w/ refined images: {}'.format(i, discrim_refined_loss))

            # update image buffer with the newly refined images
            image_buffer.update(refined_inputs)

        # save discriminator and refiner weights every 25 steps
        if step % 25 == 0:
            print('Saving weights of refiner and discriminator model...')
            sim_gan.refiner.save_weights(weights_path+'refiner.latest.hdf5')
            sim_gan.discriminator.save_weights(weights_path+'discriminator.latest.hdf5')


if __name__ == '__main__':
    train()
