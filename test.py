import os
import utils
import argparse
import tensorflow as tf
import keras.backend as K
from tqdm import tqdm
from model import SimGAN
from PIL import Image, TiffImagePlugin


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--path', required=True)

args = parser.parse_args()

expdir = 'experiments/%s'%args.name
mean_std_path = '%s/meanstd.npz'%(expdir)

# load in training mean and standard deviation to standardize test images with
mean_std = np.load(mean_std_path)
train_mean = mean_std['train_mean']
train_std = mean_std['train_std']

# load in test set of simulated images
train_images, test_images = utils.load_data(args.path, include_val=True)

# standardize simulated data
test_images = (test_images - train_mean) / train_std

# build SimGAN model
input_shape = (test_images.shape[1], test_images.shape[2], 1)
sim_gan = SimGAN(input_shape=input_shape)

# load in the weights of the refiner and discriminator model
weights_path = '%s/weights.SimGAN.'%expdir
sim_gan.refiner.load_weights(weights_path+'refiner.latest.hdf5')
sim_gan.discriminator.load_weights(weights_path+'discriminator.latest.hdf5')

# define paths to save simulated and refined tif images
refined_path = '%s/refined.tif'%expdir
simulated_path = '%s/simulated.tif'%expdir

if os.path.exists(simulated_path):
    os.remove(simulated_path)
if os.path.exists(refined_path):
    os.remove(refined_path)


def test():
    for count, im in tqdm(enumerate(test_images)):
        out = sim_gan.refiner.predict(im)  # refine a single simulated image
        refined_image = np.squeeze((out + 1) / 2.) # shift output image from [-1, 1] to [0, 1]
        simulated_image = np.squeeze(im)

        simulated = Image.fromarray(simulated_image.astype(np.uint16))
        refined = Image.fromarray(refined_image.astype(np.uint16))

        with TiffImagePlugin.AppendingTiffWriter(simulated_path) as stf:
            simulated.save(stf)
            stf.newFrame()

        with TiffImagePlugin.AppendingTiffWriter(refined_path) as rtf:
            refined.save(rtf)
            rtf.newFrame()

if __name__ == '__main__':
    test()
