import utils
import argparse
import tensorflow as tf
import keras.backend as K
from model import SimGAN
from ImageBuffer import ImageBuffer
from keras.optimizers import Adam, SGD
from keras.callback import ReduceLROnPlateau, ModelCheckpoint


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
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
synthetic_images, val_images = utils.load_data(args.synthetic_path, include_val=True)
real_images, _ = utils.load_data(args.real_path)

if args.augment:  # generate more training data through augmentation
    synthetic_images = utils.augment_images(synthetic_images)
    val_images = utils.augment_images(val_images)
    real_images = utils.augment_images(real_images)

synth_crops = utils.random_crop_generator(synthetic_images, args.crop_size, args.batch_size)
val_crops = utils.random_crop_generator(val_images, args.crop_size, args.batch_size)
real_crops = utils.random_crop_generator(real_images, args.crop_size, args.batch_size)

# select optimizer to use during training
if args.optimizer.lower() == 'sgd':
    optimizer = SGD(args.lr, momentum=0.9)
elif args.optimizer.lower() == 'adam':
    optimizer = Adam(args.lr, amsgrad=True)
else
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
labels_shape = tuple([args.batch_size] + sim_gan.discriminator.output_shape[1:])
synth_labels = np.zeros(labels_shape, dtype=np.float32)
synth_labels[:,:,:,1] = 1.0
real_labels = np.zeros(lables_shape, dtype=np.float32)
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

            # save refiner model weights every 25 steps
            if i % 25 == 0:
                sim_gan.refiner.save_weights(weights_path+'refiner.latest.hdf5', save_format='h5')
    else:
        sim_gan.refiner.load_weights(args.refiner_model_path)

    if not args.discriminator_model_path:
        # train discriminator for 200 steps (100 steps on real and 100 steps on simulated)
        for i in range(args.discrim_steps//2):
            # train discriminator on batch of real images
            real_inputs, _ = next(real_crops)
            discrim_real_loss = sim_gan.discriminator.train_on_batch(real_inputs, y=real_labels)

            # train discriminator on batch of refined images
            refiner_inputs, _ = next(synth_crops)

            # only mix in old refined images when buffer has filled up
            if len(image_buffer) == args.buffersize:
                refined_inputs = None
            else:
                refined_inputs = sim_gan.refiner.predict_on_batch(refiner_inputs)

            discrim_refined_loss = sim_gan.discriminator.train_on_batch(refined_inputs, y=synth_labels)

            # update image buffer with the newly refined images
            image_buffer.update(refined_inputs)

            # save discriminator model weights every 25 steps
            if i % 25 == 0:
                sim_gan.discriminator.save_weights(weights_path+'discriminator.latest.hdf5', save_format='h5')
    else:
        sim_gan.discriminator.load_weights(args.discriminator_model_path)

    # training procedure defined in Algorithm 1
    for step in range(args.max_steps):
        for _ in range(args.k_g):
            pass

        for _ in range(args.k_d):
            pass


if __name__ == '__main__':
    train()
