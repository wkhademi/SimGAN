import utils
import argparse
import tensorflow as tf
import keras.backend as K
from model import SimGAN
from keras.optimizers import Adam, SGD
from keras.callback import ReduceLROnPlateau, ModelCheckpoint


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--synthetic_path', required=True)
parser.add_argument('--real_path', required=True)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--augment', type=bool, default=False)
parser.add_argument('--lambda_reg', type=float, default=10.0)
parser.add_argument('--max_steps', type=int, default=1000)
parser.add_argument('--K_g', type=int, default=2)
parser.add_argument('--K_d', type=int, default=1)

args = parser.parse_args()

expdir = 'experiments/%s'%args.name
os.makedirs(expdir,exist_ok=True)

synthetic_images = utils.load_data(args.synthetic_path)
real_images = utils.load_data(args.real_path)

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

callbacks = []
callbacks.append(ModelCheckpoint(filepath=weights_path, monitor='val_loss',save_best_only=1,verbose=1,save_weights_only=True))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))


def train():
    """
    Adversarial training for refiner network.

    Following procedure taken from Algorithm 1 in: https://arxiv.org/pdf/1612.07828.pdf
    """
    for step in range(args.max_steps):
        for _ in range(args.k_g):
            pass

        for _ in range(args.k_d):
            pass


if __name__ == '__main__':
    pass
