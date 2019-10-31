import utils
import argparse
import tensorflow as tf
import keras.backend as K
from model import SimGAN
from keras.optimizers import Adam, SGD
from keras.callback import ReduceLROnPlateau, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
parser.add_argument('--lr', type=float,, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)

args = parser.parse_args()

def train():
    pass

if __name__ == '__main__':
    pass
