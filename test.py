import utils
import argparse
from model import SimGAN

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
parser.add_argument('--lr', type=float,, default=1e-3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--crop_size', type=int, default=256)

args = parser.parse_args()

def test():
    pass

if __name__ == '__main__':
    pass
