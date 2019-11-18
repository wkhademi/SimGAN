# SimGAN
Implementation of SimGAN for making simulated microscopy data more realistic.

Original paper can be found here: [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828.pdf)

## Train
`python -m train yfp3 --synthetic_path [path to simulated dataset] --real_path [path to real dataset]`

## Test
`python -m test yfp3 --path [path to simulated dataset]`
