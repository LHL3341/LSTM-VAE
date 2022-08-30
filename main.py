from ast import arg
from distutils.command.config import config
from email import parser
import os
import argparse
import torch

from anomaly_detector import Detector

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--pretrained_epoch', type=int, default=0)
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--latent_size', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('--seq_lenth', type=int, default=1)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--dataname', type=str, default='Yahoo_A4')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dropout', type=int, default=0.2)
parser.add_argument('--save_fig', type=bool, default=True)

config = parser.parse_args()
args = vars(config)

if args['device'] == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark=True

detector = Detector(args)