import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

FILE_NB = 9
E_KING = 1629
DIMENSIONS = 5 * FILE_NB * E_KING

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKA_hm', 0x5f134cb8, OrderedDict([('HalfKA_hm', DIMENSIONS)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features]
