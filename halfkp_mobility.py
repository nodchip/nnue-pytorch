import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 81
NUM_PLANES = 1548
# 動けない場合=0マス動ける場合も含めてカウントする。
MAX_MOBILITY_DRAGON = 20 + 1
MAX_MOBILITY_HORSE = 20 + 1
MAX_MOBILITY_ROOK = 16 + 1
MAX_MOBILITY_BISHOP = 16 + 1
MAX_MOBILITY_LANCE = 8 + 1

class Features(FeatureBlock):
  '''竜馬飛角香が動けるマスの数

  以下の順に並んでいる。  
  - 味方の竜が動けるマスの数
  - 相手の竜が動けるマスの数
  - 味方の馬が動けるマスの数
  - 相手の馬が動けるマスの数
  - 味方の飛車が動けるマスの数
  - 相手の飛車が動けるマスの数
  - 味方の角が動けるマスの数
  - 相手の角が動けるマスの数
  - 味方の香が動けるマスの数
  - 相手の香が動けるマスの数

  動けるマスの数は、駒が置かれているマス×動けるマスの数で表現する。
  '''
  def __init__(self):
    self.num_features = NUM_SQ * NUM_PLANES + NUM_SQ * (MAX_MOBILITY_DRAGON + MAX_MOBILITY_HORSE + MAX_MOBILITY_ROOK + MAX_MOBILITY_BISHOP + MAX_MOBILITY_LANCE) * 2
    # Hack: 複数のFeatureからなるFeatureSetを実装していない箇所がある。そのためFeatureは常に1つとし、ハッシュの計算を自前で行う。
    super(Features, self).__init__('HalfKPMobility', 0x5d69d5b8 ^ (0x6d8d203d << 1) ^ (0x6d8d203d >> 31) & 0xffffffff , OrderedDict([(
      'HalfKPMobility',
      self.num_features
      )]))

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      # ダミー実装
      indices = torch.zeros(self.num_features)
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))


'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features]
