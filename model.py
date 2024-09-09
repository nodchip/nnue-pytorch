import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import math
import collections

# 3 layer fully connected network
L1 = 512
L2 = 8
L3 = 96

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(
      self, feature_set, lambda_=[1.0], lr=[1.0],
      label_smoothing_eps=0.0, num_batches_warmup=10000, newbob_decay=0.5,
      num_epochs_to_adjust_lr=500, score_scaling=361, min_newbob_scale=1e-5,
      momentum=0.0):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_
    self.lr = lr
    self.label_smoothing_eps = label_smoothing_eps
    self.num_batches_warmup = num_batches_warmup
    self.newbob_scale = 1.0
    self.newbob_decay = newbob_decay
    self.best_loss = 1e10
    self.num_epochs_to_adjust_lr = num_epochs_to_adjust_lr
    self.latest_loss_sum = 0.0
    self.latest_loss_count = 0
    self.score_scaling = score_scaling
    # Warmupを開始するステップ数
    self.warmup_start_global_step = 0
    self.min_newbob_scale = min_newbob_scale
    self.parameter_index = 0
    self.momentum = momentum

    self._zero_virtual_feature_weights()

    self._initialize_input_transformer_weights()
  
  def _initialize_input_transformer_weights(self):
    # 手駒
    BONA_PIECE_ZERO = 0
    f_hand_pawn = BONA_PIECE_ZERO + 1
    e_hand_pawn = 20
    f_hand_lance = 39
    e_hand_lance = 44
    f_hand_knight = 49
    e_hand_knight = 54
    f_hand_silver = 59
    e_hand_silver = 64
    f_hand_gold = 69
    e_hand_gold = 74
    f_hand_bishop = 79
    e_hand_bishop = 82
    f_hand_rook = 85
    e_hand_rook = 88
    fe_hand_end = 90

    # 盤上の駒
    f_pawn = fe_hand_end
    e_pawn = f_pawn + 81
    f_lance = e_pawn + 81
    e_lance = f_lance + 81
    f_knight = e_lance + 81
    e_knight = f_knight + 81
    f_silver = e_knight + 81
    e_silver = f_silver + 81
    f_gold = e_silver + 81
    e_gold = f_gold + 81
    f_bishop = e_gold + 81
    e_bishop = f_bishop + 81
    f_horse = e_bishop + 81
    e_horse = f_horse + 81
    f_rook = e_horse + 81
    e_rook = f_rook + 81
    f_dragon = e_rook + 81
    e_dragon = f_dragon + 81
    fe_old_end = e_dragon + 81

    fe_new_end = fe_old_end

    fe_end = fe_new_end

    # file: 筋
    # rank: 段

    def is_file_inside(file: int):
      return 0 <= file < 9
    
    def is_rank_inside(rank: int):
      return 0 <= rank < 9

    def to_square(file: int, rank: int):
      return file * 9 + rank

    def halfkp_idx(king_file: int, king_rank: int, piece_file: int, piece_rank: int, piece: int):
      assert(is_file_inside(king_file))
      assert(is_rank_inside(king_rank))
      assert(is_file_inside(piece_file))
      assert(is_rank_inside(piece_rank))
      king_square = to_square(king_file, king_rank)
      piece_square = to_square(piece_file, piece_rank)
      return piece_square + piece + king_square * fe_end

    pawn_deltas = [[0, -1]]

    lance_deltas = list()
    for delta_rank in range(-8, 0):
      lance_deltas.append([0, delta_rank])
    
    knight_deltas = [[-1, -2], [1, -2]]

    silver_deltas = [
      [-1, -1], [0, -1], [1, -1],
      [-1, 0], [1, 0],
      [-1, 1], [1, 1],
    ]

    gold_deltas = [
      [-1, -1], [0, -1], [1, -1],
      [-1, 0], [1, 0],
      [0, 1],
    ]

    bishop_deltas = list()
    for delta in range(-8, 8 + 1):
      if delta == 0:
        continue;
      bishop_deltas.append([delta, delta])
      bishop_deltas.append([delta, -delta])

    horse_deltas = list(bishop_deltas)
    horse_deltas.extend([
      [0, -1], [-1, 0], [1, 0], [0, 1],
    ])

    rook_deltas = list()
    for delta in range(-8, 8 + 1):
      if delta == 0:
        continue
      rook_deltas.append([delta, 0])
      rook_deltas.append([0, delta])

    dragon_deltas = list(rook_deltas)
    dragon_deltas.extend([
      [-1, -1], [-1, 1], [1, -1], [1, 1],
    ])

    RelativePositions = collections.namedtuple('RelativePositions', ['piece', 'relative_positions'])
    relative_positions_list = [
      RelativePositions(e_pawn, pawn_deltas),
      RelativePositions(e_lance, lance_deltas),
      RelativePositions(e_knight, knight_deltas),
      RelativePositions(e_silver, silver_deltas),
      RelativePositions(e_gold, gold_deltas),
      RelativePositions(e_bishop, bishop_deltas),
      RelativePositions(e_horse, horse_deltas),
      RelativePositions(e_rook, rook_deltas),
      RelativePositions(e_dragon, dragon_deltas),
    ]
    
    with torch.no_grad():
      for king_file in range(9):
        for king_rank in range(9):
          for relative_positions in relative_positions_list:
            for relative_position in relative_positions.relative_positions:
              piece_file = king_file + relative_position[0]
              if not is_file_inside(piece_file):
                continue

              piece_rank = king_rank + relative_position[1]
              if not is_rank_inside(piece_rank):
                continue

              index = halfkp_idx(king_file, king_rank, piece_file, piece_rank, relative_positions.piece)
              self.input.weight[0][index] = 1.0

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in):
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def step_(self, batch, batch_idx, loss_type):
    us, them, white, black, outcome, score = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    nnue2score = 600
    scaling = self.score_scaling

    q = self(us, them, white, black) * nnue2score / scaling
    t = outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
    p = (score / scaling).sigmoid()

    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    lambda_ = self.lambda_[self.parameter_index]
    result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    self.log(loss_type, loss)
    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'val_loss')
  
  def validation_epoch_end(self, outputs):
    self.latest_loss_sum += float(sum(outputs)) / len(outputs);
    self.latest_loss_count += 1

    if self.newbob_decay != 1.0 and self.current_epoch > 0 and self.current_epoch % self.num_epochs_to_adjust_lr == 0:
      latest_loss = self.latest_loss_sum / self.latest_loss_count
      self.latest_loss_sum = 0.0
      self.latest_loss_count = 0
      if latest_loss < self.best_loss:
        self.print(f"{self.current_epoch=}, {latest_loss=} < {self.best_loss=}, accepted, {self.newbob_scale=}")
        sys.stdout.flush()
        self.best_loss = latest_loss
      else:
        self.newbob_scale *= self.newbob_decay
        self.print(f"{self.current_epoch=}, {latest_loss=} >= {self.best_loss=}, rejected, {self.newbob_scale=}")
        sys.stdout.flush()
    
    if self.newbob_scale < self.min_newbob_scale:
      self.parameter_index += 1
      if self.parameter_index < len(self.lr):
        self.best_loss = 1e10
        self.newbob_scale = 1.0
      else:
        self.trainer.should_stop = True
        self.print(f"{self.current_epoch=}, early stopping")

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  # learning rate warm-up
  def optimizer_step(
      self,
      epoch,
      batch_idx,
      optimizer,
      optimizer_idx,
      optimizer_closure,
      on_tpu,
      using_native_amp,
      using_lbfgs,
  ):
    # manually warm up lr without a scheduler
    if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
      warmup_scale = min(1.0, float(self.trainer.global_step - self.warmup_start_global_step + 1) / self.num_batches_warmup)
    else:
      warmup_scale = 1.0
    for pg in optimizer.param_groups:
      pg["lr"] = self.lr[self.parameter_index] * warmup_scale * self.newbob_scale
      self.log("lr", pg["lr"])

    # update params
    optimizer.step(closure=optimizer_closure)

    # clip parameters
    for child in self.children():
      if not isinstance(child, nn.Linear):
        continue

      if child == self.input:
        continue

      # FC layers are stored as int8 weights, and int32 biases
      kWeightScaleBits = 6
      kActivationScale = 127.0
      if child != self.output:
        kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
      else:
        kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
      kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
      kMaxWeight = 127.0 / kWeightScale # roughly 2.0
      child.weight.data.clamp_(-kMaxWeight, kMaxWeight)

  def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr[0], momentum=self.momentum)

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
