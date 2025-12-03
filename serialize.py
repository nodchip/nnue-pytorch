import argparse
import features
import math
import model as M
import numpy
import nnue_bin_dataset
import struct
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import reduce
import operator
import os
import matplotlib.pyplot as plt
import datetime

def ascii_hist(name, x, bins=6):
  N,X = numpy.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

# hardcoded for now
VERSION = 0x7AF32F16

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model, output_directory_path):
    self.output_directory_path = output_directory_path
    os.makedirs(self.output_directory_path, exist_ok=True)
    self.figure_index = 0
    self.buf = bytearray()

    fc_hash = self.fc_hash(model)
    self.write_header(model, fc_hash)
    self.int32(model.feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.write_feature_transformer(model)
    self.int32(fc_hash) # FC layers hash
    self.write_fc_layer(model.l1)
    self.write_fc_layer(model.l2)
    self.write_fc_layer(model.output, is_output=True)

  @staticmethod
  def fc_hash(model):
    # InputSlice hash
    prev_hash = 0xEC42E90D
    prev_hash ^= (M.L1 * 2)

    # Fully connected layers
    layers = [model.l1, model.l2, model.output]
    for layer in layers:
      layer_hash = 0xCC03DAE4
      layer_hash += layer.out_features
      layer_hash ^= prev_hash >> 1
      layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      if layer.out_features != 1:
        # Clipped ReLU hash
        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
      prev_hash = layer_hash
    return layer_hash

  def write_header(self, model, fc_hash):
    self.int32(VERSION) # version
    self.int32(fc_hash ^ model.feature_set.hash ^ (M.L1*2)) # halfkp network hash
    description = b"Features=HalfKP(Friend)[125388->256x2],"
    description += b"Network=AffineTransform[1<-256](ClippedReLU[256](AffineTransform[256<-256]"
    description += b"(ClippedReLU[256](AffineTransform[256<-512](InputSlice[512(0:512)])))))"
    self.int32(len(description)) # Network definition
    self.buf.extend(description)

  def coalesce_ft_weights(self, model, layer):
    weight = layer.weight.data
    indices = model.feature_set.get_virtual_to_real_features_gather_indices()
    weight_coalesced = weight.new_zeros((weight.shape[0], model.feature_set.num_real_features))
    for i_real, is_virtual in enumerate(indices):
      weight_coalesced[:, i_real] = sum(weight[:, i_virtual] for i_virtual in is_virtual)

    return weight_coalesced
  
  def save_histogram(self, file_name, data, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    bins = min(256, data.numel())
    frequency, value = data.to(torch.float).histogram(bins=bins)
    value += (value[1] - value[0]) * 0.5
    width = value[1] - value[0]
    value = value[:-1]
    ax.bar(value, frequency, width=width)
    ax.set_title(title)
    fig.savefig(os.path.join(self.output_directory_path, file_name))
    print(f'Saved a histogram to {file_name}')

    mean = data.to(torch.float).mean().item()
    std = data.to(torch.float).std().item()
    print(f'{mean=} {std=}')

  def stochastic_round(self, x: torch.Tensor):
    """
    Stochastic rounding:
      floor(x) with probability (ceil(x) - x)
      ceil(x)  with probability (x - floor(x))

    Returns integer tensor.
    """
    floor_x = torch.floor(x)
    ceil_x = floor_x + 1
    prob_up = x - floor_x  # 小数部分 → 切り上げ確率

    # 0〜1 の一様乱数
    rand = torch.rand_like(x)

    # rand < prob_up → ceil にする
    return torch.where(rand < prob_up, ceil_x, floor_x)


  def write_feature_transformer(self, model):
    # --- bias の処理 ---
    layer = model.input
    bias = layer.bias.data

    # scale
    bias_scaled = bias * 127.0   # float

    # stochastic rounding → int16 に変換
    bias_quant = self.stochastic_round(bias_scaled).to(torch.int16)

    ascii_hist('ft bias:', bias_quant.numpy())
    self.save_histogram(f'{self.figure_index:02}_feature_transformer_bias.png',
                        bias_quant, 'bias', 'frequency', 'feature transformer bias')
    self.figure_index += 1
    self.buf.extend(bias_quant.flatten().numpy().tobytes())

    print(datetime.datetime.now())

    # --- weight の処理 ---
    weight = self.coalesce_ft_weights(model, layer)

    weight_scaled = weight * 127.0  # float

    # stochastic rounding
    weight_quant = self.stochastic_round(weight_scaled).to(torch.int16)

    ascii_hist('ft weight:', weight_quant.numpy())
    self.save_histogram(f'{self.figure_index:02}_feature_transformer_weight.png',
                        weight_quant, 'weight', 'frequency', 'feature transformer weight')
    self.figure_index += 1

    # NNUE 形式（転置して保存）
    self.buf.extend(weight_quant.transpose(0, 1).flatten().numpy().tobytes())
    print(datetime.datetime.now())
    print()

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0

    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale  # 8128
    else:
      kBiasScale = 9600.0  # output 層のみ Ponanza constant 版

    kWeightScale = kBiasScale / kActivationScale  # = 64.0 (通常)
    kMaxWeight = 127.0 / kWeightScale             # ≈ 2.0

    # ==== Bias: fp32 → int32 (stochastic rounding) ====
    bias = layer.bias.data
    bias_scaled = bias * kBiasScale

    # Stochastic rounding
    bias_quant = self.stochastic_round(bias_scaled).to(torch.int32)

    ascii_hist('fc bias:', bias_quant.numpy())
    self.save_histogram(
        f'{self.figure_index:02}_fully_connected_layer_bias.png',
        bias_quant, 'bias', 'frequency', 'fully connected layer bias'
    )
    self.figure_index += 1
    self.buf.extend(bias_quant.flatten().numpy().tobytes())


    # ==== Weight: fp32 → int8 (clamp → stochastic rounding) ====
    weight = layer.weight.data

    # clipping
    clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
    total_elements = torch.numel(weight)
    clipped_max = torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight))
    print(f"layer has {clipped}/{total_elements} clipped weights. "
          f"Exceeding by {clipped_max} the maximum {kMaxWeight}.")

    weight_clamped = weight.clamp(-kMaxWeight, kMaxWeight)

    # scale to int8 range
    weight_scaled = weight_clamped * kWeightScale

    # stochastic rounding
    weight_quant = self.stochastic_round(weight_scaled).to(torch.int8)

    ascii_hist('fc weight:', weight_quant.numpy())
    self.save_histogram(
        f'{self.figure_index:02}_fully_connected_layer_weight.png',
        weight_quant, 'weight', 'frequency', 'fully connected layer weight'
    )
    self.figure_index += 1

    # ==== SIMD padding (32-byte alignment) ====
    num_input = weight_quant.shape[1]
    if num_input % 32 != 0:
      padded = ((num_input + 31) // 32) * 32
      new_w = torch.zeros(weight_quant.shape[0], padded, dtype=torch.int8)
      new_w[:, :num_input] = weight_quant
      weight_quant = new_w

    # Writer expects [outputs][inputs] flattened
    self.buf.extend(weight_quant.flatten().numpy().tobytes())
    print()

  def int32(self, v):
    self.buf.extend(struct.pack("<I", v))

class NNUEReader():
  def __init__(self, f, feature_set):
    self.f = f
    self.feature_set = feature_set
    self.model = M.NNUE(feature_set)
    fc_hash = NNUEWriter.fc_hash(self.model)

    self.read_header(feature_set, fc_hash)
    self.read_int32(feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.read_feature_transformer(self.model.input)
    self.read_int32(fc_hash) # FC layers hash
    self.read_fc_layer(self.model.l1)
    self.read_fc_layer(self.model.l2)
    self.read_fc_layer(self.model.output, is_output=True)

  def read_header(self, feature_set, fc_hash):
    self.read_int32(VERSION) # version
    self.read_int32(fc_hash ^ feature_set.hash ^ (M.L1*2)) # halfkp network hash
    desc_len = self.read_int32() # Network definition
    description = self.f.read(desc_len)

  def tensor(self, dtype, shape):
    d = numpy.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
    d = torch.from_numpy(d.astype(numpy.float32))
    d = d.reshape(shape)
    return d

  def read_feature_transformer(self, layer):
    layer.bias.data = self.tensor(numpy.int16, layer.bias.shape).divide(127.0)
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    weights = self.tensor(numpy.int16, layer.weight.shape[::-1])
    layer.weight.data = weights.divide(127.0).transpose(0, 1)

  def read_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers

    # FC inputs are padded to 32 elements for simd.
    non_padded_shape = layer.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, padded_shape).divide(kWeightScale)

    # Strip padding.
    layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

  def read_int32(self, expected=None):
    v = struct.unpack("<I", self.f.read(4))[0]
    if expected is not None and v != expected:
      raise Exception("Expected: %x, got %x" % (expected, v))
    return v

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
  parser.add_argument("target", help="Target file (can be .pt or .nnue)")
  features.add_argparse_args(parser)
  args = parser.parse_args()

  feature_set = features.get_feature_set_from_name(args.features)

  print('Converting %s to %s' % (args.source, args.target))

  if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
    if not args.target.endswith(".nnue"):
      raise Exception("Target file must end with .nnue")
    if args.source.endswith(".pt"):
      nnue = torch.load(args.source)
    else:
      nnue = M.NNUE.load_from_checkpoint(args.source, feature_set=feature_set)
    nnue.eval()
    writer = NNUEWriter(nnue, os.path.dirname(args.target))
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  elif args.source.endswith(".nnue"):
    if not args.target.endswith(".pt"):
      raise Exception("Target file must end with .pt")
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f, feature_set)
    torch.save(reader.model, args.target)
  else:
    raise Exception('Invalid filetypes: ' + str(args))

if __name__ == '__main__':
  main()
