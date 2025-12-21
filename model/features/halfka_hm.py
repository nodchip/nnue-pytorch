from collections import OrderedDict

from .feature_block import FeatureBlock


# ===== Shogi board constants =====
FILE_NB = 9
RANK_NB = 9
SQ_NB = 81

# Square numbering assumption:
#   sq = file * 9 + rank   (file: 0..8, rank: 0..8)
# Then SQ_61 (6-file, 1-rank) == file=5, rank=0 -> 5*9+0 = 45
SQ_61 = 45  # threshold: 6..9 files are sq >= 45 under file-major numbering


# ===== BonaPiece layout (must match your YaneuraOu build) =====
# If your build defines DISTINGUISH_GOLDS, set this to True.
# (sfnnwoP1536 でどちらかはあなたのビルドに合わせてください)
DISTINGUISH_GOLDS = False


# Hand layout (Apery style) used when EVAL_NNUE etc. is enabled in YaneuraOu
FE_HAND_END = 90  # fe_hand_end in evaluate.h


def _calc_bonapiece_ends(distinguish_golds: bool):
    """
    Reconstruct BonaPiece enumerator values from evaluate.h.
    We only need:
      - fe_hand_end (= 90)
      - fe_end
      - f_king
      - e_king
      - (packed piece range size) = e_king  (because enemy king is packed by -SQ_NB)
    """
    # Old set: pawn..dragon (friend/enemy each), 9 piece-kinds * 2 colors = 18 blocks of 81
    # f_pawn starts at FE_HAND_END
    fe_old_end = FE_HAND_END + 18 * SQ_NB  # e_dragon + 81

    if distinguish_golds:
        # Add promoted small pieces distinct from gold:
        # pro_pawn, pro_lance, pro_knight, pro_silver (friend/enemy each) => 8 blocks of 81
        fe_new_end = fe_old_end + 8 * SQ_NB
    else:
        fe_new_end = fe_old_end

    fe_end = fe_new_end
    f_king = fe_end
    e_king = f_king + SQ_NB
    fe_end2 = e_king + SQ_NB

    return fe_end, f_king, e_king, fe_end2


FE_END, F_KING, E_KING, FE_END2 = _calc_bonapiece_ends(DISTINGUISH_GOLDS)

# Real feature dimensions follow YaneuraOu half_ka_hm.h:
#   kDimensions = 5 * FILE_NB * e_king
# where e_king here is BonaPiece::e_king enumerator value (start of enemy king range).
NUM_KSQ = 5 * FILE_NB  # 45 king squares after hm (files 1..5)
NUM_INPUTS = NUM_KSQ * E_KING

# Factorized (^) follows nnue-pytorch idea:
# add one factor "A" for the piece-side index.
# virtual piece-side size is E_KING + SQ_NB to split packed enemy king back out.
NUM_PLANES_VIRTUAL = E_KING + SQ_NB


def mir_file(sq: int) -> int:
    """Mirror square by file (left-right)."""
    f = sq // RANK_NB
    r = sq % RANK_NB
    mf = (FILE_NB - 1) - f
    return mf * RANK_NB + r


def make_index_hm(sq_k: int, p: int) -> int:
    """
    Match YaneuraOu half_ka_hm.cpp:

      if (sq_k >= SQ_61) { sq_k = Mir(sq_k); if (p >= fe_hand_end) mirror square-part of p; }
      return e_king * sq_k + (p >= e_king ? p - SQ_NB : p);

    Notes:
      - sq_k is the associated king square (already perspective-transformed in engine side).
      - p is BonaPiece (in perspective view) value.
    """
    if sq_k >= SQ_61:
        # king is on files 6..9 -> mirror to 4..1
        sq_k = mir_file(sq_k)

        # mirror ONLY board pieces; hand pieces are NOT mirrored
        if p >= FE_HAND_END:
            rel = p - FE_HAND_END
            piece_index = rel // SQ_NB
            sq_p = rel % SQ_NB
            sq_p = mir_file(sq_p)
            p = FE_HAND_END + piece_index * SQ_NB + sq_p

    # pack enemy king plane into friend king plane
    # (enemy king range [E_KING, E_KING+80] becomes [F_KING, F_KING+80] by -81)
    packed_p = p - SQ_NB if p >= E_KING else p

    return E_KING * sq_k + packed_p


def decode_real_feature(idx: int) -> tuple[int, int]:
    """
    idx -> (sq_k, packed_p)
    where sq_k in [0..44] (file<=4 after hm), packed_p in [0..E_KING-1]
    """
    sq_k = idx // E_KING
    packed_p = idx % E_KING
    return sq_k, packed_p


def ksq_actual_from_bucket(sq_k_bucket: int) -> int:
    """
    sq_k_bucket is in [0..44] and corresponds to squares with file 1..5 after hm.
    Under file-major numbering:
      file = sq_k_bucket // 9  (0..4)
      rank = sq_k_bucket % 9
      actual_sq = file*9 + rank
    """
    f = sq_k_bucket // RANK_NB
    r = sq_k_bucket % RANK_NB
    return f * RANK_NB + r


def factor_A_index(idx: int) -> int:
    """
    Factorizer rule (analogous to halfka_v2_hm.py):
      - A is "piece-side" index.
      - Packed enemy king must be split back out to virtual space.

    Our packed mapping merges enemy king squares into friend king range [F_KING..F_KING+80].
    For a real feature at (sq_k, packed_p), if packed_p is in the king range but its square != ksq square,
    we treat it as the *opponent* king and map it to the virtual "enemy king" plane by +SQ_NB.
    """
    sq_k_bucket, packed_p = decode_real_feature(idx)

    a = packed_p

    # king range in packed space is [F_KING .. F_KING+80]
    if F_KING <= a < F_KING + SQ_NB:
        king_sq = ksq_actual_from_bucket(sq_k_bucket)
        piece_sq = a - F_KING
        # If it's not the associated king square, it's the packed opponent king -> move to virtual enemy-king plane
        if piece_sq != king_sq:
            a += SQ_NB  # virtual split: friend-king plane [F_KING..] and enemy-king plane [F_KING+81..]
    return a


class Features(FeatureBlock):
    def __init__(self):
        # Hash: YaneuraOu half_ka_hm.h uses 0x5f134cb9u ^ side.
        # Here we expose a single "HalfKA_hm" block; keep base hash.
        super().__init__("HalfKA_hm", 0x5F134CB9, OrderedDict([("HalfKA_hm", NUM_INPUTS)]))

    def get_active_features(self, board):
        raise Exception(
            "Not supported: for training you should use the C++ data loader. "
            "This python module provides the feature definition and factorizer mapping."
        )


class FactorizedFeatures(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKA_hm^",
            0x5F134CB9,
            OrderedDict([("HalfKA_hm", NUM_INPUTS), ("A", NUM_PLANES_VIRTUAL)]),
        )

    def get_active_features(self, board):
        raise Exception(
            "Not supported: you must use the C++ data loader for factorizer support during training."
        )

    def get_feature_factors(self, idx: int) -> list[int]:
        if idx >= self.num_real_features:
            raise Exception("Feature must be real")

        a_idx = factor_A_index(idx)
        return [idx, self.get_factor_base_feature("A") + a_idx]


def get_feature_block_clss() -> list[type[FeatureBlock]]:
    return [Features, FactorizedFeatures]
