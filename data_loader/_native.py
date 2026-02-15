import ctypes
import os
import glob

import numpy as np
import torch

from .config import CDataloaderSkipConfig


class SparseBatch(ctypes.Structure):
    _fields_ = [
        ("num_inputs", ctypes.c_int),
        ("size", ctypes.c_int),
        ("is_white", ctypes.POINTER(ctypes.c_float)),
        ("outcome", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("num_active_white_features", ctypes.c_int),
        ("num_active_black_features", ctypes.c_int),
        ("max_active_features", ctypes.c_int),
        ("white", ctypes.POINTER(ctypes.c_int)),
        ("black", ctypes.POINTER(ctypes.c_int)),
        ("white_values", ctypes.POINTER(ctypes.c_float)),
        ("black_values", ctypes.POINTER(ctypes.c_float)),
        ("layer_stack_indices", ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device):
        white_values = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.white_values, shape=(self.size, self.max_active_features)
                )
            )
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        black_values = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.black_values, shape=(self.size, self.max_active_features)
                )
            )
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        white_indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.white, shape=(self.size, self.max_active_features)
                )
            )
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        black_indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.black, shape=(self.size, self.max_active_features)
                )
            )
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        us = (
            torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1)))
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        them = 1.0 - us
        outcome = (
            torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1)))
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        score = (
            torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1)))
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        layer_stack_indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))
            )
            .long()
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        return (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            layer_stack_indices,
        )


class ProgressBatch(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("indices_per_position", ctypes.c_int),
        ("num_weights", ctypes.c_int),
        ("ply", ctypes.POINTER(ctypes.c_uint16)),
        ("indices", ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device):
        plies = (
            torch.from_numpy(np.ctypeslib.as_array(self.ply, shape=(self.size,)))
            .long()
            .to(device=device, non_blocking=True)
        )
        indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.indices,
                    shape=(self.size, self.indices_per_position),
                )
            )
            .long()
            .to(device=device, non_blocking=True)
        )
        return plies, indices, self.num_weights


class ProgressSfenBatch(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("indices_per_position", ctypes.c_int),
        ("num_weights", ctypes.c_int),
        ("ply", ctypes.POINTER(ctypes.c_uint16)),
        ("indices", ctypes.POINTER(ctypes.c_int)),
        ("sfens", ctypes.POINTER(ctypes.c_char_p)),
    ]

    def get_items(self, device):
        plies = (
            torch.from_numpy(np.ctypeslib.as_array(self.ply, shape=(self.size,)))
            .long()
            .to(device=device, non_blocking=True)
        )
        indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.indices,
                    shape=(self.size, self.indices_per_position),
                )
            )
            .long()
            .to(device=device, non_blocking=True)
        )
        sfens = [self.sfens[i].decode("utf-8") for i in range(self.size)]
        return plies, indices, sfens, self.num_weights


class Fen(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("fen", ctypes.c_char_p)]


class FenBatch(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("fens", ctypes.POINTER(Fen))]

    def get_fens(self):
        strings = []
        for i in range(self.size):
            strings.append(self.fens[i].fen.decode("utf-8"))
        return strings


class CDataLoaderAPI:
    def __init__(self):
        self.dll = self._load_library()
        self.has_progress_api = False
        self.has_progress_sfen_api = False
        self._define_prototypes()

    def _load_library(self):
        for lib in glob.glob("./*training_data_loader.*"):
            if not (
                lib.endswith(".so") or lib.endswith("dll") or lib.endswith(".dylib")
            ):
                continue
            return ctypes.cdll.LoadLibrary(os.path.abspath(lib))
        raise FileNotFoundError("Cannot find data_loader shared library.")

    def _define_prototypes(self):
        # EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(
        #     const char* feature_set_c,
        #     int concurrency,
        #     int num_files,
        #     const char* const* filenames,
        #     int batch_size,
        #     bool cyclic,
        #     DataloaderSkipConfig config
        # )
        self.dll.create_sparse_batch_stream.restype = ctypes.c_void_p
        self.dll.create_sparse_batch_stream.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_bool,
            CDataloaderSkipConfig,
        ]

        # EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
        self.dll.destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

        # EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
        self.dll.fetch_next_sparse_batch.restype = ctypes.POINTER(SparseBatch)
        self.dll.fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]

        # EXPORT Stream<ProgressBatch>* CDECL create_progress_batch_stream(
        #     int concurrency,
        #     int num_files,
        #     const char* const* filenames,
        #     int batch_size,
        #     bool cyclic,
        #     DataloaderSkipConfig config
        # )
        try:
            self.dll.create_progress_batch_stream.restype = ctypes.c_void_p
            self.dll.create_progress_batch_stream.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int,
                ctypes.c_bool,
                CDataloaderSkipConfig,
            ]

            self.dll.destroy_progress_batch_stream.argtypes = [ctypes.c_void_p]

            self.dll.fetch_next_progress_batch.restype = ctypes.POINTER(ProgressBatch)
            self.dll.fetch_next_progress_batch.argtypes = [ctypes.c_void_p]

            self.dll.destroy_progress_batch.argtypes = [ctypes.POINTER(ProgressBatch)]
            self.has_progress_api = True
        except AttributeError:
            self.has_progress_api = False

        try:
            self.dll.create_progress_sfen_batch_stream.restype = ctypes.c_void_p
            self.dll.create_progress_sfen_batch_stream.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int,
                ctypes.c_bool,
                CDataloaderSkipConfig,
            ]
            self.dll.destroy_progress_sfen_batch_stream.argtypes = [ctypes.c_void_p]
            self.dll.fetch_next_progress_sfen_batch.restype = ctypes.POINTER(
                ProgressSfenBatch
            )
            self.dll.fetch_next_progress_sfen_batch.argtypes = [ctypes.c_void_p]
            self.dll.destroy_progress_sfen_batch.argtypes = [
                ctypes.POINTER(ProgressSfenBatch)
            ]
            self.has_progress_sfen_api = True
        except AttributeError:
            self.has_progress_sfen_api = False


type SparseBatchPtr = ctypes._Pointer[SparseBatch]
type FenBatchPtr = ctypes._Pointer[FenBatch]
type ProgressBatchPtr = ctypes._Pointer[ProgressBatch]
type ProgressSfenBatchPtr = ctypes._Pointer[ProgressSfenBatch]


try:
    c_lib = CDataLoaderAPI()
except FileNotFoundError as e:
    print(e)
    exit(1)
