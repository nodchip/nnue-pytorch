from collections.abc import MutableMapping

import torch


COMPILED_MODEL_STATE_PREFIX = "_compiled_model._orig_mod."
MODEL_STATE_PREFIX = "model."


def strip_compiled_model_state_dict_keys(state_dict: MutableMapping) -> int:
    """Remove torch.compile cache weights from a Lightning checkpoint state_dict."""
    removed = 0
    for key in list(state_dict.keys()):
        if not key.startswith(COMPILED_MODEL_STATE_PREFIX):
            continue

        value = state_dict.pop(key)
        removed += 1

        model_key = MODEL_STATE_PREFIX + key[len(COMPILED_MODEL_STATE_PREFIX) :]
        if model_key not in state_dict:
            state_dict[model_key] = value

    return removed


def load_nnue_from_checkpoint(checkpoint_path: str, **kwargs):
    from ..lightning_module import NNUE

    map_location = kwargs.pop("map_location", None)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    strip_compiled_model_state_dict_keys(checkpoint["state_dict"])

    model = NNUE(**kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    return model
