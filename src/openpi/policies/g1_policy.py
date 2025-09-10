import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_g1_example() -> dict:
    """Creates a random input example for the G1 policy."""
    return {
        "state": np.ones((31,)),
        "images": {
            "cam_back": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_top": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_back": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_back": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_hand": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_hand": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Inputs for the G1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [31]
    - actions: [action_horizon, 31]
    """
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        data = _decode_g1_data(data)
        images = {
            "base_0_rgb": _parse_image(data["images"]["cam_back"]),
            "left_wrist_0_rgb": _parse_image(data["images"]["cam_left_hand"]),
            "right_wrist_0_rgb": _parse_image(data["images"]["cam_right_hand"]),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        inputs = {
            "state": data["state"],
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Outputs for the G1 policy."""
    def __call__(self, data: dict) -> dict:
        # Only return the first 31 dims.
        return {"actions": np.asarray(data["actions"][:, :31])}

def _decode_g1_data(data: dict) -> dict:
    # state is [base joint, shoulder joint, elbow joint, wrist joint, hand_0 joint, hand_1 joint. hand_2 joint]
    # dim sizes: [3, 6, 2, 6, 6, 6, 2]
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data

