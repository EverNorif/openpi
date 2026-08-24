import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_yam_example() -> dict:
    """Creates a random input example for the YAM policy."""
    return {
        "state": np.ones((14,)), # in original state ordering
        "images/top": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "images/left": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "images/right": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "prompt": "do something",
    }

def _decode_yam_data(data: dict) -> dict:
    state = np.asarray(data["state"])
    
    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    data["images"] = {
        "top": convert_image(data["images/top"]),
        "left": convert_image(data["images/left"]),
        "right": convert_image(data["images/right"]),
    }
    data["state"] = state
    return data

@dataclasses.dataclass(frozen=True)
class YAMInputs(transforms.DataTransformFn):
    """Inputs for the YAM policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [14]
    - actions: [14]
    """
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        data = _decode_yam_data(data)
        images = {
            "base_0_rgb": data["images"]["top"],
            "left_wrist_0_rgb": data["images"]["left"],
            "right_wrist_0_rgb": data["images"]["right"],
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
class YAMOutputs(transforms.DataTransformFn):
    """Outputs for the YAM policy."""
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        # Only return the first action_dim dims.
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}



