import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

"""
Original state order (25-dim):
  base_x_joint, base_y_joint, base_yaw_link,
  body_z_joint, body_y_joint,
  head_z_joint, head_y_joint,
  left_shoulder_y, left_shoulder_x, left_shoulder_z,
  left_elbow_y, left_elbow_x, left_wrist_y, left_wrist_z,
  left_gripper1, left_gripper2,
  right_shoulder_y, right_shoulder_x, right_shoulder_z,
  right_elbow_y, right_elbow_x, right_wrist_y, right_wrist_z,
  right_gripper1, right_gripper2

Original action order (21-dim):
  base_action (3), body_action (2),
  left_arm_action (7), right_arm_action (7),
  left_gripper_action (1), right_gripper_action (1)

State is remapped to match action layout by dropping head (indices 5, 6)
and the second gripper joints (indices 15, 24), then reordering to:
  base(3) -> body(2) -> left_arm(7) -> right_arm(7) -> left_gripper(1) -> right_gripper(1)
Indices below are into the original 25-dim state (25 -> 21).
"""
_X7S_STATE_PERM = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 14, 23]

def make_x7s_example() -> dict:
    """Creates a random input example for the X7S policy."""
    return {
        "state": np.ones((25,)), # in original state ordering
        "images/front": np.random.randint(256, size=(3, 720, 1280), dtype=np.uint8),
        "images/left": np.random.randint(256, size=(3, 360, 640), dtype=np.uint8),
        "images/right": np.random.randint(256, size=(3, 360, 640), dtype=np.uint8),
        "prompt": "do something",
    }

def _decode_x7s_data(data: dict) -> dict:
    # original state is [base, body, head, left_arm, right_arm, left_gripper, right_gripper]
    # dim size: [3, 2, 2, 7, 7, 2, 2]
    state = np.asarray(data["state"])
    
    # reorder state to align with action
    # now state is [base, body, left_arm, right_arm, left_gripper, right_gripper]
    # dim sizes: [3, 2, 7, 7, 1, 1]
    # state = state[_X7S_STATE_PERM]
    
    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    data["images"] = {
        "front": convert_image(data["images/front"]),
        "left": convert_image(data["images/left"]),
        "right": convert_image(data["images/right"]),
    }
    data["state"] = state
    return data

@dataclasses.dataclass(frozen=True)
class X7SInputs(transforms.DataTransformFn):
    """Inputs for the X7S policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [25]
    - actions: [action_horizon, 21]
    """
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        data = _decode_x7s_data(data)
        images = {
            "base_0_rgb": data["images"]["front"],
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
class X7SOutputs(transforms.DataTransformFn):
    """Outputs for the X7S policy."""
    action_dim: int = 21

    def __call__(self, data: dict) -> dict:
        # Only return the first action_dim dims.
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}



