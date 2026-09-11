"""YuE2 Seed Node for ComfyUI."""

import random

class YuE2Seed:
    """Dedicated seed generator node for YuE2 with randomize, increment, decrement, and fixed options."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "The random seed. Use the control below to randomize, increment, decrement, or keep fixed."
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_seed"
    CATEGORY = "YuE2/Utils"

    def get_seed(self, seed):
        if seed == 0:
            seed = random.randint(100000, 99999999)
        return (int(seed),)
