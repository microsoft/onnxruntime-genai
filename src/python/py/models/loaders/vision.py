# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import torch


class Qwen38VisionLoader:
    def load_visual(self):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig  # noqa: PLC0415
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: PLC0415
            Qwen4ExpVisionModel,
            Qwen4ExpVisionRotaryEmbedding,
        )

        try:
            config = Qwen4ExpVisionConfig(**self.vision_config)
            with torch.device("meta"):
                visual = Qwen4ExpVisionModel(config)
            prefix = "model.visual."
            state = {
                name[len(prefix) :]: self.get_tensor(name) for name in self.tensor_names() if name.startswith(prefix)
            }
            visual.load_state_dict(state, strict=True, assign=True)
            visual.rotary_pos_emb = Qwen4ExpVisionRotaryEmbedding(config)
            return visual
        finally:
            self.close()
