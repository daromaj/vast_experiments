#!/usr/bin/env python3
"""Patch pipeline to handle text_encoder on CPU."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/LongCat-Video/longcat_video/pipeline_longcat_video_avatar.py"

with open(path, 'r') as f:
    content = f.read()

# Fix encode_prompt to use text_encoder's device
old_encode = '''        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state'''

new_encode = '''        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        te_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(te_device), mask.to(te_device)).last_hidden_state'''

# Fix _get_t5_prompt_embeds to use text_encoder's device
old_embeds = '''    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.dit.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds = self.encode_prompt(prompt=prompt, device=device, num_videos_per_prompt=1)'''

new_embeds = '''    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.dit.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Use text_encoder's device (CPU) for encoding to save VRAM
        te_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.encode_prompt(prompt=prompt, device=te_device, num_videos_per_prompt=1)'''

patched = 0
if old_encode in content:
    content = content.replace(old_encode, new_encode)
    patched += 1
    print("Patched encode_prompt")
if old_embeds in content:
    content = content.replace(old_embeds, new_embeds)
    patched += 1
    print("Patched _get_t5_prompt_embeds")

with open(path, 'w') as f:
    f.write(content)

print(f"Patched {path}: {patched} functions fixed for text_encoder on CPU")
