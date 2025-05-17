# VLM-R1 with xformers

Source Project: [VLM-R1: A stable and generalizable R1-style Large Vision-Language Model](https://github.com/om-ai-lab/VLM-R1)

This project aims to deploy VLM-R1 to Volta architecture GPUs.
(The flash-attn that the project relies on does not support Volta GPUs, so xformers are used instead of flash-attn.)

This modification has been shown to run the entire training process on a V100(Mem: 32GB)*8 server.

## Setup

Recommended CUDA version: 11.8

```bash
conda create -n vlm-r1 python=3.10
conda activate vlm-r1
pip install torch==2.3.1
pip install transformers==4.50.3
pip install datasets==3.5.0
pip install trl==0.16.0
pip install deepspeed==0.15.3
pip install peft==0.15.1
pip install xformers --no-deps
bash setup.sh
```

## Core changes

Modified at `VLM-R1_with_xformers\src\open-r1-multimodal\src\open_r1\qwen2_5vl_monkey_patch.py`:

```python
import xformers.ops as xops

def qwen2_5vl_vision_xformers_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            cos = cos.to(torch.float)
            sin = sin.to(torch.float())
        
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        # Use xformers memory-efficient attention
        attn_output = xops.memory_efficient_attention(q, k, v)
        
        attn_output = self.proj(attn_output)
        return attn_output

def monkey_patch_qwen2_5vl_flash_attn():
    Qwen2_5_VLVisionFlashAttention2.forward = qwen2_5vl_vision_xformers_forward
```

