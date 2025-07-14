# PRoPE
https://www.liruilong.cn/prope/

This is the official repo for the paper

"PRoPE: Projective Positional Encoding for Multiview Transformers"

**TL;DR**: We introduce **PRoPE**, a projective positional encoding for multiview transformers that directly injects *relative camera geometry*—in the form of projective transformations—into the attention mechanism. Inspired by Rotary Positional Encoding (RoPE) in LLMs, PRoPE enhances cross-view reasoning with *no additional overhead*, remains *compatible with flash attention*, and *naturally reduces to RoPE in the single-view setting*. It delivers noticeable and consistent improvements across a diverse range tasks that requires cross-view understanding.

## Implementations

The implementation of PRoPE is extremely simple and efficient. We provide standalone, single-file implementations for both JAX and PyTorch in [`prope/jax.py`](prope/jax.py) and [`prope/torch.py`](prope/torch.py). 

## Example of Usages

Here we demo with PyTorch version:

```python
# Say we have C images, each carries with camera infomation, which would be used for cross-view understanding.
viewmats: Tensor # (B, C, 4, 4) camera world-to-camera matrix
Ks: Tensor # (B, C, 3, 3) camera intrinsic matrix

# In transformer we typically patchify the images into tokens. Say
# the image size is (256, 384) and patch size is 16.
image_width, image_height = 256, 384
patches_x, patches_y = image_width / 16, image_height / 16

# And our attention layer has mapped the images from pixels (B, C, 384, 256) to Q/K/V tokens with shape (B, num_heads, seqlen, head_dim), where `seqlen = C * patches_x * patches_y`
Q, K, V: Tensor = ... # (B, num_heads, seqlen, head_dim)

# Injecting the camera information is simply replacing the native torch attention with our impl:
output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
# -->
output = prope_dot_product_attention(
    Q, K, V,viewmats=viewmats, Ks=Ks, patches_x=patches_x, patches_y=patches_y, image_width=image_width, image_height=image_height
)
```

## Experiments

- Improve LVSM on the task of Novel View Syntheis: [Checkout `nvs` branch](https://github.com/liruilong940607/prope/tree/nvs)
- Improve UniMatch on the task of Stereo Depth Estimation: To be released
