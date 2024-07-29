# U-Net-Builder
Quickly create various types of U-Net networks.

**This is a development version and may contain errors.**


## Usage example

```python
import torch
from unet_builder import UNet

ch_input = 3
ch_output = 1


model = UNet(   ch_input,
                ch_output,
                init_features = 64, 
                u_blocks_amount = [ 1,  1,  1,  8,  1,  1,  1],
                u_blocks_variant= ['C','C','C','R','C','C','C'],
                u_blocks_resize = ['D','D','D','N','U','U','U'],
                u_connected = False,
                fin_act = torch.nn.Sigmoid()
)

# random input
batch_of_images = torch.rand(8, 3, 64, 64) # batch, channels, height, width

# print model description
print(model.description)

# print shape of the output
print("input: ", batch_of_images.shape)
print("output:", model(batch_of_images).shape)
```

**Returns**

```bash
--------unet--------
INIT Conv 3 -> 64
1, C, D, 64 -> 128
1, C, D, 128 -> 256
1, C, D, 256 -> 512
8, R, N, 512 -> 512
1, C, U, 512 -> 256
1, C, U, 256 -> 128
1, C, U, 128 -> 64
FINAL Conv 64 -> 1
FINAL Act Sigmoid()
--------------------
input:  torch.Size([8, 3, 64, 64])
output: torch.Size([8, 1, 64, 64])
```

## Description

**u_blocks_amount**
Numbers of corresponding blocks to run.
In case the amount is > 1 and corresponding u_blocks_resize is 'D' or 'U' then the change of height, width and number of features occurs in the first block.

**u_blocks_variant**
* C -> Convolution block
* R -> Residual block
* E -> Squeeze Excitation Residual block
* A -> Position Attention + Scaled Dot Product Attention (it supports only resize 'N')
* T -> Criss Cross Attention block (it supports only resize 'N')

**u_blocks_resize**
* D -> downsample (halve the height and width but double the number of features)
* U -> upsample (double the height and width but halve the number of features)
* N -> none (keep unchanged height, width and the number of features)

**u_connected**
True -> Concatenate downsampled tensors with upsampled tensors from both sides of U.
False -> Do not concatenate downsampled tensors with upsampled tensors from both sides of U.
