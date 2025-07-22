# Understanding CNN-s on Deep Learning network
**1. What is Image Data?**
- Digital Images are just numbers representing pixel intensities.

# Types of Images

## Grayscale Image

- Each pixel is a single value between 0 (black) and 255 (white)
- Example: A 28×28 grayscale image has shape `(28, 28)`
- Total pixels = 28 × 28 = 784

## RGB Image

- Each pixel has 3 values (Red, Green, Blue channels)
- Each channel value ranges 0-255
- Example: A 64×64 RGB image has shape `(64, 64, 3)`
- Total values = 64 × 64 × 3 = 12,288

## Visualization

A 3×3 grayscale image might look like this in matrix form:
[[ 0 128 255 ]
[64 192 128 ]
[32 96 224 ]]

Code sample for Greyscale numerical generation
import numpy as np
gray_img = np.random.randint(0, 256, (5,5))
print(gray_img)

