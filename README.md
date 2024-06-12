# jccc: A Python Package for Color Conversion Compatible with Numba

## Overview

`jccc` (**J**IT **C**ompilable **C**olor **C**onversion) is a Python package that provides fast color conversion and quantization functions compatible with [Numba](https://github.com/numba/numba).

## Features

- **Numba-Compatible:** All functions are optimized for performance using Numba JIT compilation.
- **Wide Range of Color Conversions:** Support for common color spaces including RGB, HSV, LAB, and more.
- **Ease of Use:** Simple and intuitive API for quick integration into your projects.

## Installation

You can install `jccc` via pip:

```sh
pip install jccc
```

## Requirements

- Python 3.6+
- Numba 0.53.1+

## Getting Started

### Importing the Package

To start using `jccc`, you first need to import the package and its functions:

```python
import jccc
from jccc import rgb_to_hsv, hsv_to_rgb
```

### Usage

Here are some examples demonstrating how to use the color conversion functions provided by `jccc`.

#### RGB to HSV Conversion

```python
import numpy as np
from jccc import rgb_to_hsv

# Define an RGB color
rgb_color = np.array([255, 0, 0])

# Convert RGB to HSV
hsv_color = rgb_to_hsv(rgb_color)

print("HSV Color:", hsv_color)
```

#### HSV to RGB Conversion

```python
import numpy as np
from jccc import hsv_to_rgb

# Define an HSV color
hsv_color = np.array([0, 1, 1])

# Convert HSV to RGB
rgb_color = hsv_to_rgb(hsv_color)

print("RGB Color:", rgb_color)
```

## Supported Conversions

Conversions are supported between the following:

- RGB
- HSL
- HSV
- XYZ
- CIELAB


## Contributing

Contributions to the `jccc` package are welcome. If you have any suggestions or bug reports or want to contribute code, please open an issue or submit a pull request on our GitHub repository.

## License

`jccc` is released under the MIT License. See the LICENSE file for more details.


