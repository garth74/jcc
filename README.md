# JCCC

This package provides several color conversion methods that are compatible with [numba](https://github.com/numba/numba). It also provides a [color quantization](https://en.wikipedia.org/wiki/Color_quantization) procedure, which can reduce an image to a set of predetermined pixels. This isn't ideal for compression tasks but is useful for tagging images with color names.

## Installation

```
pip install jccc
```

## Supported conversions

### RGB

- RGB --> HSV
- RGB --> HLS
- RGB --> XYZ
- RGB --> CIELAB

### HSV

- HSV --> RGB
- HSV --> HLS
- HSV --> XYZ
- HSV --> CIELAB

### HLS

- HLS --> RGB
- HLS --> HSV
- HLS --> XYZ
- HLS --> CIELAB

### XYZ

- XYZ --> RGB
- XYZ --> HSV
- XYZ --> HLS
- XYZ --> CIELAB

### CIE LAB

- CIELAB --> RGB
- CIELAB --> HSV
- CIELAB --> HLS
- CIELAB --> XYZ
