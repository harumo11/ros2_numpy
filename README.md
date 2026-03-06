# ros2-numpy

A collection of conversion tools between ROS 2 messages and numpy arrays.

## Features

- **sensor_msgs/Image** ↔ **numpy.ndarray** (OpenCV format)
- **sensor_msgs/CompressedImage** ↔ **numpy.ndarray**
- **sensor_msgs/Image / CompressedImage** → **PIL Image**

## Usage

### Import

```python
from ros2_numpy import numpify, msgfy, pilfy
```

### ROS 2 -> Numpy / PIL

#### To Numpy Array (OpenCV)
```python
# Returns (H, W, C) for color or (H, W) for grayscale
cv_image = numpify(image_msg)
```
Supported message types: `sensor_msgs/Image`, `sensor_msgs/CompressedImage`.

#### To PIL Image
```python
pil_image = pilfy(image_msg)
```

### Numpy / PIL -> ROS 2

#### Raw Image (`sensor_msgs/Image`)
```python
# Automatic encoding detection
msg = msgfy(cv_image)

# Explicit encoding and frame_id
msg = msgfy(cv_image, encoding='rgb8', frame_id='camera_frame')
```

#### Compressed Image (`sensor_msgs/CompressedImage`)
```python
# JPEG compression (default quality: 95)
msg = msgfy(cv_image, compress_type='jpeg', jpg_quality=90)

# PNG compression (default level: 3)
msg = msgfy(cv_image, compress_type='png', png_compression=5)
```

## Supported Encodings

The package supports most common ROS 2 image encodings:
- RGB: `rgb8`, `rgb16`, `rgba8`, `rgba16`
- BGR: `bgr8`, `bgr16`, `bgra8`, `bgra16`
- Mono: `mono8`, `mono16`
- Bayer: `bayer_rggb8`, `bayer_bggr8`, etc.
- OpenCV/CvMat types: `8UC1`, `32FC1`, etc.

Encoding is automatically detected from the numpy array's `dtype` and number of channels if not specified in `msgfy`.

## Development

### Running Tests

You can run the unit tests using `unittest`:

```bash
# Run tests from the package directory
python3 ros2_numpy/tests/test_ros2_numpy.py
```

Or using `colcon` if the package is built in a ROS 2 workspace:

```bash
colcon test --packages-select ros2_numpy
```
