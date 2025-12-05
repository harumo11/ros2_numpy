# Converter from ros2 sensor_msgs/Image to OpenCV Image(numpy.ndarray)



## Usage

### import

```python
from ros2_numpy import numpify, msgify
```

or

```python
import ros2_numpy
```



#### ROS2 -> CV2

```python
cv_image = ros2_numpy.numpify(image_msg)
```

The type of `image_msg` is `sensor_msgs/Image` or `sensor_msgs/CompresssedImage`.

#### CV2 -> ROS2

```python
# sensor_msgs/Image (raw image)
color_image_msg = ros2_numpy.msgfy(cv_image) # bgr8 encoding and without compression 
mono_image_msg = ros2_numpy.msgfy(cv_image, encoding='mono8') # one-channel image
named_image_msg = ros2_numpy.msgfy(cv_image, frame_id='my_camera') # frame_id is passed to Image.header.frame_id field.

# sensor_msgs/CompressedImage (jpeg or png)
jpg_image_msg = ros2_numpy.msgfy(cv_image, compress_type='jpeg') # encoding='jpg' is also ok
png_image_msg = ros2_numpy.msgfy(cv_image, compress_type='png')

# compression level
jpg_image_msg = ros2_numpy.msgfy(cv_image, compress_type='jpeg', jpg_quality=40) # jpg_quality = [0, 100], default = 95
png_image_msg = ros2_numpy.msgfy(cv_image, compress_type='png', png_quality=5) # png_quality = [0, 9], default = 3
```



### encoding

When you send a non-compressed image, you must set `encoding` from below list. When you use compressed image, you don't consider it.

```python
    name_to_dtypes = {
        "rgb8": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "rgb16": (np.uint16, 3),
        "rgba16": (np.uint16, 4),
        "bgr8": (np.uint8, 3),
        "bgra8": (np.uint8, 4),
        "bgr16": (np.uint16, 3),
        "bgra16": (np.uint16, 4),
        "mono8": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        # for bayer image (based on cv_bridge.cpp)
        "bayer_rggb8": (np.uint8, 1),
        "bayer_bggr8": (np.uint8, 1),
        "bayer_gbrg8": (np.uint8, 1),
        "bayer_grbg8": (np.uint8, 1),
        "bayer_rggb16": (np.uint16, 1),
        "bayer_bggr16": (np.uint16, 1),
        "bayer_gbrg16": (np.uint16, 1),
        "bayer_grbg16": (np.uint16, 1),
        # OpenCV CvMat types
        "8UC1": (np.uint8, 1),
        "8UC2": (np.uint8, 2),
        "8UC3": (np.uint8, 3),
        "8UC4": (np.uint8, 4),
        "8SC1": (np.int8, 1),
        "8SC2": (np.int8, 2),
        "8SC3": (np.int8, 3),
        "8SC4": (np.int8, 4),
        "16UC1": (np.uint16, 1),
        "16UC2": (np.uint16, 2),
        "16UC3": (np.uint16, 3),
        "16UC4": (np.uint16, 4),
        "16SC1": (np.int16, 1),
        "16SC2": (np.int16, 2),
        "16SC3": (np.int16, 3),
        "16SC4": (np.int16, 4),
        "32SC1": (np.int32, 1),
        "32SC2": (np.int32, 2),
        "32SC3": (np.int32, 3),
        "32SC4": (np.int32, 4),
        "32FC1": (np.float32, 1),
        "32FC2": (np.float32, 2),
        "32FC3": (np.float32, 3),
        "32FC4": (np.float32, 4),
        "64FC1": (np.float64, 1),
        "64FC2": (np.float64, 2),
        "64FC3": (np.float64, 3),
        "64FC4": (np.float64, 4),
    }
```



## future works

- [ ] Automatically detect encoding.
