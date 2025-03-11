# Converter from ros2 sensor_msgs/Image to OpenCV Image(numpy.ndarray)



## Usage

#### ROS2 -> CV2

```python
cv_image = ros2_numpy.numpify(image_msg)
```

The type of `image_msg` is `sensor_msgs/Image` or `sensor_msgs/CompresssedImage`.

#### CV2 -> ROS2

```python
image_msg = ros2_numpy.msgfy(cv_image)
```

The type of `image_msg` is `sensor_msgs/Image` or `sensor_msgs/CompresssedImage`. You can choose that type with argument. See bellow.



## Usage in details in CV2 -> ROS2

- Compression format PNG or JPEG
  ```python
  # jpeg
  cv_image = ros2.msgify(cv_image, compress_type = 'jpeg')
  # png
  cv_image = ros2.msgfy(cv_image, compress_type = 'png')
  ```

  
