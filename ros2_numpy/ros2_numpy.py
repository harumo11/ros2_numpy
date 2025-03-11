# this script is a ROS2 node that listens to the /image_raw topic and converts the image to a numpy array

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import rclpy
import cv2
import numpy as np


class ToNumpyConverter():
    name_to_dtypes = {
        "rgb8":    (np.uint8,  3),
        "rgba8":   (np.uint8,  4),
        "rgb16":   (np.uint16, 3),
        "rgba16":  (np.uint16, 4),
        "bgr8":    (np.uint8,  3),
        "bgra8":   (np.uint8,  4),
        "bgr16":   (np.uint16, 3),
        "bgra16":  (np.uint16, 4),
        "mono8":   (np.uint8,  1),
        "mono16":  (np.uint16, 1),

        # for bayer image (based on cv_bridge.cpp)
        "bayer_rggb8":	(np.uint8,  1),
        "bayer_bggr8":	(np.uint8,  1),
        "bayer_gbrg8":	(np.uint8,  1),
        "bayer_grbg8":	(np.uint8,  1),
        "bayer_rggb16":	(np.uint16, 1),
        "bayer_bggr16":	(np.uint16, 1),
        "bayer_gbrg16":	(np.uint16, 1),
        "bayer_grbg16":	(np.uint16, 1),

        # OpenCV CvMat types
        "8UC1":    (np.uint8,   1),
        "8UC2":    (np.uint8,   2),
        "8UC3":    (np.uint8,   3),
        "8UC4":    (np.uint8,   4),
        "8SC1":    (np.int8,    1),
        "8SC2":    (np.int8,    2),
        "8SC3":    (np.int8,    3),
        "8SC4":    (np.int8,    4),
        "16UC1":   (np.uint16,   1),
        "16UC2":   (np.uint16,   2),
        "16UC3":   (np.uint16,   3),
        "16UC4":   (np.uint16,   4),
        "16SC1":   (np.int16,  1),
        "16SC2":   (np.int16,  2),
        "16SC3":   (np.int16,  3),
        "16SC4":   (np.int16,  4),
        "32SC1":   (np.int32,   1),
        "32SC2":   (np.int32,   2),
        "32SC3":   (np.int32,   3),
        "32SC4":   (np.int32,   4),
        "32FC1":   (np.float32, 1),
        "32FC2":   (np.float32, 2),
        "32FC3":   (np.float32, 3),
        "32FC4":   (np.float32, 4),
        "64FC1":   (np.float64, 1),
        "64FC2":   (np.float64, 2),
        "64FC3":   (np.float64, 3),
        "64FC4":   (np.float64, 4)
    }

    @staticmethod
    def from_image(msg: Image):
        """
        Convert sensor_msgs.msg.Image to numpy.ndarray
        """
        # get dtype and channel from encoding
        if msg.encoding in ToNumpyConverter.name_to_dtypes:
            dtype, channel = ToNumpyConverter.name_to_dtypes[msg.encoding]
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")

        # convert to numpy.ndarray
        numpy_image = np.frombuffer(msg.data, dtype=dtype)
        numpy_image = numpy_image.reshape((msg.height, msg.width, channel))

        return numpy_image

    @staticmethod
    def from_compressed_image(msg: CompressedImage):
        """
        Convert sensor_msgs.msg.CompressedImage to numpy.ndarray
        """
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image_np


class ToRosMsgConverter():
    @staticmethod
    def to_image(numpy_image: np.ndarray, encoding: str = 'bgr8', frame_id: str = 'camera') -> Image:
        """
        Convert numpy.ndarray to sensor_msgs.msg.Image
        """
        msg = Image()
        msg.height = numpy_image.shape[0]
        msg.width = numpy_image.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = numpy_image.shape[1] * numpy_image.shape[2]
        msg.data = numpy_image.tobytes()
        # msg.data = numpy_image.reshape(1)
        msg.header.frame_id = frame_id
        return msg

    @staticmethod
    def to_jpeg_image(numpy_image: np.ndarray, quality: int = 95) -> CompressedImage:
        """
        Convert numpy.ndarray to sensor_msgs.msg.CompressedImage
        """
        msg = CompressedImage()
        msg.format = "jpeg"
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, data = cv2.imencode('.jpg', numpy_image, encode_param)
        msg.data = data.tobytes()
        if ret == False:
            raise ValueError("Failed to encode image")
        return msg

    @staticmethod
    def to_png_image(numpy_image: np.ndarray) -> CompressedImage:
        """
        Convert numpy.ndarray to sensor_msgs.msg.CompressedImage
        """
        msg = CompressedImage()
        msg.format = "png"
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        ret, data = cv2.imencode('.png', numpy_image, encode_param)
        msg.data = data.tobytes()
        if ret == False:
            raise ValueError("Failed to encode image")
        return msg


def numpify(msg):
    converter = ToNumpyConverter()
    if isinstance(msg, CompressedImage):
        return converter.from_compressed_image(msg)
    elif isinstance(msg, Image):
        return converter.from_image(msg)
    else:
        return None


def msgify(numpy_image, compress_type=''):
    converter = ToRosMsgConverter()
    if compress_type == '':
        return converter.to_image(numpy_image)
    elif compress_type == 'jpeg':
        return converter.to_jpeg_image(numpy_image)
    elif compress_type == 'png':
        return converter.to_png_image(numpy_image)
    else:
        return None
