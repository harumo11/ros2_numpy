# this script is a ROS2 node that listens to the /image_raw topic and converts the image to a numpy array

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import dataclasses
from PIL import Image as PILImage


@dataclasses.dataclass
class ImageTypes:
    """
    A converter class to transform ROS Image and CompressedImage messages to numpy arrays.
    """

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


def from_msg_to_cvimg(msg: Image):
    """
    Convert sensor_msgs.msg.Image to numpy.ndarray
    """
    # get dtype and channel from encoding
    if msg.encoding in ImageTypes.name_to_dtypes:
        dtype, channel = ImageTypes.name_to_dtypes[msg.encoding]
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    # convert to numpy.ndarray
    numpy_image = np.frombuffer(msg.data, dtype=dtype)
    numpy_image = numpy_image.reshape((msg.height, msg.width, channel))

    return numpy_image


def from_compressed_msg_to_cvimg(msg: CompressedImage):
    """
    Convert sensor_msgs.msg.CompressedImage to numpy.ndarray
    """
    np_arr = np.frombuffer(msg.data, np.uint8)
    numpy_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return numpy_image


def from_cvimg_to_raw_img_msg(
    numpy_image: np.ndarray, encoding: str = "bgr8", frame_id: str = "camera"
) -> Image:
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
    msg.header.frame_id = frame_id
    return msg


def from_cvimg_to_jpg_msg(
    numpy_image: np.ndarray, jpg_quality: int = 95
) -> CompressedImage:
    """
    Convert numpy.ndarray to sensor_msgs.msg.CompressedImage
    cv2.IMWRITE_JPEG_QUALITY: 0-100. A higher value means a higher quality and larger size.
    """
    if jpg_quality < 0 or jpg_quality > 100:
        raise ValueError("Quality must be between 0 and 100")

    msg = CompressedImage()
    msg.format = "jpeg"
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    ret, data = cv2.imencode(".jpg", numpy_image, encode_param)
    msg.data = data.tobytes()
    if ret is False:
        raise ValueError("Failed to encode image")
    return msg


def from_cvimg_to_png_msg(
    numpy_image: np.ndarray, png_compression: int = 3
) -> CompressedImage:
    """
    Convert numpy.ndarray to sensor_msgs.msg.CompressedImage
    cv2.IMWRITE_PNG_COMPRESSION: 0-9. A higher value means a smaller size and longer compression time.
    """
    if png_compression < 0 or png_compression > 9:
        raise ValueError("Compression must be between 0 and 9")

    msg = CompressedImage()
    msg.format = "png"
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression]
    ret, data = cv2.imencode(".png", numpy_image, encode_param)
    msg.data = data.tobytes()
    if ret is False:
        raise ValueError("Failed to encode image")
    return msg


def numpify(msg):
    """
    Convert ROS Image or CompressedImage message to numpy array.

    Parameters
    ----------
    msg : sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage
        The input ROS message to convert.

    Returns
    -------
    numpy.ndarray

    """
    if isinstance(msg, CompressedImage):
        return from_compressed_msg_to_cvimg(msg)
    elif isinstance(msg, Image):
        return from_msg_to_cvimg(msg)
    else:
        raise ValueError(
            "Input must be a sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage"
        )


def pilfy(msg):
    """
    Convert ROS Image or CompressedImage message to PIL Image.

    Parameters
    ----------
    msg : sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage
        The input ROS message to convert.

    Returns
    -------
    PIL.Image.Image

    """
    numpy_image = numpify(msg)
    return PILImage.fromarray(numpy_image)


def msgfy(image, compress_type='', **kwargs):
    """
    Convert numpy array to ROS Image or CompressedImage message.

    Parameters
    ----------
    image : numpy.ndarray or PIL.Image
        The input image to convert.
    compress_type : str, optional
        The type of compression to use. Options are 'jpeg', 'png', or '' for raw image.
        Default is ''.
    jpg_quality : int, optional
        The quality of the JPEG compression. Must be between 0 and 100.
        Default is 95.
    png_compression : int, optional
        The compression level for PNG. Must be between 0 and 9.
        Default is 3.

    Returns
    -------
    sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage
        The converted ROS message.
    """
    # check if image is a numpy or PIL Image
    if isinstance(image, PILImage.Image):
        numpy_image = np.array(image)
    elif isinstance(image, np.ndarray):
        numpy_image = image
    else:
        raise ValueError(
            f"Input must be a numpy array or PIL Image. Got {type(image)} instead."
        )

    # make a image message
    if compress_type == "":
        return from_cvimg_to_raw_img_msg(numpy_image)
    elif compress_type == "jpeg" or compress_type == "jpg":
        return from_cvimg_to_jpg_msg(numpy_image, **kwargs)
    elif compress_type == "png":
        return from_cvimg_to_png_msg(numpy_image, **kwargs)
    else:
        raise ValueError(
            f"Unsupported compress_type '{compress_type}'. Supported types are '', 'jpeg', 'jpg', and 'png'."
        )
