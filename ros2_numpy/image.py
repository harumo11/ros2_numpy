import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from PIL import Image as PILImage
from typing import Optional, Tuple, Dict, Union, Type

# Mapping from encoding name to (numpy dtype, number of channels)
NAME_TO_DTYPES: Dict[str, Tuple[Type, int]] = {
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

# Reverse mapping from (numpy dtype, number of channels) to encoding name
# Note: The first encoding in NAME_TO_DTYPES for a given key will be used.
DTYPES_TO_NAME: Dict[Tuple[Type, int], str] = {}
for name, (dtype, channels) in NAME_TO_DTYPES.items():
    if (dtype, channels) not in DTYPES_TO_NAME:
        DTYPES_TO_NAME[(dtype, channels)] = name


def detect_encoding(numpy_image: np.ndarray) -> str:
    """
    Detect the encoding of a numpy array based on its dtype and number of channels.

    Args:
        numpy_image (np.ndarray): Input image.

    Returns:
        str: Detected encoding name (e.g., 'rgb8', 'mono8').

    Raises:
        ValueError: If the combination of dtype and channels is not supported.
    """
    dtype = numpy_image.dtype.type
    
    if numpy_image.ndim == 2:
        channels = 1
    elif numpy_image.ndim == 3:
        channels = numpy_image.shape[2]
    else:
        raise ValueError(f"Unsupported number of dimensions: {numpy_image.ndim}. Only 2D or 3D arrays are supported.")

    try:
        return DTYPES_TO_NAME[(dtype, channels)]
    except KeyError:
        raise ValueError(f"Unsupported combination of dtype {dtype} and channels {channels}.")


def from_msg_to_cvimg(msg: Image) -> np.ndarray:
    """
    Convert sensor_msgs.msg.Image to numpy.ndarray.

    Args:
        msg (Image): ROS Image message.

    Returns:
        np.ndarray: Converted numpy array.
    """
    if msg.encoding not in NAME_TO_DTYPES:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    dtype_class, channels = NAME_TO_DTYPES[msg.encoding]
    dtype = np.dtype(dtype_class)

    numpy_image = np.frombuffer(msg.data, dtype=dtype)
    
    if channels == 1:
        shape = (msg.height, msg.width)
    else:
        shape = (msg.height, msg.width, channels)
        
    numpy_image = numpy_image.reshape(shape)

    return numpy_image


def from_compressed_msg_to_cvimg(msg: CompressedImage) -> np.ndarray:
    """
    Convert sensor_msgs.msg.CompressedImage to numpy.ndarray.

    Args:
        msg (CompressedImage): ROS CompressedImage message.

    Returns:
        np.ndarray: Converted numpy array (BGR8 usually for color images).
    """
    np_arr = np.frombuffer(msg.data, np.uint8)
    numpy_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if numpy_image is None:
        raise ValueError("Failed to decode compressed image")

    return numpy_image


def from_cvimg_to_raw_img_msg(
    numpy_image: np.ndarray, encoding: str = "", frame_id: str = "camera"
) -> Image:
    """
    Convert numpy.ndarray to sensor_msgs.msg.Image.

    Args:
        numpy_image (np.ndarray): Input image.
        encoding (str): Target encoding. If empty, it is detected automatically.
        frame_id (str): Frame ID for the header.

    Returns:
        Image: ROS Image message.
    """
    msg = Image()
    msg.height = numpy_image.shape[0]
    msg.width = numpy_image.shape[1]
    
    if encoding == "":
        msg.encoding = detect_encoding(numpy_image)
    else:
        msg.encoding = encoding
        
    # Recalculate channels to calculate step correctly
    if numpy_image.ndim == 2:
        channels = 1
    else:
        channels = numpy_image.shape[2]

    msg.is_bigendian = 0
    # step is the full row length in bytes
    msg.step = msg.width * numpy_image.itemsize * channels
    msg.data = numpy_image.tobytes()
    msg.header.frame_id = frame_id
    return msg


def from_cvimg_to_jpg_msg(
    numpy_image: np.ndarray, jpg_quality: int = 95
) -> CompressedImage:
    """
    Convert numpy.ndarray to sensor_msgs.msg.CompressedImage (JPEG).

    Args:
        numpy_image (np.ndarray): Input image.
        jpg_quality (int): JPEG quality (0-100).

    Returns:
        CompressedImage: ROS CompressedImage message.
    """
    if jpg_quality < 0 or jpg_quality > 100:
        raise ValueError("Quality must be between 0 and 100")

    msg = CompressedImage()
    msg.format = "jpeg"
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    ret, data = cv2.imencode(".jpg", numpy_image, encode_param)
    if not ret:
        raise ValueError("Failed to encode image")
        
    msg.data = data.tobytes()
    return msg


def from_cvimg_to_png_msg(
    numpy_image: np.ndarray, png_compression: int = 3
) -> CompressedImage:
    """
    Convert numpy.ndarray to sensor_msgs.msg.CompressedImage (PNG).

    Args:
        numpy_image (np.ndarray): Input image.
        png_compression (int): PNG compression level (0-9).

    Returns:
        CompressedImage: ROS CompressedImage message.
    """
    if png_compression < 0 or png_compression > 9:
        raise ValueError("Compression must be between 0 and 9")

    msg = CompressedImage()
    msg.format = "png"
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression]
    ret, data = cv2.imencode(".png", numpy_image, encode_param)
    if not ret:
        raise ValueError("Failed to encode image")

    msg.data = data.tobytes()
    return msg


def numpify(msg: Union[Image, CompressedImage]) -> np.ndarray:
    """
    Convert ROS Image or CompressedImage message to numpy array.

    Args:
        msg (Image | CompressedImage): Input ROS message.

    Returns:
        np.ndarray: Converted numpy array.
    """
    if isinstance(msg, CompressedImage):
        return from_compressed_msg_to_cvimg(msg)
    elif isinstance(msg, Image):
        return from_msg_to_cvimg(msg)
    else:
        raise ValueError(
            "Input must be a sensor_msgs.msg.Image or sensor_msgs.msg.CompressedImage"
        )


def pilfy(msg: Union[Image, CompressedImage]) -> PILImage.Image:
    """
    Convert ROS Image or CompressedImage message to PIL Image.

    Args:
        msg (Image | CompressedImage): Input ROS message.

    Returns:
        PIL.Image.Image: Converted PIL Image.
    """
    numpy_image = numpify(msg)
    # PIL expects RGB, but cv2 usually works with BGR. 
    # However, numpify(Image) preserves channel order of encoding (usually rgb8 or bgr8).
    # numpify(CompressedImage) uses cv2.imdecode which returns BGR.
    
    # If it is BGR, we might need to swap.
    # But for now, let's assume user handles color space or it's just raw data.
    # Exception: if we want to be smart about it.
    # Current implementation just does Image.fromarray(numpy_image).
    return PILImage.fromarray(numpy_image)


def msgfy(image: Union[np.ndarray, PILImage.Image], compress_type: str = '', **kwargs) -> Union[Image, CompressedImage]:
    """
    Convert numpy array or PIL Image to ROS Image or CompressedImage message.

    Args:
        image (np.ndarray | PIL.Image): Input image.
        compress_type (str): Compression type ('', 'jpeg', 'jpg', 'png'). Default is '' (raw).
        **kwargs: Additional arguments for compression (jpg_quality, png_compression) or raw encoding.

    Returns:
        Image | CompressedImage: Converted ROS message.
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
    if compress_type == '':
        return from_cvimg_to_raw_img_msg(numpy_image, **kwargs)
    elif compress_type in ('jpeg', 'jpg'):
        return from_cvimg_to_jpg_msg(numpy_image, **kwargs)
    elif compress_type == 'png':
        return from_cvimg_to_png_msg(numpy_image, **kwargs)
    else:
        raise ValueError(
            f"Unsupported compress_type '{compress_type}'. Supported types are '', 'jpeg', 'jpg', and 'png'."
        )
