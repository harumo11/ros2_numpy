
import unittest
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
import cv2
import sys
import os

# Ensure we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ros2_numpy import numpify, msgfy, pilfy

class TestImageConversion(unittest.TestCase):
    def test_numpify_image_rgb8(self):
        # Create a dummy Image message
        msg = Image()
        msg.height = 10
        msg.width = 10
        msg.encoding = "rgb8"
        msg.step = 30
        expected_data = np.zeros((10, 10, 3), dtype=np.uint8)
        # Fill with some data
        expected_data[:] = 100
        msg.data = expected_data.tobytes()

        result = numpify(msg)
        np.testing.assert_array_equal(result, expected_data)

    def test_numpify_image_mono8(self):
        msg = Image()
        msg.height = 10
        msg.width = 10
        msg.encoding = "mono8"
        msg.step = 10
        expected_data = np.zeros((10, 10), dtype=np.uint8)
        expected_data[:] = 50
        msg.data = expected_data.tobytes()

        result = numpify(msg)
        np.testing.assert_array_equal(result, expected_data)

    def test_msgfy_image_rgb8(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[:] = 200
        
        msg = msgfy(data, encoding="rgb8")
        self.assertEqual(msg.height, 10)
        self.assertEqual(msg.width, 10)
        self.assertEqual(msg.encoding, "rgb8")
        self.assertEqual(msg.step, 30)
        np.testing.assert_array_equal(np.frombuffer(msg.data, dtype=np.uint8).reshape(10, 10, 3), data)

    def test_msgfy_image_auto_encoding(self):
        # Test auto encoding detection
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        msg = msgfy(data)
        self.assertEqual(msg.encoding, "rgb8") # Default for 3 channels (impl detail)
        
        data_mono = np.zeros((10, 10), dtype=np.uint8)
        msg_mono = msgfy(data_mono)
        self.assertEqual(msg_mono.encoding, "mono8") # 1 channel

    def test_compressed_image(self):
        # Create a dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 0, 0), -1)
        
        # Encode to compressed message
        msg = msgfy(img, compress_type='jpeg')
        self.assertIsInstance(msg, CompressedImage)
        self.assertEqual(msg.format, 'jpeg')
        
        # Decode back
        result = numpify(msg)
        # Note: JPEG compression is lossy, so exact match is not expected, but shape should be same
        self.assertEqual(result.shape, img.shape)
        
    def test_pilfy(self):
        msg = Image()
        msg.height = 10
        msg.width = 10
        msg.encoding = "rgb8"
        msg.step = 30
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[:] = 255
        msg.data = data.tobytes()
        
        pil_img = pilfy(msg)
        self.assertEqual(pil_img.size, (10, 10))
        self.assertEqual(pil_img.mode, "RGB")

if __name__ == '__main__':
    unittest.main()
