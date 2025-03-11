from ros2_numpy import numpify
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import numpy as np
from rclpy.node import Node
import rclpy
import cv2 as cv


class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        #self.subscription = self.create_subscription(Image, '/image_raw/compressed', self.listener_callback, 10)
        self.subscription = self.create_subscription(CompressedImage, '/image_raw/compressed', self.listener_callback, 10)
        print('Subscribed to /image_raw/compressed')

    def listener_callback(self, msg: Image):
        print('Received image')
        numpy_image = numpify(msg)
        if numpy_image is None:
            print('Failed to convert image')
            return
        cv.imshow('image', numpy_image)
        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    image_listener = ImageListener()
    rclpy.spin(image_listener)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
