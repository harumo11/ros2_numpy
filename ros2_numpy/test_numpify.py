from ros2_numpy import numpify
from sensor_msgs.msg import Image
import numpy as np
from rclpy.node import Node
import rclpy
import cv2 as cv


class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.listener_callback, 10)
        self.subscription

    def listener_callback(self, msg: Image):
        numpy_image = numpify(msg)
        print(numpy_image.shape)
        print(f'encoding: {msg.encoding}')
        cv.imshow('image', numpy_image)
        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    image_listener = ImageListener()
    rclpy.spin(image_listener)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
