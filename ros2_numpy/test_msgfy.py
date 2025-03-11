from ros2_numpy import numpify, msgify
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
import rclpy
import cv2 as cv


class ImageListener(Node):
    def __init__(self):
        super().__init__('image_capture')
        self.subscription = self.create_subscription(
            Image, '/my/image_raw', self.listener_callback, 10)
        print('Subscribed to /image_raw/compressed')

    def listener_callback(self, msg: Image):
        print('Received image')
        numpy_image = numpify(msg)
        self.image = numpy_image
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
