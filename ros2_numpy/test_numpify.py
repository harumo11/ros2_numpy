from ros2_numpy import numpify, msgfy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
import rclpy
import cv2 as cv


class ImageListener(Node):
    def __init__(self):
        super().__init__("image_listener")
        # self.subscription = self.create_subscription(Image, '/image_raw/compressed', self.listener_callback, 10)
        self.subscription = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.listener_callback_compressed,
            10,
        )
        print("Subscribed to /image_raw/compressed")
        self.publisher = self.create_publisher(
            CompressedImage, "/my/image_raw/compressed", 10
        )
        # self.publisher = self.create_publisher(Image, '/my/image_raw', 10)
        self.timer = self.create_timer(1, self.timer_callback)
        self.image = None

    def listener_callback_compressed(self, msg: CompressedImage):
        print("Received image")
        numpy_image = numpify(msg)
        self.image = numpy_image
        if numpy_image is None:
            print("Failed to convert image")
            return
        cv.imshow("image", numpy_image)
        cv.waitKey(1)

    def timer_callback(self):
        if self.image is None:
            return
        cv.imshow("self image", self.image)
        cv.waitKey(1)
        # msg = msgify(self.image)
        msg = msgfy(self.image, compress_type="png")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)
        print("Published image")


def main(args=None):
    rclpy.init(args=args)
    image_listener = ImageListener()
    rclpy.spin(image_listener)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
