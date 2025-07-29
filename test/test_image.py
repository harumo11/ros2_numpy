

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import pytest
import numpy as np
from sensor_msgs.msg import Image
import ros2_numpy as ros2_image


class ImagePublisher(Node):
    def __init__(self, topic, image):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, topic, 10)
        self.image = image
        self.timer = self.create_timer(0.5, self.publish_image)
        self.published = False

    def publish_image(self):
        if not self.published:
            self.get_logger().info(f'type of image: {type(self.image)}')
            msg = ros2_image.msgify(self.image, compress_type='')
            self.publisher_.publish(msg)
            self.published = True


class ImageSubscriber(Node):
    def __init__(self, topic):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.listener_callback,
            10)
        self.received_image = None

    def listener_callback(self, msg):
        self.received_image = ros2_image.numpify(msg)


def test_image_transport():
    rclpy.init()
    topic = 'test_image_topic'
    # Create a dummy image (e.g., 100x100 RGB)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pub_node = ImagePublisher(topic, image)
    sub_node = ImageSubscriber(topic)

    executor = SingleThreadedExecutor()
    executor.add_node(pub_node)
    executor.add_node(sub_node)

    try:
        for _ in range(20):  # Run for up to 10 seconds
            executor.spin_once(timeout_sec=0.5)
            if sub_node.received_image is not None:
                break
        assert sub_node.received_image is not None, "Image not received"
        # Check that the received image matches the sent image
        # Ensure received_image is a numpy array
        received_np = np.array(sub_node.received_image)
        assert np.array_equal(image, received_np), "Received image does not match sent image"
    finally:
        executor.remove_node(pub_node)
        executor.remove_node(sub_node)
        pub_node.destroy_node()
        sub_node.destroy_node()
        rclpy.shutdown()


# CompressedImage transport test
from sensor_msgs.msg import CompressedImage

class CompressedImagePublisher(Node):
    def __init__(self, topic, image):
        super().__init__('compressed_image_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, topic, 10)
        self.image = image
        self.timer = self.create_timer(0.5, self.publish_image)
        self.published = False
    def publish_image(self):
        if not self.published:
            msg = ros2_image.msgify(self.image, compress_type='png')
            self.publisher_.publish(msg)
            self.published = True

class CompressedImageSubscriber(Node):
    def __init__(self, topic):
        super().__init__('compressed_image_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            topic,
            self.listener_callback,
            10)
        self.received_image = None
    def listener_callback(self, msg):
        self.received_image = ros2_image.numpify(msg)

def test_compressed_image_transport():
    rclpy.init()
    topic = 'test_compressed_image_topic'
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pub_node = CompressedImagePublisher(topic, image)
    sub_node = CompressedImageSubscriber(topic)
    executor = SingleThreadedExecutor()
    executor.add_node(pub_node)
    executor.add_node(sub_node)
    try:
        for _ in range(20):
            executor.spin_once(timeout_sec=0.5)
            if sub_node.received_image is not None:
                break
        assert sub_node.received_image is not None, "CompressedImage not received"
        received_np = np.array(sub_node.received_image)
        assert np.array_equal(image, received_np), "Received CompressedImage does not match sent image"
    finally:
        executor.remove_node(pub_node)
        executor.remove_node(sub_node)
        pub_node.destroy_node()
        sub_node.destroy_node()
        rclpy.shutdown()
