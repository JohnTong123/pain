import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class YellowLineDetector(Node):
    def __init__(self):
        super().__init__('yellow_line_detector')
        self.declare_parameter('image_topic', '/D435i/color/image_raw')
        self.declare_parameter('min_area', 200)
        topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.min_area = int(self.get_parameter('min_area').value)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, topic, self.image_cb, 10)
        self.pub_overlay = self.create_publisher(Image, '/yellow_line/image', 10)
        self.pub_mask = self.create_publisher(Image, '/yellow_line/mask', 10)
        self.get_logger().info(f'Subscribed to {topic}, publishing /yellow_line/image and /yellow_line/mask')

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h = frame.shape[0]
        split = h // 2

        bottom = frame[split:, :]
        hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 100, 100])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        overlay = frame.copy()
        # Dim the top half so bottom stands out
        overlay[:split] = (overlay[:split] * 0.5).astype(np.uint8)
        # Draw the boundary between halves
        cv2.line(overlay, (0, split), (overlay.shape[1], split), (255, 255, 255), 1)

        bottom_overlay = overlay[split:]
        # Tint yellow pixels bright magenta for visibility
        tint = np.zeros_like(bottom_overlay)
        tint[:] = (255, 0, 255)
        alpha = 0.5
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
        bottom_overlay[mask3] = (
            (1 - alpha) * bottom_overlay[mask3] + alpha * tint[mask3]
        ).astype(np.uint8)

        # Contours + bounding boxes for the largest yellow blobs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            detections += 1
            cv2.drawContours(bottom_overlay, [c], -1, (0, 255, 255), 2)
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(bottom_overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawMarker(bottom_overlay, (cx, cy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
            label = f'yellow: x={cx} y={cy} area={int(M["m00"])} blobs={detections}'
        else:
            label = 'no yellow detected'

        overlay[split:] = bottom_overlay
        cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        out = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        out.header = msg.header
        self.pub_overlay.publish(out)

        mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask_full[split:] = mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask_full, encoding='mono8')
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)


def main():
    rclpy.init()
    node = YellowLineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
