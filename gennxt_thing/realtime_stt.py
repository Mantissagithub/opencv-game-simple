import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from RealtimeSTT import AudioToTextRecorder


class RealTimeSpeechToText(Node):
    def __init__(self):
        super().__init__("RealTimeSpeechToText")
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.recorder = AudioToTextRecorder()

        self.recorder.start()
        self.get_logger().info("Listening for commands...")
        
        self.last_command = ""  

        try:
            while rclpy.ok():
                recorded_text = self.recorder.text()
                if recorded_text and recorded_text != self.last_command:
                    self.get_logger().info(f"Transcription: {recorded_text}")
                    self.process_transcription(recorded_text)
                    self.last_command = recorded_text 
        except KeyboardInterrupt:
            self.get_logger().info("Stopping real-time speech-to-text...")
        finally:
            self.recorder.stop()

    def process_transcription(self, text):
        twist = Twist()
        if "forward" in text.lower():
            self.get_logger().info("MOving forward")
            twist.linear.x = 1.0 
            twist.angular.z = 0.0
        elif "backward" in text.lower():
            self.get_logger().info("MOving backward")
            twist.linear.x = -1.0  
            twist.angular.z = 0.0
        elif "left" in text.lower():
            self.get_logger().info("MOving left")
            twist.linear.x = 0.0
            twist.angular.z = 0.5 
        elif "right" in text.lower():
            self.get_logger().info("MOving right")
            twist.linear.x = 0.0
            twist.angular.z = -0.5 
        else:
            twist.linear.x = 0.0 
            twist.angular.z = 0.0

        self.publisher.publish(twist)
        self.get_logger().info(f"Published Twist: linear.x={twist.linear.x}, angular.z={twist.angular.z}")


def main(args=None):
    rclpy.init(args=args)
    controller = RealTimeSpeechToText()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
