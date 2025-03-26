import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
import time

class HandGestureControl(Node):
    def __init__(self):
        super().__init__('hand_gesture_control')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.camera_loop) 
        self.rate = self.create_rate(10)
        
        self.hands = mp.solutions.hands
        self.hand_obj = self.hands.Hands(max_num_hands=1)
        self.drawing = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        self.prev = -1
        self.start_init = False
        self.start_time = time.time()

    def count_fingers(self, lst):
        cnt = 0
        thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

        if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
            cnt += 1
        if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
            cnt += 1
        if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
            cnt += 1
        if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
            cnt += 1
        if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
            cnt += 1
        return cnt

    def publish_cmd_vel(self, count):
        twist = Twist()
        if count == 1:
            twist.linear.x = 0.5  #forward
        elif count == 2:
            twist.angular.z = 0.5  #left
        elif count == 3:
            twist.angular.z = -0.5  #right
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        self.publisher.publish(twist)

    def camera_loop(self):
        end_time = time.time()
        ret, frm = self.cap.read()
        if not ret:
            return
            
        frm = cv2.flip(frm, 1)
        frame_height, frame_width, _ = frm.shape

        cv2.line(frm, (frame_width//3, 0), (frame_width//3, frame_height), (0,255,0), 2)
        cv2.line(frm, (2*frame_width//3, 0), (2*frame_width//3, frame_height), (0,255,0), 2)

        res = self.hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        cnt = 0
        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]
            cnt = self.count_fingers(hand_keyPoints)

            if cnt == 0:  
                hand_x = int(hand_keyPoints.landmark[0].x * frame_width)
                seg_width = frame_width // 3
                twist = Twist()
                
                if hand_x < seg_width:
                    twist.angular.z = 0.5 
                elif hand_x > 2*seg_width:
                    twist.angular.z = -0.5  
                self.publisher.publish(twist)
                self.prev = cnt
            else:
                if self.prev != cnt:
                    if not self.start_init:
                        self.start_time = time.time()
                        self.start_init = True
                    elif (end_time - self.start_time) > 0.2:
                        self.publish_cmd_vel(cnt)
                        self.prev = cnt
                        self.start_init = False

            self.drawing.draw_landmarks(frm, hand_keyPoints, self.hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Gesture Control', frm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    controller = HandGestureControl()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.cap.release()
        cv2.destroyAllWindows()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
