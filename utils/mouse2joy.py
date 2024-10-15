import evdev
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from threading import Thread

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
found_path = ""
for device in devices:
   print(device.path, device.name, device.phys)
   if device.name == "Logitech MX Master 2S":
       found_path = device.path
       break

class VirtualJoytick(Node):
    def __init__(self):
        super().__init__('virtual_joystick')
        self.axes = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.buttons = [0,0,0,0,0,0,0,0,0,0,0]
        self.pub = self.create_publisher(Joy, '/joy', 10)
        self.tid = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        msg = Joy()
        msg.axes = self.axes
        msg.buttons = self.buttons
        self.pub.publish(msg)


if found_path:
    print("found mouse!")
    rclpy.init()
    vj = VirtualJoytick()

    def spin():
        rclpy.spin(vj)

    t = Thread(target=spin, args=())
    t.start()

    device = evdev.InputDevice(found_path)
    L1_pressed = False
    R1_pressed = False
    for event in device.read_loop():
        #print(event)
        if event.type == evdev.ecodes.EV_KEY:
            if event.code == 272: # left
                print(f"left {event.value} L1({L1_pressed}) R1({R1_pressed})") # 0 release 1 press 2 pressing
                v = 0.0
                if event.value == 1:
                    v = 0.5
                elif event.value == 2:
                    v = 0.7
                vj.axes[3] = v
            elif event.code == 273: # right
                print(f"right {event.value} L1({L1_pressed}) R1({R1_pressed})") # 0 release 1 press 2 pressing
                v = 0.0
                if event.value == 1:
                    v = -0.5
                elif event.value == 2:
                    v = -0.7
                vj.axes[3] = v
            elif event.code == 275: # L1
                print(f"L1 {event.value}") # 0 release 1 press 2 pressing
                L1_pressed = (event.value != 0)
                vj.buttons[4] = int(event.value != 0)
            elif event.code == 276: # R1
                print(f"R1 {event.value}") # 0 release 1 press 2 pressing
                R1_pressed = (event.value != 0)
                vj.buttons[5] = int(event.value != 0)
        elif event.type == evdev.ecodes.EV_REL:
            if event.code == 11: # up/down
                v = 0.0
                if event.value > 0: #(0 - 15)
                    print(f"up {event.value} L1({L1_pressed}) R1({R1_pressed})") # 0 ~ 255
                    v = 1.0
                else:
                    print(f"down {event.value} L1({L1_pressed}) R1({R1_pressed})") # 0 ~ -255
                    v = -1.0
                vj.axes[1] = v
    rclpy.shutdown()
