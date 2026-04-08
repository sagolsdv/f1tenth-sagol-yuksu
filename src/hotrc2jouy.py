#!/usr/bin/env python3
import gpiod
import time
from datetime import timedelta

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from threading import Thread

CHIP_NAME="tegra234-gpio"
CHIP_NAME = "gpiochip0"

STEER_LINE   = 43
THROTTLE_LINE = 53
DEAD_LINE    = 124

def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

def norm(pw_us, center, span=500.0):
    return clamp((pw_us - center) / span)

# -----------------------
# chip + line request
# -----------------------
chip = gpiod.chip(CHIP_NAME)

steer_line  = chip.get_line(STEER_LINE)
throt_line  = chip.get_line(THROTTLE_LINE)
dead_line   = chip.get_line(DEAD_LINE)

cfg = gpiod.line_request()
cfg.consumer = "hotrc"
cfg.request_type = gpiod.line_request.EVENT_BOTH_EDGES

steer_line.request(cfg)
throt_line.request(cfg)
dead_line.request(cfg)

last_rise_ns = {"steer": None, "throt": None, "dead": None}
pulse_us = {
    "steer": 1500.0,
    "throt": 1500.0,
    "dead":  1000.0,
}

DEAD_THRESHOLD = 1500.0  # 1000(released) ~ 2000(pressed)

print("PyPI gpiod==1.5.4 RC Reader (CH1/CH2/CH3 Deadman)")
print("CTRL+C to exit\n")

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

rclpy.init()
vj = VirtualJoytick()

def spin():
    rclpy.spin(vj)

th = Thread(target=spin, args=())
th.start()

try:
    # 1ms non-blocking polling
    to = timedelta(milliseconds=1)

    while True:
        # ---- steering ----
        if steer_line.event_wait(to):
            ev = steer_line.event_read()

            t = ev.timestamp
            if isinstance(t, int):
                t_ns = t
            else:
                t_ns = int(t.timestamp() * 1_000_000_000)

            if ev.event_type == gpiod.line_event.RISING_EDGE:
                last_rise_ns["steer"] = t_ns
            elif ev.event_type == gpiod.line_event.FALLING_EDGE and last_rise_ns["steer"] is not None:
                dt = (t_ns - last_rise_ns["steer"]) / 1000.0
                if 800 <= dt <= 2200:
                    pulse_us["steer"] = dt

        # ---- throttle ----
        if throt_line.event_wait(to):
            ev = throt_line.event_read()
            t = ev.timestamp
            if isinstance(t, int):
                t_ns = t
            else:
                t_ns = int(t.timestamp() * 1_000_000_000)

            if ev.event_type == gpiod.line_event.RISING_EDGE:
                last_rise_ns["throt"] = t_ns
            elif ev.event_type == gpiod.line_event.FALLING_EDGE and last_rise_ns["throt"] is not None:
                dt = (t_ns - last_rise_ns["throt"]) / 1000.0
                if 800 <= dt <= 2200:
                    pulse_us["throt"] = dt

        # ---- deadman ----
        if dead_line.event_wait(to):
            ev = dead_line.event_read()
            t = ev.timestamp
            if isinstance(t, int):
                t_ns = t
            else:
                t_ns = int(t.timestamp() * 1_000_000_000)

            if ev.event_type == gpiod.line_event.RISING_EDGE:
                last_rise_ns["dead"] = t_ns
            elif ev.event_type == gpiod.line_event.FALLING_EDGE and last_rise_ns["dead"] is not None:
                dt = (t_ns - last_rise_ns["dead"]) / 1000.0
                if 800 <= dt <= 2200:
                    pulse_us["dead"] = dt

        # ---- regulation & deadman ----
        steer_norm      = norm(pulse_us["steer"],  center=1495.0)
        throt_norm_raw  = norm(pulse_us["throt"], center=1485.0)
        deadman_ok      = pulse_us["dead"] > DEAD_THRESHOLD

        vj.axes[3] = -steer_norm       # steering
        vj.axes[1] = throt_norm_raw    # throttle
        vj.buttons[4] = not deadman_ok # L1

        vj.buttons[5] = deadman_ok     # R2 (deadman)

        # throttle kill if deadman is not pressed
        throttle_cmd = throt_norm_raw if deadman_ok else 0.0

        print(
            f"steer={steer_norm:+.2f} ({pulse_us['steer']:4.0f}us) | "
            f"throt_raw={throt_norm_raw:+.2f} ({pulse_us['throt']:4.0f}us) | "
            f"dead={pulse_us['dead']:4.0f}us -> {deadman_ok} | "
            f"throt_cmd={throttle_cmd:+.2f}      ",
            end="\r",
            flush=True,
        )

        time.sleep(0.001)

except KeyboardInterrupt:
    pass
finally:
    steer_line.release()
    throt_line.release()
    dead_line.release()
    print("\nend")


