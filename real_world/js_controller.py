from evdev import InputDevice, categorize, ecodes
from xarm.wrapper import XArmAPI


class xArm7GripperEnv:
    def __init__(self, robot_ip="130.245.125.30", arm_speed=1000, gripper_speed=1000, step_size=10, grip_size=100):
        self.robot_ip = robot_ip
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.step_size = step_size
        self.grip_size = grip_size
        self.arm: XArmAPI = None
        self.arm_pos: tuple = None
        self.arm_rot: tuple = None
        self.gripper_pos: int = None
        self.gripper_pos_counter: int = 0
        self.wait = False

    def update_arm_state(self):
        _, arm_pos = self.arm.get_position(is_radian=False)
        self.arm_pos = tuple(arm_pos[:3])
        self.arm_rot = tuple(arm_pos[3:])
        _, gripper_pos = self.arm.get_gripper_position()
        self.gripper_pos = gripper_pos
        self.gripper_pos_counter = gripper_pos

    def __enter__(self):
        arm = XArmAPI(self.robot_ip)
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        code = arm.set_gripper_mode(0)
        print("set gripper mode: location mode, code={}".format(code))
        code = arm.set_gripper_enable(True)
        print("set gripper enable, code={}".format(code))
        code = arm.set_gripper_speed(self.gripper_speed)
        print("set gripper speed, code={}".format(code))
        self.arm = arm
        self.update_arm_state()
        print("Initilized")
        return self

    def __exit__(self, *arg, **kwargs):
        self.arm.disconnect()
        print("Disconnected")

    def x_plus(self):
        self.arm.set_position(x=self.step_size, relative=True, wait=self.wait, speed=self.arm_speed)

    def x_minus(self):
        self.arm.set_position(x=-self.step_size, relative=True, wait=self.wait, speed=self.arm_speed)

    def y_plus(self):
        self.arm.set_position(y=self.step_size, relative=True, wait=self.wait, speed=self.arm_speed)

    def y_minus(self):
        self.arm.set_position(y=-self.step_size, relative=True, wait=self.wait, speed=self.arm_speed)

    def z_plus(self):
        self.arm.set_position(z=self.step_size, relative=True, wait=self.wait, speed=self.arm_speed)

    def z_minus(self):
        self.arm.set_position(z=-self.step_size, relative=True, wait=self.wait, speed=self.arm_speed)

    def gripper_open(self):
        k = self.gripper_pos_counter
        delta = self.grip_size
        self.arm.set_gripper_position(min(k + delta, 850), wait=self.wait)
        self.gripper_pos_counter = min(self.gripper_pos_counter + delta, 850)

    def gripper_close(self):
        k = self.gripper_pos_counter
        delta = self.grip_size
        self.arm.set_gripper_position(max(k - delta, 0), wait=self.wait)
        self.gripper_pos_counter = max(self.gripper_pos_counter - delta, 0)

    def clean_errors(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)


class JSController(xArm7GripperEnv):
    def __init__(self, *args, input_device="/dev/input/event15", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = InputDevice(input_device)

    @staticmethod
    def parse_key(key):
        if isinstance(key, tuple):
            if "BTN_A" in key:
                key = "BTN_A"
            elif "BTN_B" in key:
                key = "BTN_B"
            elif "BTN_X" in key:
                key = "BTN_X"
            else:
                print(f"Unknown key: {key}")
        return key

    def controller_listen(self):
        for event in self.device.read_loop():
            if event.type == ecodes.EV_KEY:
                key = categorize(event).keycode
                key = self.parse_key(key)
                pressed = event.value == 1

                if pressed:
                    if key == "BTN_A":
                        print("z-axis plus")
                        self.z_plus()
                    elif key == "BTN_C":
                        print("z-axis minus")
                        self.z_minus()
                    elif key == "BTN_X":
                        print("Close gripper")
                        self.gripper_close()
                    elif key == "BTN_B":
                        print("Open gripper")
                        self.gripper_open()
                    elif key == "BTN_Z":
                        print("Robot State")
                        self.update_arm_state()
                        print(f"Pos: {self.arm_pos}")
                        print(f"Rot: {self.arm_rot}")
                        print(f"Gripper: {self.gripper_pos}")
                    elif key == "BTN_TR":
                        print("Clean Errors")
                        self.clean_errors()

            elif event.type == ecodes.EV_ABS:
                code = ecodes.ABS[event.code]
                value = event.value

                if code == "ABS_X":
                    dx = (value - 128) / 128.0  # Normalize to -1..1
                    if abs(dx) > 0.2:
                        if dx > 0:
                            print("x-axis plus")
                            self.y_plus()  # misaligned coord systems
                        else:
                            print("x-axis minus")
                            self.y_minus()  # misaligned coord systems

                elif code == "ABS_Y":
                    dy = (value - 128) / 128.0
                    if abs(dy) > 0.2:
                        if dy < 0:
                            print("y-axis plus")
                            self.x_plus()  # misaligned coord systems
                        else:
                            print("y-axis minus")
                            self.x_minus()  # misaligned coord systems

                # time.sleep(0.05)  # Reduce update rate


contorller_args = {
    "robot_ip": "130.245.125.30",
    "arm_speed": 1000,
    "gripper_speed": 1000,
    "step_size": 10,
    "grip_size": 100,
    "input_device": "/dev/input/event15",
}
while True:
    try:
        with JSController(**contorller_args) as arm:
            arm.controller_listen()
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
