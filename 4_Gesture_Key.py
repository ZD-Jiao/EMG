import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt

import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
import UDF_Gesture



class HandControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Controller")
        self.setGeometry(100, 100, 400, 300)

        self.label = QLabel("1: Open<br>2: Fist<br>3: Six<br>4: OK", self)
        self.label.setGeometry(50, 40, 300, 200)

        # shadow hand initialization
        rospy.init_node("robot_commander_qt", anonymous=True)
        self.hand_commander = SrHandCommander(name="right_hand")

        # 设置速度比例（范围 0.0 ~ 1.0，1.0 表示最大速度）
        self.hand_commander.set_max_velocity_scaling_factor(1.0)  # 设置为 20% 速度
        self.hand_commander.set_max_acceleration_scaling_factor(1.0)  # 设置加速度比例

    def keyPressEvent(self, event):
        if event.text().lower() == '1':
            joints_states = UDF_Gesture.open
            # self.hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
            self.hand_commander.move_to_joint_value_target(joints_states, wait=False, angle_degrees=True)
        elif event.text().lower() == '2':
            joints_states = UDF_Gesture.fist
            # self.hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
            self.hand_commander.move_to_joint_value_target(joints_states, wait=False, angle_degrees=True)
        elif event.text().lower() == '3':
            joints_states = UDF_Gesture.six
            # self.hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
            self.hand_commander.move_to_joint_value_target(joints_states, wait=False, angle_degrees=True)
        elif event.text().lower() == '4':
            joints_states = UDF_Gesture.ok
            # self.hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
            self.hand_commander.move_to_joint_value_target(joints_states, wait=False, angle_degrees=True)
        else:
            self.label.setText(f"Pressed key: {event.text()} (no action)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandControlWindow()
    window.show()
    sys.exit(app.exec_())
