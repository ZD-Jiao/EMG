import sys
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import joblib
from scipy.signal import welch
import argparse
import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
import UDF_Gesture

# === 配置参数 ===
channel_num = 4
# sample_rate = 250   # 默认为250 Hz，无法修改
ser_port = '/dev/ttyUSB0'
max_enabled = 1             # 是否有显示数据点数量的限制
max_points = 1000           # 最多显示数据点的数量
update_interval_ms = 150     # 更新间隔，单位：毫秒
predict_interval = 1        # 刷新5次预测一次
display_enabled = 1         # turn on/off Plotter
window_size = 100           # should be same as 2_train.py
model = joblib.load('emg_rf_model.pkl')

# shadow hand initialization
rospy.init_node("robot_commander_examples", anonymous = True)
hand_commander = SrHandCommander(name="right_hand")

# 设置速度比例（范围 0.0 ~ 1.0，1.0 表示最大速度）
hand_commander.set_max_velocity_scaling_factor(1.0)  # 设置为 20% 速度
hand_commander.set_max_acceleration_scaling_factor(1.0)  # 设置加速度比例

class EMGPlotter(QtWidgets.QMainWindow):
    def __init__(self, board_id=0, serial_port='/dev/ttyUSB0', channel_num=4,
                 max_enabled=0, max_points=1000, update_interval_ms=50, 
                 predict_interval=1, window_size=150, display_enabled = 1):
        super().__init__()
        
        # if self.display_enabled:
        self.setWindowTitle("Real Time EMG Signals + Gesture Prediction")
        self.resize(1000, 600)

        # 初始化参数
        self.board_id = board_id
        self.chan_num = channel_num
        self.update_interval_ms = update_interval_ms
        self.sample_rate = BoardShim.get_sampling_rate(self.board_id)
        self.time_buffer = []
        self.data_buffer = [[] for _ in range(channel_num)]
        self.start_time = time.time()
        self.flag = 1
        self.last_time = 0
        self.max_enabled = max_enabled
        self.max_points = max_points
        self.predict_interval = predict_interval
        self.window_size = window_size
        self.flag_pred = 0
        self.pred_buffer = ['rest']*5   #
        self.display_enabled = display_enabled
        self.setFocusPolicy(QtCore.Qt.StrongFocus)     # 确保主窗口接收按键事件

        # 设置绘图窗口
        if self.display_enabled:
            self.plot_widget = pg.PlotWidget(title="EMG RT Curves")
            self.setCentralWidget(self.plot_widget)
            self.plot_widget.setLabel('left', 'Volt (μV)')
            self.plot_widget.setLabel('bottom', 'Time (s)')
            self.plot_widget.addLegend()
            self.plot_widget.showGrid(x=True, y=True)

            self.curves = []
            colors = ['r', 'g', 'b', 'y']
            for i in range(self.chan_num):
                curve = self.plot_widget.plot(pen=pg.mkPen(colors[i % len(colors)], width=1),
                                            name=f'Channel {i+1}')
                self.curves.append(curve)

        # 初始化 BrainFlow
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        self.board.start_stream()

        # 预测输出框
        if self.display_enabled:
            self.label = QtWidgets.QLabel("预测手势: ", self)
            self.label.setGeometry(20, 10, 300, 30)

        # 定时器
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval_ms)

        # 数据保存
        timestamp = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
        self.output_file = f'./result-{timestamp}.txt'
        with open(self.output_file, 'w') as f:
            header = 'Time(s) ' + ' '.join([f'Ch{i+1}(μV)' for i in range(self.chan_num)]) + '\n'
            f.write(header)

    def update_plot(self):
        if self.flag == 1:
            data = self.board.get_board_data()
            self.last_time = time.time() - self.start_time
            self.flag = 0
        else:
            # data = self.board.get_board_data()    # 容易获取空数据，导致曲线出现不连续
            window_size = int(self.sample_rate * self.update_interval_ms / 1000)
            data = self.board.get_current_board_data(window_size)
            self.flag_pred += 1
            if data.shape[1] == 0:
                return

            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            emg_data = data[eeg_channels[:self.chan_num], :]
            num_samples = emg_data.shape[1]

            # update time list
            current_time = time.time() - self.start_time
            temp_x = [self.last_time + j * (current_time - self.last_time)/num_samples for j in range(num_samples)]
            self.time_buffer += temp_x
            if self.max_enabled:
                while len(self.time_buffer) >= self.max_points:
                    self.time_buffer.pop(0)
            self.last_time = current_time

            # update data list
            for i in range(self.chan_num):
                self.data_buffer[i] += list(emg_data[i])
                # self.data_pred_buffer[i] += list(emg_data[i])
                if self.max_enabled:
                    while len(self.data_buffer[i]) >= self.max_points:
                        self.data_buffer[i].pop(0)

            # update curve list
            if self.display_enabled:
                for i in range(self.chan_num):
                    self.curves[i].setData(self.time_buffer, self.data_buffer[i])

            # 写入 txt
            with open(self.output_file, 'a') as f:
                for j in range(num_samples):
                    line = f"{temp_x[j]:.3f} " + ' '.join([f"{emg_data[i][j]:.2f}" for i in range(self.chan_num)]) + '\n'
                    f.write(line)

            # --- 手势预测 ---
            if len(self.data_buffer[1]) >= self.window_size & self.flag_pred >= self.predict_interval:
                self.flag_pred = 0
                # 使用最近的100个数据点做预测（对应训练时 window_size=100）
                window = np.array(self.data_buffer)[:, -1*self.window_size:]

                features = []

                for ch in window:
                    # --- 时域特征 ---
                    features += [
                        np.mean(ch),
                        np.std(ch),
                        np.max(ch),
                        np.min(ch),
                        np.median(ch),
                        np.sum(np.abs(ch)),  # MAV
                        np.sqrt(np.mean(ch ** 2)),  # RMS
                    ]

                    # --- 频域特征 ---
                    freqs, psd = welch(ch, fs=self.sample_rate, nperseg=len(ch))
                    total_power = np.trapz(psd, freqs)  # 面积积分（频谱能量）
                    mean_freq = np.sum(freqs * psd) / np.sum(psd)
                    median_freq = freqs[np.where(np.cumsum(psd) >= np.sum(psd) / 2)[0][0]]
                    peak_freq = freqs[np.argmax(psd)]

                    features += [
                        mean_freq,
                        median_freq,
                        total_power,
                        peak_freq
                    ]

                X = np.array(features).reshape(1, -1)
                try:
                    pred = model.predict(X)[0]
                    self.label.setText(f"Predicted gesture: {pred}")
                    print(f"[Prediction] {pred}")
                    self.pred_buffer.append(pred)
                    # modify four numbers
                    repeat_num = 2
                    if self.pred_buffer[-1*repeat_num:] == ['fist'] * repeat_num:
                        joints_states = UDF_Gesture.fist
                        hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
                        hand_commander.move_to_joint_value_target(joints_states, wait=True, angle_degrees=True)
                    if self.pred_buffer[-1*repeat_num:] == ['rest'] * repeat_num:
                        joints_states = UDF_Gesture.open
                        hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
                        hand_commander.move_to_joint_value_target(joints_states, wait=True, angle_degrees=True)
                except Exception as e:
                    print(f"Prediction Error: {e}")

    def keyPressEvent(self, event):
        key = event.key()
        # print(f"Key pressed: {key} ({event.text()})")
        if event.text().lower() == '1':
            joints_states = UDF_Gesture.open
            hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
            hand_commander.move_to_joint_value_target(joints_states, wait=True, angle_degrees=True)
        if event.text().lower() == '2':
            joints_states = UDF_Gesture.fist
            hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
            hand_commander.move_to_joint_value_target(joints_states, wait=True, angle_degrees=True)
        if event.text().lower() == '3':
            joints_states = UDF_Gesture.six
            hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
            hand_commander.move_to_joint_value_target(joints_states, wait=True, angle_degrees=True)
        if event.text().lower() == '4':
            joints_states = UDF_Gesture.ok
            hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
            hand_commander.move_to_joint_value_target(joints_states, wait=True, angle_degrees=True)

        # 可选：你可以根据按键执行特定操作
        if event.text().lower() == 'q':
            print("Q pressed: quitting application")
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        self.board.stop_stream()
        self.board.release_session()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # window = EMGPlotter()
    window_EMG = EMGPlotter(serial_port=ser_port, \
                        channel_num=channel_num, \
                        max_enabled=max_enabled, \
                        max_points=max_points, \
                        predict_interval=predict_interval, \
                        display_enabled=display_enabled, \
                        update_interval_ms=update_interval_ms)
    window_EMG.show()
    window_EMG.setFocus()
    sys.exit(app.exec_())
