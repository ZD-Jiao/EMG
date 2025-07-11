import sys
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import joblib
from scipy.signal import welch
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, DetrendOperations
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
import argparse
import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
import UDF_Gesture

# === 配置参数 ===
channel_num = 14
start_ch = 0
channel_list = list(range(start_ch, start_ch + channel_num))
# channel_list = [0, 1, 2, 3, 64, 65, 66, 67]     # 实际的通道编号，用于 显示曲线
# sample_rate = 250   # 默认为250 Hz，无法修改
ser_port = '/dev/ttyUSB0'
max_enabled = 1             # 是否有显示数据点数量的限制
max_points = 500           # 最多显示数据点的数量
update_interval_ms = 150     # 更新间隔，单位：毫秒
board_id = 2                # CYTON_BOARD=0 (8 channels); CYTON_DAISY_BOARD=2 (16 channels)
predict_interval = 2        # 刷新5次预测一次
display_enabled = 1         # turn on/off Plotter; has some errors
window_size = 50           # should be same as 2_train.py
model = joblib.load('emg_rf_model.pkl')

# shadow hand initialization
rospy.init_node("robot_commander_examples", anonymous = True)
hand_commander = SrHandCommander(name="right_hand")

# 设置速度比例（范围 0.0 ~ 1.0，1.0 表示最大速度）
hand_commander.set_max_velocity_scaling_factor(1.0)  # 设置为 20% 速度
hand_commander.set_max_acceleration_scaling_factor(1.0)  # 设置加速度比例

class EMGPlotter(QtWidgets.QMainWindow):
    def __init__(self, board_id=0, serial_port='/dev/ttyUSB0', channel_list=list(range(0, 8)),
                 max_enabled=0, max_points=1000, update_interval_ms=50, 
                 predict_interval=1, window_size=150, display_enabled = 1):
        super().__init__()

        # 初始化参数
        self.board_id = board_id
        self.chan_num = channel_num
        self.update_interval_ms = update_interval_ms
        self.sample_rate = BoardShim.get_sampling_rate(self.board_id)
        self.time_buffer = []
        self.data_buffer = [[] for _ in range(channel_num)]
        self.data_buffer_filter = [[] for _ in range(channel_num)]  # N个通道滤波数据
        self.start_time = time.time()
        self.flag = 1
        self.last_time = 0
        self.last_time_save = 0
        self.max_enabled = max_enabled
        self.max_points = max_points
        self.predict_interval = predict_interval
        self.window_size = window_size
        self.flag_pred = 0
        self.pred_buffer = ['rest']*4   #
        self.display_enabled = display_enabled
        self.setFocusPolicy(QtCore.Qt.StrongFocus)     # 确保主窗口接收按键事件

        if self.display_enabled:
            # 设置绘图窗口
            self.setWindowTitle("Real Time EMG Signals + Gesture Prediction")
            height = 1080
            width = [800, 1000, 1200, 1400]
            self.resize(width[int((self.chan_num-1) // 8)], height) 

            # 设置绘图窗口
            self.setWindowTitle("Real Time EMG Signals + Gesture Prediction")
            height = 1080
            width = [800, 1000, 1200, 1400]
            self.resize(width[int((self.chan_num-1) // 8)], height) 
            
            self.win = pg.GraphicsLayoutWidget()
            self.setCentralWidget(self.win)

            self.curves = []
            self.plots = []

            for idx, ch in enumerate(channel_list):  # ch 是通道号
                col = idx // 4  # 每列最多 8 个图
                row = idx % 4   # 当前列中的第几行

                p = self.win.addPlot(title=f'CH{ch}', row=row, col=col)
                p.setLabel('left', 'Volt', units='μV')
                p.setLabel('bottom', 'Time', units='s')
                p.showGrid(x=True, y=True)
                p.setYRange(min=-300, max=300, padding=0)  # 设置固定范围
                # p.enableAutoRange(axis='y', enable=True)

                # 创建绘图曲线
                c = p.plot(pen=pg.mkPen(color=pg.intColor(ch), width=1.5))
                self.curves.append(c)
                self.plots.append(p)

            # 初始化 BrainFlow
            self.params = BrainFlowInputParams()
            self.params.serial_port = serial_port
            self.board = BoardShim(self.board_id, self.params)
            self.board.prepare_session()
            self.board.start_stream()
            board_descr = BoardShim.get_board_descr(board_id)
            for key in board_descr:     # 获取通道标签
                print(f"{key}: {board_descr[key]}")

            # 预测输出框
            if self.display_enabled:
                self.label = QtWidgets.QLabel("预测手势: ", self)
                self.label.setGeometry(20, 10, 300, 30)

        # 定时器
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval_ms)

        # 设置数据保存
        timestamp = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
        self.output_file = f'./result-{timestamp}_raw.txt'
        self.output_file_filter = f'./result-{timestamp}_filter.txt'            
        with open(self.output_file, 'a') as f1, open(self.output_file_filter, 'a') as f2:
            header = 'Time ' + ' '.join([f'Ch{i+1}' for i in range(self.chan_num)]) + '\n'
            f1.write(header)
            f2.write(header)
        
        # 初始化写入线程
        self.writer_thread = DataWriterThread(self.output_file_filter, self.output_file, self.chan_num)
        self.writer_thread.start()
        self.hand_thread = HandControlThread()
        self.hand_thread.start()

    def update_plot(self):
        if self.flag == 1:
            data = self.board.get_board_data()
            self.last_time = time.time() - self.start_time
            self.last_time_save = time.time() - self.start_time
            self.flag = 0
        else:
            # --- 读取sensor数据 ---
            # data = self.board.get_board_data()    # 容易获取空数据，导致曲线出现不连续
            window_size = int(self.sample_rate * self.update_interval_ms / 1000)
            data = self.board.get_current_board_data(window_size)
            self.flag_pred += 1
            if data.shape[1] == 0:
                print('Empty data')
                return

            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            emg_data = data[eeg_channels[:self.chan_num], :]
            num_samples = emg_data.shape[1]

            # Update time list
            current_time = time.time() - self.start_time
            temp_x = [self.last_time + j * (current_time - self.last_time)/num_samples for j in range(num_samples)]
            self.time_buffer += temp_x
            if self.max_enabled:
                while len(self.time_buffer) >= self.max_points:
                    self.time_buffer.pop(0)
            self.last_time = current_time

            # Update data list
            temp_y = [[] for _ in range(channel_num)]  # N个通道的临时数据，用于写入.txt
            for i in range(self.chan_num):
                self.data_buffer[i] += list(emg_data[i])
                # self.data_pred_buffer[i] += list(emg_data[i])
                if self.max_enabled:
                    if len(self.data_buffer[i]) >= self.max_points:
                        self.data_buffer[i] = self.data_buffer[i][-max_points:] 
                
                # OpenBCI自带滤波器
                filter_period = 2 * self.sample_rate   # 滤波需要一定的数据长度，即2s的数据量
                if len(self.data_buffer[i]) >= filter_period:
                    data_array = np.array(self.data_buffer[i], dtype=np.float64)
                    DataFilter.detrend(data_array, DetrendOperations.CONSTANT.value)    # 去除直流漂移                
                    DataFilter.perform_bandpass(data_array, self.sample_rate,
                                                5.0, 47.0, 4, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)    # 带通滤波
                    # DataFilter.perform_bandstop(data_array, self.sample_rate,
                                                # 47.0, 53.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                    DataFilter.perform_rolling_filter(data_array, 3, AggOperations.MEAN.value)  # 滚动均值滤波（平滑曲线）
                    
                    self.data_buffer_filter[i] = list(data_array)

            # update curve list
            if self.display_enabled:
                for i in range(self.chan_num):
                    if len(self.data_buffer_filter[i]) != 0:
                        min_len = min(len(self.time_buffer), len(self.data_buffer_filter[i]))
                        self.curves[i].setData(self.time_buffer[-min_len:], self.data_buffer_filter[i][-min_len:])
                    else:
                        min_len = min(len(self.time_buffer), len(self.data_buffer[i]))
                        self.curves[i].setData(self.time_buffer[-min_len:], self.data_buffer[i][-min_len:])

            # --- 多线程写入.txt文件---  1秒写入一次
            if len(self.data_buffer_filter[i]) != 0:
                if current_time - self.last_time_save >= 2:
                    t_start = self.last_time_save
                    t_end = current_time - 1     
                    
                    selected_idx = [j for j, t in enumerate(self.time_buffer) if t_start < t <= t_end]
                    selected_time = [self.time_buffer[j] for j in selected_idx]
                    filtered_segment = [[self.data_buffer_filter[i][j] for j in selected_idx] for i in range(self.chan_num)]
                    raw_segment = [[self.data_buffer[i][j] for j in selected_idx] for i in range(self.chan_num)]

                    # 发给线程
                    self.writer_thread.add_data(selected_time, filtered_segment, raw_segment)
                    self.last_time_save = t_end

            # --- 手势预测 ---滤波后的数据预测
            if len(self.data_buffer_filter[1]) >= self.window_size & self.flag_pred >= self.predict_interval:
                self.flag_pred = 0
                # 使用最近的100个数据点做预测（对应训练时 window_size=100）
                window = np.array(self.data_buffer_filter)[:, -1*self.window_size:]

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
                    if self.display_enabled:
                        self.label.setText(f"Predicted gesture: {pred}")
                    print(f"[Prediction] {pred}")
                    self.pred_buffer.append(pred)
                    # modify four numbers
                    repeat_num = 2
                    if self.pred_buffer[-1*repeat_num:] == ['fist'] * repeat_num:
                        joints_states = UDF_Gesture.fist
                        self.hand_thread.add_command(joints_states)
                        # hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
                        # hand_commander.move_to_joint_value_target(joints_states, wait=True, angle_degrees=True)
                    if self.pred_buffer[-1*repeat_num:] == ['rest'] * repeat_num:
                        joints_states = UDF_Gesture.open
                        self.hand_thread.add_command(joints_states)
                        # hand_commander.plan_to_joint_value_target(joints_states, angle_degrees=True)
                        # hand_commander.move_to_joint_value_target(joints_states, wait=True, angle_degrees=True)
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
        self.writer_thread.stop()
        self.writer_thread.wait()  # 等待线程退出
        self.hand_thread.stop()
        self.hand_thread.wait()
        event.accept()

class DataWriterThread(QThread):
    write_signal = pyqtSignal(list, list, list)  # 时间戳、滤波后数据、原始数据

    def __init__(self, file_filtered, file_raw, chan_num):
        super().__init__()
        self.file_filtered = file_filtered
        self.file_raw = file_raw
        self.chan_num = chan_num
        self.queue = []
        self.lock = QtCore.QMutex()
        self.running = True

    def run(self):  # 启动线程后一直运行
        while self.running:
            self.msleep(10)
            self.lock.lock()
            if self.queue:
                time_buffer, filtered_data, raw_data = self.queue.pop(0)    # 弹出列表的第一个元素（先进先出），并将其从列表中移除
                self.lock.unlock()
                self._write_to_file(time_buffer, filtered_data, raw_data)
            else:
                self.lock.unlock()

    def add_data(self, time_buffer, filtered_data, raw_data):
        self.lock.lock()
        self.queue.append((time_buffer, filtered_data, raw_data))   # 列表增加一个元素
        self.lock.unlock()

    def _write_to_file(self, time_buffer, filtered_data, raw_data):
        with open(self.file_filtered, 'a') as f1, open(self.file_raw, 'a') as f2:
            for j in range(len(time_buffer)):
                t = time_buffer[j]
                line1 = f"{t:.3f} " + ' '.join([f"{filtered_data[i][j]:.2f}" for i in range(self.chan_num)]) + '\n'
                line2 = f"{t:.3f} " + ' '.join([f"{raw_data[i][j]:.2f}" for i in range(self.chan_num)]) + '\n'
                f1.write(line1)
                f2.write(line2)
    def stop(self):
        self.running = False

class HandControlThread(QThread):
    def __init__(self):
        super().__init__()
        self.queue = []  # 存储待执行的关节角度
        self.lock = QMutex()
        self.running = True

    def run(self):
        while self.running:
            self.msleep(10)
            self.lock.lock()
            if self.queue:
                joints = self.queue.pop(0)
                self.lock.unlock()
                try:
                    # 控制 Shadow Hand
                    hand_commander.plan_to_joint_value_target(joints, angle_degrees=True)
                    hand_commander.move_to_joint_value_target(joints, wait=True, angle_degrees=True)
                except Exception as e:
                    print(f"[Hand Thread Error] {e}")
            else:
                self.lock.unlock()

    def add_command(self, joints):
        self.lock.lock()
        # 如果队列为空，则添加；否则等待当前执行完
        if not self.queue:
            self.queue.append(joints)
        self.lock.unlock()

    def stop(self):
        self.running = False

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)    
    # window = EMGPlotter()
    window_EMG = EMGPlotter(board_id=board_id, \
                        serial_port=ser_port, \
                        channel_list=channel_list, \
                        max_enabled=max_enabled, \
                        max_points=max_points, \
                        predict_interval=predict_interval, \
                        display_enabled=display_enabled, \
                        update_interval_ms=update_interval_ms, \
                        window_size=window_size)
    window_EMG.show()
    window_EMG.setFocus()
    sys.exit(app.exec_())
