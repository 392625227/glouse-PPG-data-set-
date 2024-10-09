# 使用PPG数据预测血糖的程序（输出图像）
# 输入：PPG数据文本文件（每16个数据对应一个血糖数值）；输出：血糖曲线，血糖数值文本文件

import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QFileDialog, QLabel, QWidget, \
    QGridLayout
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

# 加载模型
cnn_model = load_model('D:\python\My_Python_Works\ppg_glucose_model.h5')    #请注意模型路径！


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Prediction with PyQt and Matplotlib')

        # 创建布局
        layout = QVBoxLayout()

        # 创建按钮
        self.btn_upload = QPushButton('Upload File')
        self.btn_upload.clicked.connect(self.upload_file)

        self.btn_predict = QPushButton('Predict')
        self.btn_predict.clicked.connect(self.predict)

        # 创建标签用于显示输出
        self.label_output = QLabel('Output:')

        # 创建Matplotlib图表
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        # 将组件添加到布局中
        layout.addWidget(self.btn_upload)
        layout.addWidget(self.btn_predict)
        layout.addWidget(self.label_output)
        layout.addWidget(self.canvas)

        # 创建一个容器并设置布局
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_file(self):
        # 打开文件对话框，让用户选择文件
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt)")

        if file_name:
            try:
                # 读取文件内容，假设每行只有一个数
                with open(file_name, 'r') as file:
                    data = np.array([float(line.strip()) for line in file])

                    # 检查行数
                    if data.shape[0] > 2400:
                        print(':( 数据量太大，请重新选择文件')

                        # 删除每隔17行的数据
                wd = np.delete(data, np.s_[::17])

                # 设置数据段的大小为类属性（而不是全局变量）
                self.SEG = 16
                seg_size = self.SEG

                # 计算最多分多少段，并截断数据
                seg_count = len(wd) // seg_size
                wd = wd[: seg_count * seg_size]

                # 对每个数据段进行规格化（标准化）
                for i in range(seg_count):
                    m, s = data[i].mean(), data[i].std()
                    data[i] -= m
                    data[i] /= s if s != 0 else 1  # 避免除以零错误
                    # 根据3σ准则，删除绝对值大于3的数据点
                    # data[i] = data[i][np.abs(data[i]) <= 3]

                # 将数据重新整形为二维数组，每个段包含SEG个数据点
                data = wd.reshape(-1, seg_size)

                    # 将数据存储在实例变量中
                self.input_data = data

            except IOError as e:
                print(f"An I/O error occurred: {e}")
                # 处理错误情况，例如通知用户或记录日志

        else:
            # 用户取消了文件选择
            pass

    
    def predict(self):
        # # 检查是否有输入数据
        # if hasattr(self, 'input_data'):
            # 进行预测
            predictions = cnn_model.predict(self.input_data)

            # 提取第二列数据
            blood_glucose =( predictions[:, 1] /10 ) /1000000  

            # 写入TXT文件
            txt_file_path = 'blood_glucose.txt'  # TXT文件路径
            with open(txt_file_path, 'w') as file:
                for value in blood_glucose:
                    file.write(f"{value}\n")  # 写入每个值，并在其后加上换行符

            # 对blood_glucose进行滑动滤波
            window_size = 5  # 设置滑动窗口的大小，可以根据需要调整
            blood_glucose_smoothed = uniform_filter1d(blood_glucose, size=window_size)

            # 创建与blood_glucose相同长度的索引数组
            x = np.arange(len(blood_glucose))

            # 使用线性插值创建一个插值函数
            f = interp1d(x, blood_glucose_smoothed, kind='cubic')

            # 创建用于绘图的平滑曲线的数据点
            xnew = np.linspace(x.min(), x.max(), num=1000, endpoint=True)
            ynew = f(xnew)

            # 清除之前的绘图
            self.ax.clear()
            # 绘制平滑曲线
            self.ax.plot(xnew, ynew, label='Blood Glucose (mmol/L)')
            # 设置坐标轴标签和图例
            self.ax.set_xlabel('Index')
            self.ax.set_ylabel('Blood Glucose (mmol/L)')
            self.ax.legend()
            # 刷新画布以显示新的图形
            self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())