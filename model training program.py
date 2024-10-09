### keras 2.3, tensorflow 2.0
# 注释：代码使用的库版本，keras 2.3 和 tensorflow 2.0
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import os, sys
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import shutil

try:
    shutil.rmtree(r'data1\.ipynb_checkpoints')
except:
    print('')
    print('')

def sao2data9(sao2file, file):
    print(file)
    # 打印文件名
    wd = np.loadtxt(sao2file)
    # 从文件中加载数据
    wd = np.delete(wd, np.s_[::17])
    # 把11551155行 删掉
    # 原始数据文件格式: 11551155 xx xx xx ... xx  (1+16)个数据
    global SEG
    SEG = 16
    # 定义数据段的大小
    seg = len(wd) // SEG
    # 计算最多分多少段
    xdata = wd[: seg * SEG].reshape(-1, SEG)
    # 将数据重新整形为二维数组，每个段包含SEG个数据点
    # 数据规格化
    for i in range(seg):
        m = xdata[i].mean()
        s = xdata[i].std()
        xdata[i] -= m
        xdata[i] /= s
        # 对每个数据段进行规格化（标准化），使其均值为0，标准差为1
    ydata = []
    # 初始化ydata列表，用于存储目标数据
    for i in range(seg):
        # 数据文件名命名规则 HHH_SSS_XXX_GGG   H- hr, 3位， S-SaO2, 3位， XXX-任意, GGG-glucose
        ## ydata = np.append(ydata, int(file[0:3]))
        ## 从文件名的第0到第3个字符提取hr（心率）信息，并转换为整数，然后追加到ydata中
        ydata = np.append(ydata, int(file[4:7]))
        # 从文件名的第4到第7个字符提取SaO2（血氧饱和度）信息，并转换为整数，然后追加到ydata中
        ydata = np.append(ydata, int(file[12:15]))
        # 从文件名的第12到第15个字符提取glucose信息（任意3位），并转换为整数，然后追加到ydata中
    ydata = ydata.reshape(-1, 2)
    # 将ydata重新塑形为(-1, 2)的二维数组，其中-1表示让numpy自动计算行数
    # 使用train_test_split函数将xdata和ydata划分为训练集和测试集
    xtr, xte, ytn, yten = train_test_split(xdata, ydata)  # , test_size=0.030, random_state=42)
    # 返回训练集和测试集
    return (xtr, ytn), (xte, yten)
def getdata4():
    # 获取数据并处理，返回训练集和测试集的数据与标签。
    path = r'data1\\'
    # 设置数据所在文件夹路径
    dirs = os.listdir(path)
    # 获取文件夹下的所有文件/文件夹名称
    # 初始化空的训练集数据和标签数组
    train_data = np.empty((0, 16))  # 初始化为一个0行16列的二维数组
    train_targets = np.empty((0, 2))  # 初始化为一个0行2列的二维数组
    # 初始化空的测试集数据和标签数组
    test_data = np.empty((0, 16))  # 初始化为一个0行16列的二维数组
    test_targets = np.empty((0, 2))  # 初始化为一个0行2列的二维数组
    for file in dirs:
        # 遍历文件夹下的所有文件
        (x_train, y_train_num), (x_test, y_test_num) = sao2data9(os.path.join(path, file), file)
        # 调用sao2data9函数处理单个文件，并获取数据和标签
        # 将单个文件的数据和标签添加到对应的训练集和测试集中
        train_data = np.append(train_data, x_train, axis=0)
        train_targets = np.append(train_targets, y_train_num, axis=0)
        test_data = np.append(test_data, x_test, axis=0)
        test_targets = np.append(test_targets, y_test_num, axis=0)
    # 将训练集和测试集的数据和标签合并
    xdata = np.append(train_data, test_data, axis=0)
    ydata = np.append(train_targets, test_targets, axis=0)
    # xdata = xdata/10000.
    ydata = ydata
    # 使用train_test_split函数将合并后的数据随机分割为训练集和测试集
    # 注意：原代码缺失了test_size和random_state参数，这里假定使用默认值
    train_data, test_data, train_targets, test_targets = train_test_split(xdata,
                                                                          ydata)  # , test_size=0.30, random_state=42)
    # 返回分割后的训练集和测试集的数据与标签
    return train_data, test_data, train_targets, test_targets

###--------main--------------------------------------
x_scale, testd, y, testt = getdata4()
print(x_scale.shape, y.shape)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(SEG,)))  # 假设SEG是一个整数，表示输入特征数
model.add(Dense(32, activation='relu'))  # 这一层有32个神经元
model.add(Dense(2))  # 这一层有2个神经元，没有激活函数，通常用于输出层（取决于任务类型）

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

# 对x_scale和testd进行形状重塑，以匹配模型的输入要求
x_scale = x_scale.reshape(x_scale.shape[0], 16)
testd = testd.reshape(testd.shape[0], 16)

# 使用均方误差作为损失函数，RMSprop作为优化器，并指定平均绝对误差作为评估指标来编译模型
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

# 打印模型的概要信息
model.summary()
# 初始化一个变量来存储最后的分数，默认值为1
lastscore = 1
# 初始化一个列表来存储所有的分数
all_scores = []
# 迭代500次
for i in range(500):
    # 训练模型，使用x_scale作为输入数据，y作为标签，批处理大小为512，训练轮数为1000，不打印训练进度信息
    model.fit(x_scale, y, batch_size=512, epochs=1000, verbose=0)  # , callbacks=callback_lists )
    # 评估训练后的模型在测试集上的性能
    scores = model.evaluate(testd, testt, verbose=2)
    # 将分数添加到列表中
    all_scores.append(scores)
    # 打印迭代次数和测试集上的损失值
    print(i, '.Test loss:', scores)
    # 如果当前模型的测试损失小于之前的最低损失
    if (scores[1] < lastscore):
        # 更新最低损失值
        lastscore = scores[1]
        # 保存当前模型到文件，文件名中包含损失的前5位数字
        model.save('glucose-' + str(scores[1])[:5] + '.h5')
# 使用训练好的模型对x_scale进行预测
y_hap = model.predict(x_scale)
# 将所有分数保存到Excel文件
df = pd.DataFrame(all_scores, columns=['MSE', 'MAE'])  # 假设scores有两个元素，第一个是损失，第二个是其他指标（如准确率）
# 如果scores只有一个元素（只有损失），则只设置一列
if len(scores) == 1:
    df = pd.DataFrame(all_scores, columns=['Loss'])

excel_path = 'all_scores_1.xlsx'
df.to_excel(excel_path, index=False)
print(f"所有分数已保存到 {excel_path}")
# 数据导出excel
data_y = y[:, 1] / 10
data_y_hap = y_hap[:, 1] / 10
df = pd.DataFrame({
    'Original': data_y,
    'Prediction': data_y_hap
})
# 导出DataFrame到Excel文件
excel_path = 'output_1.xlsx'
# index=False表示不导出行索引
df.to_excel(excel_path, index=False)
print(f'数据已成功导出到 {excel_path}')
model_save_path = 'D:\\python\\My_Python_Works\\ppg_glucose_model.h5'    #请注意模型路径！
model.save(model_save_path)