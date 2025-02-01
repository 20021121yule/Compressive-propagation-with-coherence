clear;

Optimizer_Pixel = 64;% 调制器像素
Optimizer_Pitch  = 4.5; % 调制器像素间距，单位mm

Sensor_Pixel = 64;% 接收器像素
Sensor_Pitch  = 4.5; % 接收器像素间距，单位mm

% 读取图片
img = imread('peppers.png'); % 读取图片
img_gray = double(rgb2gray(img)); % 转换为灰度图，并转换为 double 类型
img_resized = imresize(img_gray, [Sensor_Pixel, Sensor_Pixel], 'bilinear'); % 调整大小到 Sensor_Pixel×Sensor_Pixel

img_normalized = img_resized / max(img_resized(:)); % 归一化

% 转换为列向量
target_pattern = img_normalized(:); 

%%
% 参数设置
unitSize = [Optimizer_Pixel, Sensor_Pixel];
unitWidth = [Optimizer_Pitch Sensor_Pitch];

layerDistance = 72;% 调制器和接收器的距离，单位mm

frequency = 26.4e9;% 光波长，单位Hz,由于计算资源有限，我们这里直接用mm波来算。

M = 30;% 一组随机波前个数

radius = 40;% Spatial Frequnency Filter的低通半径，尽量在Optimizer_Pixel的一半左右。
% radius 太小会导致光强不足，从而梯度爆炸。radius 太大会导致光的相干性差，最后拟合效果很糟糕。

net = CP_R(unitSize, unitWidth, layerDistance, frequency, M, radius);

%% 训练参数
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',300, ...
    'InitialLearnRate', 0.15, ...
    'MaxEpochs',1000);

net = net.trainCP_R(target_pattern, options, 'Amp');

%% 作图
figure;
net.plotTrainingRMSECurves();

figure;
net.plotOptimizedPattern();

figure;
net.plotHologramPattern();

