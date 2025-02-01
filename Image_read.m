% 读取图片
img = imread('peppers.png'); % 读取图片
img_gray = double(rgb2gray(img)); % 转换为灰度图，并转换为 double 类型
img_resized = imresize(img_gray, [128, 128], 'bilinear'); % 调整大小到 28×28

img_normalized = img_resized / max(img_resized(:)); % 归一化


% 转换为列向量
target_pattern = img_normalized(:); 