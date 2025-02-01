clc; clear; close all;

% 生成随机波前 (代表部分相干光)
img_size = 256;  % 图像大小
wavefront = rand(img_size, img_size); 

% 计算 2D FFT 并中心化
F_wavefront = fftshift(fft2(wavefront));

% 生成空间频率滤波器
filter_types = {'pinhole', 'partial_open', 'fully_open', 'ring'};
filter_size = 20;  % 滤波器大小（调整影响相干性）

% 计算频率坐标
[u, v] = meshgrid(linspace(-1, 1, img_size), linspace(-1, 1, img_size));
radius = sqrt(u.^2 + v.^2);

% 生成滤波器
filters = struct();
filters.pinhole = double(radius <= filter_size / img_size);  % 小孔（高相干）
filters.partial_open = double(radius <= 2 * filter_size / img_size);  % 部分开孔
filters.fully_open = ones(img_size, img_size);  % 全开孔（低相干）
filters.ring = double(radius <= 3 * filter_size / img_size & radius >= filter_size / img_size);  % 环形孔

% 应用不同滤波器并进行逆变换
figure;
for i = 1:length(filter_types)
    filter_name = filter_types{i};
    filtered_F = F_wavefront .* filters.(filter_name);  % 频域滤波
    filtered_wavefront = abs(ifft2(ifftshift(filtered_F)));  % 逆变换回到空间域
    
    subplot(2,2,i);
    imagesc(filtered_wavefront);
    colormap gray; axis off; axis image;
    title(['Filtered: ', filter_name]);
end