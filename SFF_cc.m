clear;

% 参数设置
N = 64; % 波前的尺寸
M = 30; % 波前的数量
img_size = N;

% 生成随机波前（N^2 x M）
wavefronts = exp(1i * pi * rand(N^2, M));

% 设计圆形低通滤波器
radius = 20; % 圆形滤波器的半径（控制滤波器的带宽）
low_pass_filter = create_circular_low_pass_filter(img_size, radius);

% 调用函数进行滤波
filtered_wavefronts = filter_wavefronts(wavefronts, low_pass_filter);

% 可视化滤波器
figure;
imagesc(low_pass_filter); % 显示滤波器
axis image;
title('圆形低通滤波器');
colormap gray;

% 计算任意两个光束的相干性
beam1_index = 1; % 第一个光束的索引
beam2_index = 2; % 第二个光束的索引
coherence_value = calculate_coherence(filtered_wavefronts, beam1_index, beam2_index);
fprintf('光束 %d 和光束 %d 的相干性值为: %.4f\n', beam1_index, beam2_index, coherence_value);

function filtered_wavefronts = filter_wavefronts(wavefronts, low_pass_filter)
    % 输入：
    %   wavefronts: N^2 x M 的矩阵，每一列是一个波前（N^2 是波前的像素数，M 是波前的数量）
    %   low_pass_filter: N x N 的低通滤波器（频域滤波器）
    % 输出：
    %   filtered_wavefronts: N^2 x M 的矩阵，每一列是经过滤波后的波前

    % 获取输入尺寸
    [N2, M] = size(wavefronts);
    N = sqrt(N2); % 波前的尺寸（假设是正方形）
    N = round(N); % 确保 N 是整数

    % 将 wavefronts 转换为 N x N x M 的三维矩阵
    wavefronts_3d = reshape(wavefronts, [N, N, M]);

    % 对每个波前进行频域滤波
    F_wavefronts = fft2(wavefronts_3d); % 对每个波前进行二维傅里叶变换

    % 将零频移到中心
    F_wavefronts = fftshift(F_wavefronts, 1);
    F_wavefronts = fftshift(F_wavefronts, 2);

    % 扩展低通滤波器以匹配波前的数量
    low_pass_filter_3d = repmat(low_pass_filter, [1, 1, M]);

    % 频域滤波
    F_filtered = F_wavefronts .* low_pass_filter_3d;

    % 将零频移回角落
    F_filtered = ifftshift(F_filtered, 1);
    F_filtered = ifftshift(F_filtered, 2);

    % 逆傅里叶变换并提取相位
    filtered_wavefronts_3d = ifft2(F_filtered); % 逆变换
    smoothed_phase = angle(filtered_wavefronts_3d); % 提取相位

    % 将相位转换为复振幅
    filtered_wavefronts_3d = exp(1i * smoothed_phase);

    % 将结果转换回 N^2 x M 的矩阵
    filtered_wavefronts = reshape(filtered_wavefronts_3d, [N2, M]);
end

function low_pass_filter = create_circular_low_pass_filter(img_size, radius)
    % 创建圆形低通滤波器
    % 输入：
    %   img_size: 图像的尺寸（假设为正方形）
    %   radius: 圆形滤波器的半径
    % 输出：
    %   low_pass_filter: img_size x img_size 的圆形低通滤波器

    % 创建网格
    [X, Y] = meshgrid(-img_size/2:img_size/2-1, -img_size/2:img_size/2-1);

    % 计算每个点到中心的距离
    distance = sqrt(X.^2 + Y.^2);

    % 创建圆形掩膜
    low_pass_filter = double(distance <= radius);

    % 不需要额外的 fftshift，因为网格已经以零频为中心
end

function coherence_value = calculate_coherence(filtered_wavefronts, beam1_index, beam2_index)
    % 计算任意两个光束的相干性
    % 输入：
    %   filtered_wavefronts: N^2 x M 的矩阵，每一列是一个波前
    %   beam1_index: 第一个光束的索引
    %   beam2_index: 第二个光束的索引
    % 输出：
    %   coherence_value: 两个光束的相干性值（范围 [0, 1]）

    % 提取两个光束的复振幅
    beam1 = filtered_wavefronts(:, beam1_index);
    beam2 = filtered_wavefronts(:, beam2_index);

    % 计算互相干函数
    mutual_coherence = abs(beam1' * beam2) / (norm(beam1) * norm(beam2));

    % 返回相干性值
    coherence_value = mutual_coherence;
end