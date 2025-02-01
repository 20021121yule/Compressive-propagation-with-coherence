function gamma = coherence_coefficient(U1, U2)
    % 计算相干系数
    numerator = abs(mean(conj(U1) .* U2, 'all'));  % 互相干项
    denominator = sqrt(mean(abs(U1).^2, 'all') * mean(abs(U2).^2, 'all'));  % 归一化
    gamma = numerator / denominator;
end

% 生成两个测试波前（一个是高相干，另一个是低相干）
img_size = 64;
wavefront1 = exp(1i * 2 * pi * rand(img_size, img_size));  % 完全随机波前
wavefront2 = wavefront1 .* exp(1i * 1 * randn(img_size, img_size));  % 加入小扰动，保持高相干
wavefront3 = exp(1i * 2 * pi * rand(img_size, img_size));  % 完全独立的随机波前（低相干）

% 计算相干系数
gamma_12 = coherence_coefficient(wavefront1, wavefront2);
gamma_13 = coherence_coefficient(wavefront1, wavefront3);

disp(['Coherence between wavefront1 and wavefront2 (High coherence): ', num2str(gamma_12)]);
disp(['Coherence between wavefront1 and wavefront3 (Low coherence): ', num2str(gamma_13)]);