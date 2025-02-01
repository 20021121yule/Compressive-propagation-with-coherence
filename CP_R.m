classdef CP_R
    %此程序复现随机干涉压缩传播，即CP_R模式
    %   referrence: < Compressive propagation with coherence >

    properties (Constant, GetAccess = private)
        c0 = 299792458 * 1e3;   % 光速，单位mm/s
    end

    properties (SetAccess = immutable, GetAccess = public)  % 不可变类
        unitSize        % 每层单元（光调制和Snensor）一维边长个数，大小2 * 1
        % 注意，unitSize有每一层又不同的物理意义。
        % unitSize(1)是调制器的像素大小，unitSize(2)是接收器(sensor)的像素大小

        unitNums        % 每层单元数，大小2 * 1，等于unitSize.^2
        unitWidth       % 每层单元边长，单位mm，大小2 * 1

        layerDistance   % 层间距离，单位mm，表示调制层与最后Sensor的距离

        frequency       % 频率，单位Hz

        M               % M个波前
    end

    properties (SetAccess = private, GetAccess = public)
        W               % SFF处理后波前。每一次梯度更新都生成M个SFF处理后波前。
        % 三维矩阵类型，大小为unitNums(1)*M
    end

    properties (SetAccess = private, GetAccess = public)
        O               % 调制矩阵，Diagnoal Diffractive Modulation Matrix
        % 大小为unitSize(1)^2 * 1
        % 这里原本为矩阵，现在改为列向量，乘法 * 变为 .*

    end

    properties (SetAccess = immutable, GetAccess = public)
        P               % 衍射矩阵，Diffractive Weight Matrix，是从光调制器到Sensor的衍射矩阵
        % 矩阵，大小（unitSize（1）^2 * unitSize（2）^2）
    end

    properties (SetAccess = immutable, GetAccess = public)
        spatialFrequencyFilter       % 空间频率滤波器。
        % 大小unitSize(1) * unitSize(1）
        % 先将波转换成unitSize(1) * unitSize(1）后，再用spatialFrequencyFilter作点乘
    end

    properties (Constant, GetAccess = private)
        % 光学元件优化类型
        %   梯度计算公式gradientUpdate
        %
        netTypeAmplitude = 0           % 振幅型调制优化
        netTypePhase = 1               % 相位型调制优化
    end

    properties (SetAccess = private, GetAccess = public)
        netType          % 网络类型

        trainingOptions  % 参考MATLAB内置机器学习的参数

        trainStartTime      % 训练起始时间
        trainDuration       % 训练时间

        mu               % 学习率（动态变化）

        iter             % 迭代次数

        RMSE       % Minibatch均方差(mean-squared-error)

    end

    methods(Access = public)
        function obj = CP_R(unitSize, unitWidth, layerDistance, frequency, M, radius)
            %CP_R 构造此类的实例
            %   Input:
            %       unitSize            层阵列单元边长数（光调制和Snensor）
            %       unitWidth           单元边长（mm）
            %       layerDistance       层间距离（mm）
            %       frequency           频率（Hz）
            %       radius              滤波器像素半径

            assert(length(unitSize) == 2, '单元数组与层数不一致');
            assert(length(unitWidth) == 2, '单元边长数组与层数不一致');
            assert(isscalar(layerDistance), '间距数组与层数不一致');
            assert(layerDistance(1) >=0 && all(layerDistance(2:end)>0), '间距数组数值需大于0');

            obj.unitSize = unitSize;
            obj.unitNums = unitSize.^2;
            obj.unitWidth = unitWidth;
            obj.layerDistance = layerDistance;

            obj.frequency = frequency;

            obj.M = M;

            % 生成衍射矩阵P
            obj.P = ones(obj.unitNums(1), obj.unitNums(2));
            obj.P = obj.PGenerate(obj.unitWidth, obj.unitSize, obj.layerDistance, obj.frequency);

            % 生成空间频率滤波器
            obj.spatialFrequencyFilter = zeros(obj.unitSize(1), obj.unitSize(1));
            obj.spatialFrequencyFilter = obj.create_circular_low_pass_filter(radius);% 所选择的SFF半径要尽量满足大以及尽量让相干性大于0.6

            % 初始化调制矩阵O
            obj.O = ones(obj.unitNums(1), 1);

            % 初始化波前(没有滤波)，注意，random_wavefrounts的seed需要保证其空间相干性。
            obj.W = obj.filter_wavefronts(exp(1j * (pi * rand(obj.unitNums(1), obj.M))));

        end

        function obj = trainCP_R(obj, target_pattern, trainingOptions, inputNetType)
            %trainCR_R 基于我们所选取的target_pattern
            %   Input: target_pattern
            %          trainingOptions
            %          inputNetType

            % 网络类型参数
            if(strcmpi(inputNetType, 'Amp'))
                obj.netType = obj.netTypeAmplitude;
            elseif(strcmpi(inputNetType, 'Phase'))
                obj.netType = obj.netTypePhase;
            else
                error("Error: inputNetType(光学元件优化类型)仅支持'Amp'与'Phase'。");
            end

            % trainOpt复制过来
            obj.trainingOptions = trainingOptions;

            % 学习率
            obj.mu = obj.trainingOptions.InitialLearnRate;

            % RMSE数据集，注意，这种提前占位置的操作能够极大地提升运行速度。
            obj.RMSE = zeros(obj.trainingOptions.MaxEpochs, 1);% 总共计算MaxEpochs * batchNum次，一个batch计算完就计算一次miniBatchRMSE

            % 训练迭代初始
            obj.iter = 1;
            obj.trainStartTime = datetime('now');
            fprintf("%s\t%s\t%s\n", "Duration", "Iter", "RMSE");

            %
            for epoch = 1:obj.trainingOptions.MaxEpochs
                % 每一次迭代重新选择一套随机波前
                obj.W = obj.filter_wavefronts(exp(1j * (pi * rand(obj.unitNums(1), obj.M))));

                % 梯度更新
                obj = obj.gradientUpdate(target_pattern);

                % 学习率逐epoch下降（mu的动态变化）
                if(~mod(epoch, obj.trainingOptions.LearnRateDropPeriod))
                    obj.mu = obj.mu * obj.trainingOptions.LearnRateDropFactor;
                end

                % 每 100 次迭代显示一次计算时间和 RMSE
                if mod(obj.iter, 100) == 0
                    duaration = datetime('now') - obj.trainStartTime; % 计算时间
                    fprintf("%s\t%d(%d)\t%.3e\n",duaration, obj.iter, obj.trainingOptions.MaxEpochs, obj.RMSE(obj.iter));
                end

                obj.iter = obj.iter + 1;% 更新规则： 每梯度下降一次，就迭代次数加一次。

            end

            % 结束训练
            obj.trainDuration = datetime('now') - obj.trainStartTime;

        end

        function P = PGenerate(obj, width, unitSize, dis, freq)
            %WGenerate 基于位置分布生成第ii个散射矩阵

            % 角波数
            k = 2 * pi * freq / obj.c0;

            % 子边框
            l1 = width(1) * ((1:unitSize(1)) - (unitSize(1)+1)/2);
            l2 = width(2) * ((1:unitSize(2)) - (unitSize(2)+1)/2);

            [x1, y1] = meshgrid(l1, l1);
            [x2, y2] = meshgrid(l2, l2);

            % 空间坐标
            x1 = x1(:)'; y1 = y1(:)';
            x2 = x2(:); y2 = y2(:);

            r = sqrt((x1-x2).^2 + (y1-y2).^2 + dis^2);

            % 散射矩阵
            P = exp(-1j * k * r) ./ (r.^2);

            % 归一化(保证变换前后能量守恒)
            % 矩阵2范数：|W|_2 = sqrt[λ_max[W*W^H]],这个范数不要求矩阵是否有奇异值（也不要求是否是方阵），总之就是可以被应用矩阵2范数归一化的。
            P = P / norm(P, 2);
        end

        function g = netPredictLayers(obj)
            %前向传播计算（Forward propagation）
            % 计算 g = (1/M) * sum_m | P * diag(O) * W_m |^2

            propagated_W = obj.P * (obj.O .* obj.W); % (N^2, M)% 计算传播 P * (O .* W)
            intensity = abs(propagated_W).^2; % (N^2, M)% 计算强度 |P * (O .* W)|^2

            % 对 M 个波前取均值
            g = mean(intensity, 2); % (N^2, 1)

        end

        function spatialFrequencyFilter = create_circular_low_pass_filter(obj, radius)
            % 创建圆形低通滤波器
            % 输入：
            %   img_size: 图像的尺寸（假设为正方形）
            %   radius: 圆形滤波器的半径
            % 输出：
            %   spatialFrequencyFilter: img_size x img_size 的圆形低通滤波器

            % 创建网格，网格化调制器端
            [X, Y] = meshgrid(-obj.unitSize(1)/2:obj.unitSize(1)/2-1, -obj.unitSize(1)/2:obj.unitSize(1)/2-1);

            % 计算每个点到中心的距离
            distance = sqrt(X.^2 + Y.^2);

            % 创建圆形掩膜
            spatialFrequencyFilter = double(distance <= radius);
        end

        function filtered_wavefronts = filter_wavefronts(obj, random_wavefronts)
            % 输入：
            %   wavefronts: N^2 x M 的矩阵，每一列是一个波前（N^2 是波前的像素数，M 是波前的数量）
            %   low_pass_filter: N x N 的低通滤波器（频域滤波器）
            % 输出：
            %   filtered_wavefronts: N^2 x M 的矩阵，每一列是经过滤波后的波前

            % 获取输入尺寸
            [N2, ~] = size(random_wavefronts);
            N = sqrt(N2); % 波前的尺寸（假设是正方形）
            N = round(N); % 确保 N 是整数

            % 将 wavefronts 转换为 N x N x M 的三维矩阵
            wavefronts_3d = reshape(random_wavefronts, [N, N, obj.M]);

            % 对每个波前进行频域滤波
            F_wavefronts = fft2(wavefronts_3d); % 对每个波前进行二维傅里叶变换

            % 将零频移到中心
            F_wavefronts = fftshift(F_wavefronts, 1);
            F_wavefronts = fftshift(F_wavefronts, 2);

            % 扩展低通滤波器以匹配波前的数量
            low_pass_filter_3d = repmat(obj.spatialFrequencyFilter, [1, 1, obj.M]);

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
            filtered_wavefronts = reshape(filtered_wavefronts_3d, [N2, obj.M]);
        end

        function obj = gradientUpdate(obj, target_pattern)
            %对obj进行梯度更新
            % 此版本只是计算的是对入射波的相位调制版本。如果要计算振幅调制，就令最后的O的俯角为0即可。

            % 计算误差 e = g - ĝ
            g = obj.netPredictLayers(); % 计算 g
            e = g - target_pattern;     % 计算误差 (N^2, 1)

            % 计算 diag(P * diag(O) * W_m) * e
            modulated_W = obj.O .* obj.W; % (N^2, M)
            propagated_W = obj.P * modulated_W; % (N^2, M)
            diag_term = propagated_W .* e; % 逐元素乘法 (N^2, M)

            % 计算 P^H * diag_term
            adjoint_term = obj.P' * diag_term; % Hermitian 转置 (N^2, M)

            % 计算 diag(F W) 的 Hermitian 转置
            W_H = conj(obj.W); % 取共轭 (N^2, M)

            % 计算最终梯度
            gradient = (4 / obj.M) * sum(W_H .* adjoint_term, 2); % 按列求和，得到 (N^2, 1)

            obj.O = obj.O - obj.mu * gradient;  % 标准 SGD 更新

            switch(obj.netType)
                case(obj.netTypeAmplitude)

                    obj.O = real(obj.O);  % 取实部，确保 O 是实数
                    obj.O = max(0, min(1, obj.O));  % 强制 O 在 [0,1] 之间

                case(obj.netTypePhase)

                    obj.O = obj.O ./ abs(obj.O); % 使 |O| = 1

            end

            obj.RMSE(obj.iter) = sqrt(mean((g - target_pattern).^2));

        end

        function plotTrainingRMSECurves(obj)
            %作图程序，作出RMSE曲线

            plot(obj.RMSE);

            switch(obj.netType)
                case(obj.netTypeAmplitude)

                    legend("RMSE_Amplitude");

                case(obj.netTypePhase)

                    legend("RMSE_Phase");

            end

        end

        function plotOptimizedPattern(obj)
            %作图程序，作出所需优化图案

            switch(obj.netType)
                case(obj.netTypeAmplitude)

                    imagesc(abs(reshape(obj.O, obj.unitSize(1),[])));% 这里将O转换为调制器器的像素
                    colormap gray; colorbar;
                    title('Optimized Ampiltude Pattern');

                case(obj.netTypePhase)

                    imagesc(angle(reshape(obj.O, obj.unitSize(1),[])));% 这里将O转换为调制器器的像素
                    colormap gray; colorbar;
                    title('Optimized Phase Pattern');

            end

        end

        function plotHologramPattern(obj)
            %作图程序，作出最终优化计算得到的全息图案

            g = obj.netPredictLayers();
            imagesc(reshape(g, obj.unitSize(2),[]));% 这里将g转换为接收器的像素
            colormap gray; colorbar;
            title('Hologram picture');

        end


    end
end