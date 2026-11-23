clear
clc
dbstop if error
warning off
% =========================================================
% 实验脚本
% ---------------------------------------------------------
% 作用：
%   使用 Bayesian Optimization 搜索 WGB-POJG-USC 模型的 3 个超参数
%   (1) pojg
%   (2) wgb_num
%   (3) learning_rate_feature_weight
%
% 优化指标：
%   NMI 和 ARI
%
% 实验结果：
%   保存在 experimental_result.mat 中
%
% 设计目标：
%   1. 实验可重复
%   2. 避免非法搜索空间
%   3. 避免 BayesOpt 参数冲突
%   4. 自动恢复实验
% =========================================================


%% ======================= 路径初始化 =======================
initialize_path();


%% ================== 存储实验变量初始化 ====================
dataset_num   = 18;     % 数据集数量
parameter_num = 3;      % 超参数数量
performance = load_or_init_performance(dataset_num, parameter_num);


%% ======================= 实验部分 =========================
for no = 1:dataset_num

    fprintf('\n=================================================\n');
    fprintf('正在计算第 %d 个数据集\n', no);
    fprintf('=================================================\n');

    % ------------------ 载入数据 --------------------------
    file_name = ['D', num2str(no), '.mat'];
    load(file_name);
    [instance_num, feature_num] = size(data);
    fprintf('样本数: %d  特征数: %d\n', instance_num, feature_num);


    % ------------------ 数据归一化 ------------------------
    % Min-Max 归一化
    % 先缓存 min 和 max 避免重复计算
    data_min = min(data, [], 1); 
    data_max = max(data, [], 1);
    range = data_max - data_min;
    range(range == 0) = 1;
    data = (data - data_min) ./ range;


    % ---------------- 评估预算设置 ------------------------
    % 根据数据规模动态调整 BayesOpt 预算

    if instance_num <= 1000
        max_objective_evaluation = 200;
    elseif instance_num < 5000
        max_objective_evaluation = 170;
    elseif instance_num < 10000
        max_objective_evaluation = 140;
    else
        max_objective_evaluation = 100;
    end

    % NumSeedPoints 不应大于 MaxObjectiveEvaluations
    num_seed_points = min(10, max_objective_evaluation);


    % ------------------ 搜索空间定义 ----------------------

    % power_num 搜索范围
    % 需要避免出现负数或非法区间

    lower_bound = class_num;
    upper_bound = min(fix(instance_num/5), 500);

    % 防止 lower > upper
    if lower_bound > upper_bound
        lower_bound = upper_bound;
    end

    wgb_num_range = [lower_bound, upper_bound];

    % pojg 参数（log-space）
    hyper_parameter_0 = optimizableVariable( ...
        'pojg', [1e-2, 1e2], ...
        'Type', 'real', ...
        'Transform', 'log');

    % wgb_num 参数
    hyper_parameter_1 = optimizableVariable( ...
        'wgb_num', wgb_num_range, ...
        'Type', 'integer');

    % learning_rate 参数
    hyper_parameter_2 = optimizableVariable( ...
        'learning_rate', [1e-2, 1], ...
        'Type', 'real', ...
        'Transform', 'log');

    hyper_parameter = [hyper_parameter_0, hyper_parameter_1, hyper_parameter_2];


    % =================== NMI 指标优化 =====================
    fprintf('\n---- 开始优化 NMI ----\n');

    fun_nmi = @(x) dpc_wgb_repeat(data, label, class_num, ...
        x.wgb_num, wgbOptions("hyperparameter_pojg", x.pojg,...
        "learning_rate_feature_weight", x.learning_rate), ...
        20, 'nmi');

    results_nmi = bayesopt( ...
        fun_nmi, hyper_parameter, ...
        'Verbose', 1, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', max_objective_evaluation, ...
        'NumSeedPoints', num_seed_points, ...
        'UseParallel', false, ...
        'PlotFcn', [],...
        'IsObjectiveDeterministic', true);

    % bayesopt 默认最小化目标函数
    performance.nmi(no) = -results_nmi.MinObjective;

    % 保存最优参数
    performance.nmi_parameter(no, :) = ...
        [results_nmi.XAtMinObjective.pojg, results_nmi.XAtMinObjective.wgb_num, results_nmi.XAtMinObjective.learning_rate];

    fprintf('NMI 最优值: %.6f\n', performance.nmi(no));


    % ---------------- 保存中间结果 ------------------------
    save('experimental_result.mat', "performance", "-v7.3");


    % =================== ARI 指标优化 =====================
    fprintf('\n---- 开始优化 ARI ----\n');

    fun_ari = @(x) dpc_wgb_repeat(data, label, class_num, ...
        x.wgb_num, wgbOptions("hyperparameter_pojg", x.pojg,...
        "learning_rate_feature_weight", x.learning_rate), ...
        20, 'ari');

    results_ari = bayesopt( ...
        fun_ari, hyper_parameter, ...
        'Verbose', 1, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', max_objective_evaluation, ...
        'NumSeedPoints', num_seed_points, ...
        'UseParallel', false, ...
        'PlotFcn', [],...
        'IsObjectiveDeterministic', true);

    % bayesopt 默认最小化目标函数
    performance.ari(no) = -results_ari.MinObjective;

    % 保存最优参数
    performance.ari_parameter(no, :) = ...
        [results_ari.XAtMinObjective.pojg, results_ari.XAtMinObjective.wgb_num, results_ari.XAtMinObjective.learning_rate];

    fprintf('ARI 最优值: %.6f\n', performance.ari(no));


    % ---------------- 保存中间结果 ------------------------
    save('experimental_result.mat', "performance", "-v7.3");

end


fprintf('\n所有实验完成\n');



%% 函数定义部分

function [] = initialize_path()
% 初始化函数路径与数据集路径

addpath(fullfile(pwd, 'functions'));
addpath(fullfile(pwd, '..', '..', 'datasets'));

end



function performance = load_or_init_performance(dataset_num, parameter_num)
% 若 experimental_result.mat 已存在则加载
% 否则初始化 performance 结构体

if exist('experimental_result.mat', 'file') == 2
    loaded_data = load('experimental_result.mat');
    if isfield(loaded_data, 'performance')
        performance = loaded_data.performance;
    else
        error('experimental_result.mat 不包含变量 "performance"');
    end
else
    performance = struct( ...
        'nmi', zeros(dataset_num, 1), ...
        'nmi_parameter', zeros(dataset_num, parameter_num), ...
        'ari', zeros(dataset_num, 1), ...
        'ari_parameter', zeros(dataset_num, parameter_num));
end

end