%% 使用等价的FISTA方法和等步长梯度下降法并比较结果
% 参数说明：
% X：每次迭代时自变量 X 的值 (X表示FISTA方法，X1表示梯度下降)
% K：迭代次数
% F: 每次迭代的目标函数值
% dF_norm: 每次迭代的梯度范数


clc; clear; close all;

A = [0 2; -2 7; 2 4];
b = [7; 8; 9];

% 调用FISTA_equal方法
[K, x_min, f_min, X, F, dF_norm] = FISTA_equal_method([-10; -1], 1e-3,A,b);
disp(['利用FISTA_equal方法: 迭代次数 = ', num2str(K), ', 最优值点 x1 = ', num2str(x_min(1)), ', x2 = ', num2str(x_min(2)), ', 最小值 = ', num2str(f_min)]);

% 调用梯度下降法
[K1, x_min1, f_min1, X1, F1, dF_norm1] = DG_method([-10; -1], 1e-3,A,b);
disp(['利用梯度下降法：迭代次数 = ', num2str(K1), ', 最优值点 x1 = ', num2str(x_min1(1)), ', x2 = ', num2str(x_min1(2)), ', 最小值 = ', num2str(f_min1)]);

% 绘制图形
Go_plot( X, F, dF_norm, X1, F1, dF_norm1,A,b);

% FISTA_equal 方法
function [K, x_min, f_min, X, F, dF_norm] = FISTA_equal_method(start, eps,A,b)

    
    % 初始化记录数据
    max_iter = 1000;
    X = zeros(2, max_iter);
    F = zeros(1, max_iter);
    dF = zeros(2, max_iter);
    dF_norm = zeros(1, max_iter);
    
    % 初始值设置
    X(:, 1) = start;
    F(1) = f(X(:, 1),A,b);
    dF(:, 1) = df(X(:, 1),A,b);
    dF_norm(1) = norm(dF(:, 1));
    
    L = norm(A' * A, 2);
    alpha = 1 / L;

    % 先跑一下
    
    r = 2/3;
    X(:, 2) = X(:, 1) - alpha * dF(:, 1);
    F(2) = f(X(:, 2),A,b);
    dF(:, 2) = df(X(:, 2),A,b);
    dF_norm(2) = norm(dF(:, 2));
    v = X(:, 1) + (1/r)*(X(:, 2) - X(:, 1));
    k = 2;
    while dF_norm(k) >= eps && k < max_iter
        % 更新迭代点
        r = 2/(k+2); %r3

        Y = (1 - r) * X(:, k) + r * v;    %y3
        dF_Y = df(Y,A,b);                  %dy3
        X(:, k + 1) = Y - alpha * dF_Y;    %x3
        F(k + 1) = f(X(:, k + 1),A,b);
        dF(:, k + 1) = df(X(:, k + 1),A,b);
        dF_norm(k + 1) = norm(dF(:, k + 1));
        
        v = X(:, k) + (1/r) * (X(:, k + 1) - X(:, k));    %v3
        k = k + 1;

    end

    K = k;
    x_min = X(:, k);
    f_min = F(k);
    X = X(:, 1:K);
end

% 梯度下降法
function [K, x_min, f_min, X, F, dF_norm] = DG_method(start, eps,A,b)

    
    % 初始化记录数据
    max_iter = 1000;
    X = zeros(2, max_iter);
    F = zeros(1, max_iter);
    dF = zeros(2, max_iter);
    dF_norm = zeros(1, max_iter);
    
    % 初始值设置
    X(:, 1) = start;
    F(1) = f(X(:, 1),A,b);
    dF(:, 1) = df(X(:, 1),A,b);
    dF_norm(1) = norm(dF(:, 1));
    
    L = norm(A' * A, 2);
    alpha = 0.8 / L;

    k = 1;
    while dF_norm(k) >= eps && k < max_iter
        % 更新迭代点
        X(:, k + 1) = X(:, k) - alpha * dF(:, k);
        F(k + 1) = f(X(:, k + 1),A,b);
        dF(:, k + 1) = df(X(:, k + 1),A,b);
        dF_norm(k + 1) = norm(dF(:, k + 1));
        
        k = k + 1;
    end

    K = k;
    x_min = X(:, k);
    f_min = F(k);
    X = X(:, 1:K);
end

% 目标函数
function val = f(x,A,b)

    val = 0.5 * (norm(A * x - b, 2))^2;
end

% 梯度函数
function grad = df(x,A,b)

    grad = A' * A * x - A' * b;
end

% 绘制图形函数
function Go_plot( X1, F1, dF_norm1, X2, F2, dF_norm2,A,b)
    % 获取精确解用于参考

    x_optimal = A \ b;
    f_optimal = 0.5*norm(A*x_optimal - b)^2;
    
    % 创建等高线背景
    x1_range = linspace(min([X1(1,:), X2(1,:)])-1, max([X1(1,:), X2(1,:)])+1, 80);
    x2_range = linspace(min([X1(2,:), X2(2,:)])-1, max([X1(2,:), X2(2,:)])+1, 80);
    [XX, YY] = meshgrid(x1_range, x2_range);
    ZZ = arrayfun(@(x,y) 0.5*norm(A*[x;y]-b)^2, XX, YY);
    
    % 配色方案
    color_fista = [0.8500 0.3250 0.0980]; % 橙色
    color_sd = [0 0.4470 0.7410];        % 蓝色
    opt_marker = [1 0 0];                % 红色
    
    %% 图1: 迭代路径 (带等高线)
    figure('Position', [100 100 900 400])
    subplot(1,2,1)
    
    % 绘制等高线
    contour(XX, YY, ZZ, 15, 'LineWidth', 0.8, 'LineColor', [0.5 0.5 0.5]);
    hold on
    
    % 绘制优化路径
    p1 = plot(X1(1,:), X1(2,:), 'o-', 'Color', color_fista,...
        'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', color_fista);
     p2 = plot(X2(1,:), X2(2,:), 's-', 'Color', color_sd,...
          'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', color_sd);
    
    % 标注最优解
    p3 = plot(x_optimal(1), x_optimal(2), 'p', 'Color', opt_marker,...
        'MarkerSize', 15, 'LineWidth', 2, 'MarkerFaceColor', opt_marker);
    
    % 图形美化
    axis tight equal
    grid on
    box on
    set(gca, 'FontSize', 11, 'LineWidth', 1.2)
    xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold')
    ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold')
    title('优化路径对比', 'FontSize', 14, 'FontWeight', 'bold')
    legend([p1 p2 p3], {'FISTA','梯度下降法','最优解'},...
        'Location', 'best', 'FontSize', 10)
    colormap(parula) % 设置等高线配色

    %% 图2: 梯度范数下降曲线
    subplot(1,2,2)
    
    % 绘制曲线
    semilogy(dF_norm1, 'o-', 'Color', color_fista,...
        'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', color_fista)
    hold on
    semilogy(dF_norm2, 's-', 'Color', color_sd,...
        'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', color_sd)
    
    % 图形美化
    grid on
    box on
    set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'YScale', 'log')
    xlabel('迭代次数', 'FontSize', 12, 'FontWeight', 'bold')
    ylabel('梯度范数', 'FontSize', 12, 'FontWeight', 'bold')
    title('梯度下降过程', 'FontSize', 14, 'FontWeight', 'bold')
    legend({'FISTA','梯度下降法'}, 'Location', 'best', 'FontSize', 10)
    ylim([1e-4, max([dF_norm1, dF_norm2])])
    
    %% 图3: 目标函数值变化曲线
    figure('Position', [100 100 500 400])
    
    % 计算与最优值的差值
    F1_diff = F1 - f_optimal;
    F2_diff = F2 - f_optimal;
    
    % 绘制曲线
    semilogy(F1_diff, 'o-', 'Color', color_fista,...
        'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', color_fista)
    hold on
    semilogy(F2_diff, 's-', 'Color', color_sd,...
        'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', color_sd)
    
    % 图形美化
    grid on
    box on
    set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'YScale', 'log')
    xlabel('迭代次数', 'FontSize', 12, 'FontWeight', 'bold')
    ylabel('f(x) - f^*', 'FontSize', 12, 'FontWeight', 'bold')
    title('目标函数收敛过程', 'FontSize', 14, 'FontWeight', 'bold')
    legend({'FISTA','梯度下降法'}, 'Location', 'best', 'FontSize', 10)
    ylim([1e-6, max([F1_diff, F2_diff])])
end
