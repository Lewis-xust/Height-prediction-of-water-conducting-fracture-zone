%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('shuffled_combined_150_samples.xlsx');
rng(42);%随机种子

%%  数据分析
num_size = 0.8;                              % 训练集占数据集比例
outdim = 1;                                  % 最后三列为输出
num_samples = size(res, 1);                  % 样本个数
%res = res(randperm(num_samples), :);         % 打乱数据集
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  创建网络
net = newff(p_train, t_train, 10);  % [10 10] 隐含层即可设置2层 每层神经元个数为10 

%%  设置训练参数
net.trainParam.epochs = 1000;     % 迭代次数 
net.trainParam.goal = 1e-6;       % 误差阈值
net.trainParam.lr = 0.01;         % 学习率

%%  训练网络
net= train(net, p_train, t_train);

%%  仿真测试
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  绘图
figure    % 创建一个新的图形窗口
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b--^', 'LineWidth', 1)   %  'r-*': 红色实线带星号标记，'b-o': 蓝色实线带圆圈标记
legend('Actual value', 'Predicted value')
xlabel('Sample number')
ylabel('Height of WCFZ/(m)')
%string = {'训练集训练结果'; ['RMSE=' num2str(error1)]};
%title(string)
xlim([1, M])
box on     % 去掉图表边框的上方和右侧部分
grid on;
ax = gca;
% 设置一套灰色的半透明虚线网格
ax.GridLineStyle = '--'; % 虚线
ax.GridColor = [0.7, 0.7, 0.7]; % 浅灰色
ax.GridAlpha = 0.5; % 半透明
ax.LineWidth = 1; % 线宽稍粗

figure    % 创建一个新的图形窗口
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b--^', 'LineWidth', 1)   %  'r-*': 红色实线带星号标记，'b-o': 蓝色实线带圆圈标记
legend('Actual value', 'Predicted value')
xlabel('Sample number')
ylabel('Height of WCFZ/(m)')
%string = {'训练集训练结果'; ['RMSE=' num2str(error1)]};
%title(string)
xlim([1, N])
box on     % 去掉图表边框的上方和右侧部分
grid on;
ax = gca;
% 设置一套灰色的半透明虚线网格
ax.GridLineStyle = '--'; % 虚线
ax.GridColor = [0.7, 0.7, 0.7]; % 浅灰色
ax.GridAlpha = 0.5; % 半透明
ax.LineWidth = 1; % 线宽稍粗

% figure
% plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {strcat('测试集预测结果对比：', ['RMSE=' num2str(error2)])};
% title(string)
% xlim([1, N])
% grid

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

% 平均绝对百分比误差计算
mape1 = 100 * mean(abs((T_sim1 - T_train) ./ T_train));
mape2 = 100 * mean(abs((T_sim2 - T_test) ./ T_test));

disp(['训练集数据的MAPE为：', num2str(mape1), '%'])
disp(['测试集数据的MAPE为：', num2str(mape2), '%'])

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])

%  RMSE
disp(['训练集数据的RMSE为：', num2str(error1)])
disp(['测试集数据的RMSE为：', num2str(error2)])

%%  绘制散点图
sz = 45;
c = [0.9, 0, 0];
figure
% 更详细的方块标记设置
scatter(T_train, T_sim1, sz, c, 'filled', 'Marker', 's','MarkerEdgeColor', 'k')
hold on
plot(xlim, ylim, '-k',LineWidth=1.2)
xlabel('Actual value');
ylabel('Predicted value');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('BP model')
box on  % 去掉图表边框的上方和右侧部分
grid on;
ax = gca;
% 设置一套灰色的半透明虚线网格
ax.GridLineStyle = '--'; % 虚线
ax.GridColor = [0.7, 0.7, 0.7]; % 浅灰色
ax.GridAlpha = 0.5; % 半透明
ax.LineWidth = 1.2; % 线宽稍粗

figure
% 更详细的方块标记设置
scatter(T_test, T_sim2, sz, c, 'filled', 'Marker', 's','MarkerEdgeColor', 'k')
hold on
plot(xlim, ylim, '-k',LineWidth=1.2)
xlabel('Actual value');
ylabel('Predicted value');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('BP model')
box on  % 去掉图表边框的上方和右侧部分
grid on;
ax = gca;
% 设置一套灰色的半透明虚线网格
ax.GridLineStyle = '--'; % 虚线
ax.GridColor = [0.7, 0.7, 0.7]; % 浅灰色
ax.GridAlpha = 0.5; % 半透明
ax.LineWidth = 1.2; % 线宽稍粗


% figure
% scatter(T_test, T_sim2, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('测试集真实值');
% ylabel('测试集预测值');
% xlim([min(T_test) max(T_test)])
% ylim([min(T_sim2) max(T_sim2)])
% title('测试集预测值 vs. 测试集真实值')