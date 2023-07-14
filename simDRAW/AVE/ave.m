clc
clear all
data = csvread('averageUtility.csv');
% 创建一个新的图形窗口
figure;

% 设置整张图的标题
%sgtitle('整张图的标题');

% 定义子图的位置和尺寸
sub1_position = [0.1, 0.57, 0.8, 0.37];  % [left, bottom, width, height]
sub2_position = [0.1, 0.1, 0.8, 0.37];   % [left, bottom, width, height]

% 创建上方子图
subplot('Position', sub1_position);
% 在上方子图中绘制内容
plot(data(2, 2:21), '-o', 'MarkerSize', 8, 'Color', '#FD6D5A', 'LineWidth', 2.1);
hold on
plot(data(3, 2:21), '-s', 'MarkerSize', 8, 'Color', '#FEB40B',  'LineWidth', 2.1);
hold on
plot(data(4, 2:21), '-^', 'MarkerSize', 8, 'Color','#6DC354', 'LineWidth', 2.1)
hold on
g0=legend('EVs', 'LUs', 'JAs');
hy0=ylabel('Average Utility','FontSize',16);
t1 = title('Discretized Power Settings');

% 创建下方子图
subplot('Position', sub2_position);
% 在下方子图中绘制内容
% aaa = [8.68854, 8.22837, 9.02321, 8.20746 ,7.39170,7.76820,6.72237,5.17453,6.13670,5.36278,5.04903,5.86478,3.85678,4.71436,4.29603,4.40061,4.02411,3.52211,3.60578,3.71036];
% bbb = rescale(aaa, -100, 30);
% ccc = [2.2105, 3.1790, 3.7103, 4.0737, 4.9255, 3.8383, 4.5916, 4.6152, 6.0278, 6.7106, 7.0402, 7.4875, 7.3933, 8.0055, 7.9584, 8.2857, 8.0761, 8.4528, 8.3115, 8.2880];
% ddd = rescale(ccc, -180, 20);
% eee = [10.6140,10.4887,10.1953,9.5173,9.6133,9.6426,9.7366,9.2039,8.7652,8.1746,7.4491,7.5431,7.1357,7.6058,7.3864,7.5744,7.4491,7.3237
% ];
% fff = rescale(eee, 50,110);
plot(data(5, 2:21), '-o', 'MarkerSize', 8, 'Color', '#FD6D5A', 'LineWidth', 2.1);
hold on
plot(data(6, 2:21), '-s', 'MarkerSize', 8, 'Color', '#FEB40B',  'LineWidth', 2.1);
hold on
plot(data(7, 2:21), '-^', 'MarkerSize', 8, 'Color','#6DC354', 'LineWidth', 2.1)
hold on
%plot(data(:, 5), '-*', 'MarkerSize', 12, 'Color',[0.8549 0.64706 0.12549], 'LineWidth', 3.2)
%hold on

%g2=legend('EVs', 'LUs', 'JAs');
hx2=xlabel('Training Rounds $(\times 10^4)$','FontSize',16); 
%hy2=ylabel('Average Utility','FontSize',12);
t2 = title('Continuous Power Settings');

set(g0,'Interpreter','latex','FontSize',10,'FontWeight','normal')
set(hy0,'Interpreter','latex','FontSize',12,'FontWeight','normal')
%set(g2,'Interpreter','latex','FontSize',12,'FontWeight','normal')
set(hx2,'Interpreter','latex','FontSize',12,'FontWeight','normal')
%set(hy2,'Interpreter','latex','FontSize',12,'FontWeight','normal')
set(t1,'Interpreter','latex','FontSize',10,'FontWeight','normal')
set(t2,'Interpreter','latex','FontSize',10,'FontWeight','normal')
