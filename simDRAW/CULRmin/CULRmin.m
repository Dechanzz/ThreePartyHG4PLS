clc
clear all
data = csvread('CULRmin.csv');
plot(data(3, 2:41), '-o', 'MarkerSize', 7, 'Color', '#FD6D5A', 'LineWidth', 1.5);
hold on
plot(data(4, 2:41), '-^', 'MarkerSize', 7, 'Color', '#FEB40B',  'LineWidth', 1.5);
hold on
plot(data(5, 2:41), '-d', 'MarkerSize', 7, 'Color', '#6DC354',  'LineWidth', 1.5);
hold on
plot(data(6, 2:41), '-v', 'MarkerSize', 7, 'Color', '#994487',  'LineWidth', 1.5);
hold on
plot(data(7, 2:41), '-s', 'MarkerSize', 7, 'Color', '#518CD8',  'LineWidth', 1.5);
hold on
%plot(data(:, 4), '-^', 'MarkerSize', 12, 'Color',[0.18039 0.5451 0.34118], 'LineWidth', 3.2)
%hold on
%plot(data(:, 5), '-*', 'MarkerSize', 12, 'Color',[0.8549 0.64706 0.12549], 'LineWidth', 3.2)
%hold on

g=legend('$R^T_{min}=2$', '$R^T_{min}=3$', '$R^T_{min}=4$', '$R^T_{min}=5$', '$R^T_{min}=6$');
h = xlabel('Time Slots (0.5s)','FontSize',18); ylabel('Cumulative Utility of LUs','FontSize',18);
%xlabel('Block interval T^I (s)','FontSize',15); ylabel('Total reward','FontSize',15);
k=set(gca,'xtick',0:5:40,'XTickLabel',{'0','25','50','75','100','125','150','175','200'});
set(g,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(h,'FontSize',16,'FontWeight','normal')
