clc
clear all
data = csvread('CUJcconf.csv');
plot(data(3, 2:41), '-o', 'MarkerSize', 6, 'Color', '#FD6D5A', 'LineWidth', 1.5);
hold on
plot(data(4, 2:41), '-^', 'MarkerSize', 6, 'Color', '#FEB40B',  'LineWidth', 1.5);
hold on
plot(data(5, 2:41), '-d', 'MarkerSize', 6, 'Color', '#6DC354',  'LineWidth', 1.5);
hold on
plot(data(6, 2:41), '-v', 'MarkerSize', 6, 'Color', '#994487',  'LineWidth', 1.5);
hold on
plot(data(7, 2:41), '-s', 'MarkerSize', 6, 'Color', '#518CD8',  'LineWidth', 1.5);
hold on
%plot(data(:, 4), '-^', 'MarkerSize', 12, 'Color',[0.18039 0.5451 0.34118], 'LineWidth', 3.2)
%hold on
%plot(data(:, 5), '-*', 'MarkerSize', 12, 'Color',[0.8549 0.64706 0.12549], 'LineWidth', 3.2)
%hold on

g=legend('$c_{conf}=0.3$','$c_{conf}=0.4$','$c_{conf}=0.5$','$c_{conf}=0.6$','$c_{conf}=0.7$');
h = xlabel('Time Slots (0.5s)','FontSize',16); ylabel('Cumulative Utility of JAs','FontSize',16);
%xlabel('Block interval T^I (s)','FontSize',15); ylabel('Total reward','FontSize',15);
k=set(gca,'xtick',0:5:40,'XTickLabel',{'0','25','50','75','100','125','150','175','200'});
set(g,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(h,'FontSize',16,'FontWeight','normal')
