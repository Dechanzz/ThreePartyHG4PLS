clc
clear all
data = csvread('VSUJ.csv');
plot(data(2, 2:41), '-o', 'MarkerSize', 8, 'Color', '#FD6D5A', 'LineWidth', 2.1);
hold on
plot(data(3, 2:41), '-^', 'MarkerSize', 8, 'Color', '#FEB40B',  'LineWidth', 2.1);
hold on
plot(data(4, 2:41), '-s', 'MarkerSize', 8, 'Color', '#6DC354', 'LineWidth', 2.1)
hold on
plot(data(5, 2:41), '-*', 'MarkerSize', 8, 'Color', '#999999',  'LineWidth', 2.1);
hold on
plot(data(6, 2:41), '-+', 'MarkerSize', 8, 'Color', '#82B0D2', 'LineWidth', 2.1)
hold on
%plot(data(:, 5), '-*', 'MarkerSize', 12, 'Color',[0.8549 0.64706 0.12549], 'LineWidth', 3.2)
%hold on

g=legend('Proposed Solution', 'EFJ-DRL', 'LFJ-DRL', 'EFJ-PSO', 'LFJ-PSO' );
hx=xlabel('Time Slots (0.5s)','FontSize',16); 
hy=ylabel('Cumulative Utility of JAs','FontSize',16);
k=set(gca,'xtick',0:5:40,'XTickLabel',{'0','25','50','75','100','125','150','175','200'});
set(g,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hx,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hy,'Interpreter','latex','FontSize',16,'FontWeight','normal')
