clc
clear all
data = csvread('VSUE.csv');
plot(data(2, 2:41), '-o', 'MarkerSize', 8, 'Color', '#FD6D5A', 'LineWidth', 2.1);
hold on
plot(data(3, 2:41), '-^', 'MarkerSize', 8, 'Color', '#FEB40B',  'LineWidth', 2.1);
hold on
plot(data(4, 2:41), '-s', 'MarkerSize', 8, 'Color', '#82B0D2',  'LineWidth', 2.1);
hold on
%plot(data(:, 4), '-^', 'MarkerSize', 12, 'Color',[0.18039 0.5451 0.34118], 'LineWidth', 3.2)
%hold on
%plot(data(:, 5), '-*', 'MarkerSize', 12, 'Color',[0.8549 0.64706 0.12549], 'LineWidth', 3.2)
%hold on

g=legend('Proposed Solution', 'LFJ-DRL', 'LFJ-PSO');
hx=xlabel('Time Slots (0.5s)','FontSize',16); 
hy=ylabel('Cumulative Utility of EVs','FontSize',16);


k=set(gca,'xtick',0:5:40,'XTickLabel',{'0','25','50','75','100','125','150','175','200'});
set(g,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hx,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hy,'Interpreter','latex','FontSize',16,'FontWeight','normal')

