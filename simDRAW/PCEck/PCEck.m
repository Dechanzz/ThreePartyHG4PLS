clc
clear all
data = csvread('EVPowerCostCk.csv');
plot(data(3, 2:6), '-o', 'MarkerSize', 8, 'Color', '#FD6D5A', 'LineWidth', 2);
hold on
plot(data(4, 2:6), '-^', 'MarkerSize', 8, 'Color', '#FEB40B',  'LineWidth', 2);
hold on
% plot(data(5, 2:6), '-d', 'MarkerSize', 8, 'Color', '#6DC354',  'LineWidth', 2);
% hold on
% g=legend('Proposed Approach', '$R^E$-Greedy', '$R^E$-EE');
g=legend('Proposed Solution', '$R^E$-Max');
hx = xlabel('Activation Cost of Single EV ($c_k$)','FontSize',16); 
hy = ylabel('Total Cost of All EVs','FontSize',16);
%xlabel('Block interval T^I (s)','FontSize',15); ylabel('Total reward','FontSize',15);
k=set(gca,'xtick',[1 2 3 4 5],'XTickLabel',{'0.1','0.2','0.3','0.4','0.5'});
set(g,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hx,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hy,'Interpreter','latex','FontSize',16,'FontWeight','normal')
