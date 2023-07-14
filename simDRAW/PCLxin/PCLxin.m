clc
clear all
data = csvread('PowerCostzeta2.csv');
plot(data(3, 2:7), '-o', 'MarkerSize', 8, 'Color', '#FD6D5A', 'LineWidth', 2);
hold on
plot(data(4, 2:7), '-^', 'MarkerSize', 8, 'Color', '#FEB40B',  'LineWidth', 2);
hold on
% plot(data(5, 2:7), '-d', 'MarkerSize', 8, 'Color', '#6DC354',  'LineWidth', 2);
% hold on
plot(data(6, 2:7), '-v', 'MarkerSize', 8, 'Color', '#994487',  'LineWidth', 2);
hold on
% plot(data(7, 2:7), '-s', 'MarkerSize', 8, 'Color', '#518CD8',  'LineWidth', 2);
% hold on
% g=legend('Proposed Approach', '$R^T$-Greedy', '$R^T$-EE', '$R^S$-Greedy', '$R^S$-EE');
g=legend('Proposed Solution', '$R^T$-Max', '$R^S$-Max');
hx = xlabel('Unit Power Cost of Single LU ($\xi_n$)','FontSize',16); 
hy = ylabel('Total Cost of All LUs','FontSize',16);
%xlabel('Block interval T^I (s)','FontSize',15); ylabel('Total reward','FontSize',15);
k=set(gca,'xtick',[1 2 3 4 5 6],'XTickLabel',{'5','7','9','11','13','15'});
set(g,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hx,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hy,'Interpreter','latex','FontSize',16,'FontWeight','normal')
