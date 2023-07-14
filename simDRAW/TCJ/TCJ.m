clc
clear all
data = csvread('TCJ.csv');
plot(data(3, 2:7), '-o', 'MarkerSize', 8, 'Color', '#FD6D5A', 'LineWidth', 2);
hold on
plot(data(4, 2:7), '-^', 'MarkerSize', 8, 'Color', '#FEB40B',  'LineWidth', 2);
hold on
g=legend('Proposed Solution', '$V_{\mathcal J}-Max$');
hy = ylabel('Total Cost of All JAs','FontSize',16); 
hx = xlabel('Unit Jamming Power Cost of Single JA ($\eta_j$)','FontSize',16);
k=set(gca,'xtick',[1 2 3 4 5 6],'XTickLabel',{'5','7','9','11','13','15'});
set(g,'Interpreter','latex','FontSize',15,'FontWeight','normal')
set(hx,'Interpreter','latex','FontSize',16,'FontWeight','normal')
set(hy,'Interpreter','latex','FontSize',16,'FontWeight','normal')
