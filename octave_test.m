
load('debug.mat')
i;
z2 = iw1 * i;
a2 = sigmoid(z2);
% a2
z3 = iw2 * [1; a2];
a3 = sigmoid(z3);

d3 = a3 - y;
d2 = (iw2(:, 2:end)' * d3) .* sigmoidGradient(z2);
% d3
% d2
D2 = d3 * a2';
D1 = d2 * i';

