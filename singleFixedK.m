%Experiments to generate numerical results in Section 2.3

%R is effective rank of matrix - fixed throughout 
R = 20
% xi controls size of largest singular values in matrix 1
phi = 1e8;
% phi controls noise in matrix 2
xi = 1e-4;
u = 2^-24;
type = 1; % choose which matrix type
points = 15;
nValues = round(logspace(2, 3, points));
errors = zeros(points, 1);
powerErrors = zeros(points, 1);
exactBound = zeros(points, 1);
worstBound = zeros(points, 1);
probBound = zeros(points, 1);

for i = 1:points   
    p = 1; t = 1; mu = 1;
    n = nValues(i);
    m = n;
    k = floor(sqrt(n));
    Om = single(randn(n, k + p));
    if type == 1
        A = generateMatrix1(m, n, R, xi);
    else
        A = generateMatrix2(m, n, R, 3, phi);
    end
   

    s = svd(A);
    a = 1 + t*sqrt(12*k/p);
    b = mu*t*exp(1)*sqrt(k + p)/(p+1);
    
    exactBound(i) = (a*norm(s(k+1:end), 2) + b*s(k+1))/norm(A, 'fro');
    worstBound(i) = exactBound(i) + m*sqrt(n)*k*u;
    probBound(i) = exactBound(i) + sqrt(m*n*k)*u;
    A = single(A);
    Y = A*Om;
    
    [Q, ~] = qr(Y, 0);
    B = Q'*A;
    [U, Sig, V] = svd(B, 'econ');
    U = Q*U;
    A = double(A);
    U = double(U); Sig = double(Sig); V = double(V);
    errors(i) = norm(A - U*Sig*V', 'fro')/norm(A, 'fro');
    
    fprintf('%d/%d\n', i, points)
end

ms = 8;
lw = 3;
fs = 16;

loglog(nValues, worstBound, '-', 'LineWidth', lw)
hold on
loglog(nValues, probBound, '-', 'LineWidth', lw)
loglog(nValues, exactBound, '--', 'LineWidth', lw)
loglog(nValues, errors, 'o-', 'LineWidth', lw, 'MarkerSize', ms)
yline(u, '--')
xlabel('$n$', 'Interpreter', 'Latex', 'FontSize', fs)
h = legend({'Worst bound', 'Prob bound', 'Exact bound', 'Error'}, 'Interpreter', 'Latex', 'FontSize', 16, 'Location', 'west')
set(gca, 'XTick', round(logspace(2, 3, 5)))
if type == 2 % slight shifting
    pos = get(h, 'Position')
    pos(2) = pos(2) + 0.02
    set(h, 'Position', pos)
end
hold off

if type == 1
    exportgraphics(gca, 'singleType1.pdf')
else
    exportgraphics(gca, 'singleType2.pdf')
end

function A = generateMatrix1(m,n,r,xi)
% low rank + noise

A = diag([1e4*ones(r,1); zeros(n-r, 1)]);
G = randn(n);
C = (1/n)*xi*(G*G');
A = A + C;

end


function A = generateMatrix2(m,n,r,d,phi)

% left singular vectors
U = randn(m);
[U,~] = qr(U);

% right singular vectors
V = randn(n);
[V,~] = qr(V);

s1 = phi*ones(r,1);
s2 = [2:1:(n-r+1)]';
s2 = s2.^(-d); 
%U = eye(m); V = eye(m);
S = [diag([s1;s2]);zeros(m-n,n)];
A = U*S*V';

end


