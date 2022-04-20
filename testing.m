% Experiments to generate numerical results in Section 3.3
clear all

A = generateMatrix2(5e2,5e2,100,2,1);
%A = gallery('randsvd', 5e2, 10^10);

prec = [2^-53 2^-24 2^-11];

tol = 1e-4;
bsize = 10;
q = 1;

theta = 0.1;

[Qm, errm, km] = randMixed(A, tol, prec, theta, bsize, q);
[Qd, R, errd, kd] = randDouble(A, tol, bsize, q);

fprintf('---Double results---\n')
fprintf('Its: %d\nError: %e\n\n', kd, errd)

fprintf('---Mixed results\n')
fprintf('D Its: %d S Its: %d H Its: %d\n', km(1), km(2), km(3))
fprintf('Error: %e\n', errm(end))

function A = generateMatrix1(m,n,r,xi)
% low rank + noise

A = diag([ones(r,1); zeros(n-r, 1)]);
G = randn(n);
C = (1/n)*xi*(G*G');
A = A + C;

end

function A = generateMatrix2(m,n,r,d,xi)

% left singular vectors
U = randn(m);
[U,~] = qr(U);

% right singular vectors
V = randn(n);
[V,~] = qr(V);

s1 = xi*ones(r,1);
s2 = [2:1:(n-r+1)]';
s2 = s2.^(-d); 
%U = eye(m); V = eye(m);
S = [diag([s1;s2]);zeros(m-n,n)];
A = U*S*V';

end
