% Experiments to generate numerical results in Section 3.3
clear all
rng(10)
mat = gallery('randsvd', 5e2, 1e10);
mat(:, :, 2) = generateMatrix2(5e2,5e2,100,2,1);
mat(:, :, 3) = generateMatrix3(5e2,5e2,100,0.1);

tols = [1e-1 1e-3 1e-5 1e-7; 1e-1 1e-2 1e-3 1e-4; 1e-1 1e-3 1e-5 1e-7];
thetas = [0.1 1 10];
bsize = 10;

q = 1;
prec = [2^-53 2^-24 2^-11];

% loop over matrices
for i = 1:3
    A = mat(:, :, i);
    fprintf('-- Matrix %d--\n', i)
    % get double precn. ref
    
    % loop over tols
    for j = 1:length(tols)
        tol = tols(i, j);
        fprintf('\nTol = %e\n\n', tol)
        
        [Qd, Bd, R, errd, kd] = randDouble(A, tol, bsize, q);   
        [Ud, Sd, Vd] = svd(Bd, 'econ');
        Ud = Qd*Ud;
        fprintf('---Double results---\n')
        fprintf('Its: %d\nError: %e\n\n', kd, norm(A - Ud*Sd*Vd', 'fro')/norm(A, 'fro'))

        %loop over thetas
        for k = 1:length(thetas)
            theta = thetas(k);
            
            [Qm, Bm, errm, km] = randMixed(A, tol, prec, theta, bsize, q);
            try
                [Um, Sm, Vm] = svd(Bm, 'econ');
                Um = Qm*Um;
                fprintf('---\nTheta = %d\n', theta)
                fprintf('D Its: %d S Its: %d H Its: %d\n', km(1), km(2), km(3))
                fprintf('Mixed SVD Error: %e\n', norm(A - Um*Sm*Vm', 'fro')/norm(A, 'fro'))
                fprintf('Cost: %f\n', computeCost(5e2, 5e2, bsize, km))
            catch
                fprintf('---\nTheta = %d\n', theta)                
                fprintf('Not converged!\n')
            end            
        end
    end
end

function A = generateMatrix1(m,n,r,xi)
% low rank + noise

A = diag([ones(r,1); zeros(n-r, 1)]);
G = randn(n);
C = (1/n)*xi*(G*G');
A = A + C;

end

function A = generateMatrix2(m,n,r,d,xi)
% polynomial decay
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

function A = generateMatrix3(m, n, r, q)
% exponential decay

s1 = ones(r, 1);
s2 = [1:(n-r)]';
s2 = 10.^(-q*s2);

U = randn(m);
[U,~] = qr(U);

V = randn(n);
[V,~] = qr(V);

S = [diag([s1;s2]);zeros(m-n,n)];

A = U*S*V';
        
end

function cost = computeCost(m, n, b, its)
    % the cost both pay for orthog
        totalIts = sum(its);
        cost = 2*(totalIts^2)*m*(b^2);
        % add the rest of the double precn its
        total = (10*m*n*b + 8*(b^2)*(m - b/3))*totalIts + cost;
    
        halfCost = (1/4)*((10*m*n*b + 8*(b^2)*(m - b/3))*its(3));
        singleCost = (1/2)*((10*m*n*b + 8*(b^2)*(m - b/3))*its(2));
        doubleCost = (10*m*n*b + 8*(b^2)*(m - b/3))*its(1);
        cost = (cost + halfCost + singleCost + doubleCost)/total;
end
