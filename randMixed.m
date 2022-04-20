function [Q,err,k] = randMixed(A,tol,p,theta,bsize,q)
%RANDOMIZED_MP Mixed Precision Randomized subspace iteration 
%
%			   A - m-by-n input matrix
%			   Global tolerance - tolerance for half precision
%
%
%			   Q -- Estimation of range of A

[m,n] = size(A);
if nargin < 6, q = 0, end
if nargin < 5
   q = 0;
   bSize = min(max(floor(0.1*m), 5), n);
end

k = zeros(length(p), 1);

err = norm(A, 'fro');
%calculate theta based off dimensions with a scaling
if q == 0
    alpha = 1/(theta*sqrt(m*n*bsize));
else
    alpha = 1/(theta*sqrt(m)*bsize);
end
    
tols = [alpha*tol/p(2) alpha*tol/p(3) tol];
Q = [];

its = 0;

while (err(end,1) >= tol && its < length(p))
    idx = find(err(end, 1) > tols, 1, 'first');
    its = its + 1;
    if idx == 1
        [Qd, Rd, errd, kd] = randDouble(A, tols(1), bsize, q);
        Q = [Q Qd];
        A = Rd;
        err(its+1, 1) = errd;
        k(1) = kd;
    elseif idx == 2
        [Qs, Rs, errs, ks] = randSingle(A, tols(2), bsize, q);        
        Q = [Q Qs];
        A = Rs;
        k(2) = ks;
        err(its+1, 1) = errs;
    elseif idx == 3
        [Qh, Rh, errh, kh] = randHalf(A, tols(3), bsize, q);
        A = Rh;
        Q = [Q Qh];
        k(3) = kh;
        err(its+1, 1) = errh;
    end
end
end
