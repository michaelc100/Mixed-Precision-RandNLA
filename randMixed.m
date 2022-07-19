function [Q, B, err, k] = randMixed(A, tol, precns, theta, b, q)
% RANDMIXED Mixed precision randomized subspace iteration
% Inputs:
% A - input matrix
% tol - target approximation error (absolute, frobenius norm)
% p - sequence of increasing unit roundoffs, corresponding to available precisions
% theta - user chosen parameter which affects precisio switching
% b - block size
% q - no. of power iterations

% Outputs:
% Q - orthogonal factor
% err - vector of errors at each iteration
% k - vector of no. iterations at each precision
    
[m, n] = size(A);
% set some default values

if nargin < 6, q = 0, end

if nargin < 5
    q = 0
    b = min(max(floor(0.1*m), 5), n);
end

%calculate alpha needed to set switching tols
nrmA = norm(A, 'fro');
if q == 0
    alpha = 1/(theta*sqrt(m*n*b));
else
    alpha = 1/(theta*sqrt(m)*b);
end

% initialize k
k = zeros(length(precns), 1);

% pick tolerances that will be used
% drop those that have unit roundoff greater than tol

p = length(find(precns < tol));
precns = precns(1:p);

% set up sequence of tolerances
tols = [];
for i = 1:p-1
    tols(i) = alpha*tol/precns(i+1);
end
tols(p) = tol;

% initial error
err = 1;

Q = zeros(m, 0);
B = zeros(0, n);

% its will count no. of iterations at each precision
its = 0;

while (err(end, 1) >= tol && its < p)
    % find lowest precn that will converge
    idx = find(err(end, 1) > tols, 1, 'first');
    its = its + 1;
    % just assume we'd only use fp64, fp32 and fp16
    if precns(idx) == 2^(-53)
        [Q, B, Rd, errd, kd] = randDouble(A, tols(idx), b, q);
        %Q = [Q Qd];
        A = Rd;
        err = [err; errd];
        k(idx) = kd;        
    elseif precns(idx) == 2^(-24)
        [Q, B, Rs, errs, ks] = randSingle(A, tols(idx), b, q, n, Q, B, nrmA);        
        % if Q not empty reorthog
        %if ~isempty(Q)
        %    [Qs, ~] = qr(Qs - (Q*(Q'*Qs)), 0);
        %end
        
        %Q = [Q Qs];
        A = Rs;
        k(idx) = ks;
        err = [err; errs];
    else
        % it's half precision
        [Q, B, Rh, errh, kh] = randHalf(A, tols(idx), b, q, n, Q, B, nrmA);
        A = Rh;
        % if Q not empty re orthog
        %if ~isempty(Q)
        %    [Qh, ~] = qr(Qh - (Q*(Q'*Qh)), 0);
        %end
        
        %Q = [Q Qh];
        k(idx) = kh;
        err = [err; errh];

    end    
end
end

