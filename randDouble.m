function [Q,A,err,k] = randDouble(A,tol,bsize,q,maxDim)
%RANDOMIZED_DOUBLE Randomized subspace iteration in fp64
%
%			   A - m-by-n input matrix
%                          bsize - block size
%			   tol - $\|A-QQ^TA\| \leq tol$
%			   q -- Number of power iterations. default is 0;
%			   MaxDim -- Max subspace dimension
%
%			   Q -- approximate subspace
%                          err -- Vector approximation errors
%                          k -- number of iterations
    [m,n] = size(A);
    
    if nargin == 4
        maxDim = n;
    elseif nargin == 3
        maxDim = n; 
        q = 0;
    end

    err = norm(A,'fro');
    Q = zeros(m, 0); a = 1;

    while (err(end,1) >= tol && ((a*bsize) <= maxDim))
        Om = randn(n, bsize);
        Y = A*Om;
        [Qi,~] = qr(Y,0);
        % power scheme
        for i = 1:(2*q)
            if (mod(i,2) == 1)
                Yi = A'*Qi;
                [Qi,~] = qr(Yi,0);
            elseif (mod(i,2) == 0)
                Yi = A*Qi;
                [Qi,~] = qr(Yi,0);
            end
        end
        %re orthog if desired        
        %[Qi, ~] = qr(Qi - (Q*(Q'*Qi)), 0);
        
        A = A-(Qi*Qi'*A);
        Q = [Q Qi];
        err(a,1) = norm(A,'fro');
        a = a+1;
    end
    k = a - 1;        
    err = err(end, 1);
end

