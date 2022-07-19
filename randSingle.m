function [Q,B,A,err,k] = randSingle(A,tol,bsize,q,maxDim,Q,B, nrmA)
%RANDOMIZED_SINGLE Randomized subspace iteration in fp32
%
%			   A - m-by-n input matrix
%    
%                          bsize - block size
%			   tol - $\|A-QQ^TA\| \leq tol$
%			   q -- Number of power iterations. default is 0;
%			   MaxDim -- Max subspace dimension
%                          Qprev - previously computed Q, only used in randMixed impl.
    
%			   Q -- approximate subspace
%                          err -- Vector approximation errors
%                          k -- number of iterations
    
    [m,n] = size(A);
    
    if ~isa(A, 'single')
        A = single(A);
    end
    
    if nargin == 4
        maxDim = n;
        Q = zeros(m, 0);
        B = zeros(0, n);
        nrmA = norm(A, 'fro');
    elseif nargin == 3
        maxDim = n; 
        q = 0;
        Q = zeros(m, 0);
        B = zeros(0, n);
        nrmA = norm(A, 'fro');
    end

    err = norm(A,'fro')/nrmA;
    a = 1;

    while (err(end,1) >= tol && ((a*bsize) <= maxDim))
        Om = single(randn(n, bsize));
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
        Qi = double(Qi);
        [Qi, ~] = qr(Qi - (Q*(Q'*Qi)), 0);
        
        Bi = double(Qi'*A);

        A = A-(Qi*Bi);
        
        Q = [Q Qi];
        B = [B; Bi];
        if ~isa(Q, 'double')
            error('Error. \nQ is not double\n')
        end
        err(a,1) = norm(A,'fro')/nrmA;
        a = a+1;
        
    end
    k = a - 1;        
    err = double(err(end, 1));

    A = double(A);
end

