function [Q,B,A,err,k] = randHalf(A,tol,bsize,q,maxDim,Q,B, nrmA)
%RANDOMIZED_HALF Randomized subspace iteration in fp16
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
% Require chop  in path
    [m,n] = size(A);
    
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
    fp.format = 'h'; fp.explim = 1;
    chop([], fp)
    [~,~,~,xmax] = float_params('h');
    %MaxA = max(abs(A(:)));

    %D = diag(1./vecnorm(A,Inf));
    %S = chop(A*D);
    %A = S;
    
    while (err(end,1) >= tol && ((a*bsize) <= maxDim))
        Om = randn(n, bsize);
        Y = hgemm(A, Om, fp);
        [Qi,~] = house_qr_lp(Y,0,fp);
        % power scheme
        for i = 1:(2*q)
            if (mod(i,2) == 1)
                Yi = hgemm(A', Qi, fp);
                [Qi,~] = house_qr_lp(Yi, 0, fp);
            elseif (mod(i,2) == 0)
                Yi = hgemm(A, Qi, fp);
                [Qi,~] = house_qr_lp(Yi, 0, fp);
            end
            
        end
        %re orthog if desired        
        [Qi, ~] = qr(Qi - (Q*(Q'*Qi)), 0);

        Bi = hgemm(Qi', A, fp);        
        A = chop(A - hgemm(Qi, Bi, fp));
        Q = [Q Qi];
        B = [B; Bi];
        err(a,1) = norm(A,'fro')/nrmA;
        a = a+1;
    end
    k = a - 1;        
    err = err(end, 1);
end

