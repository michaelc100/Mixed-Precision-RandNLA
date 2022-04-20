function [Q,A,err,k] = randHalf(A,tol,bsize,q,maxDim)
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
    elseif nargin == 3
        maxDim = n; 
        q = 0;
    end

    err = norm(A,'fro');
    Q = zeros(m, 0); a = 1;
    fp.format = 'h';
    chop([], fp)
    [~,~,~,xmax] = float_params('h');
    %MaxA = max(abs(A(:)));

    %if MaxA >= (0.1*xmax)
    %    D = diag(1./vecnorm(A,Inf));
    %    S = chop(A*D);
    %    A = S;
    %end
    
    while (err(end,1) >= tol && ((a*bsize) <= maxDim))
        Om = randn(n, bsize);
        Y = hgemm(A, Om);
        [Qi,~] = house_qr_lp(Y,0,fp);
        % power scheme
        for i = 1:(2*q)
            if (mod(i,2) == 1)
                Yi = hgemm(A', Qi);
                [Qi,~] = house_qr_lp(Yi, 0, fp);
            elseif (mod(i,2) == 0)
                Yi = hgemm(A, Qi);
                [Qi,~] = house_qr_lp(Yi, 0, fp);
            end
            
        end
        %re orthog if desired        
        %[Qi, ~] = qr(Qi - (Q*(Q'*Qi)), 0);

        temp = hgemm(Qi', A);
        A = chop(A - hgemm(Qi, temp));
        Q = [Q Qi];
        err(a,1) = norm(A,'fro');
        a = a+1;
    end
    k = a - 1;        
    err = err(end, 1);
end

