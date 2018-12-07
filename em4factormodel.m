function [Z, Lambda, Phi] = em4factormodel(X, q, max_numLoop, tol)
 
    % factor model x = Lambda' * z + N(0, diag(Phi)).
    % x is a p-variate variable.
    % z is a latent q-variate normal random variable.
    % Lambda is the factor loading.
    % Phi is a p-dimension vector of variance. 
    % In matrix notation, X = Z * Lambda + N(0, diag(Phi))
    % X: T * p 
    % Z: T * q
    % Lambda : q * p 
    
    if nargin == 2 
        max_numLoop = 10000;
        tol         = 0.0001;
            
    elseif nargin == 3
        assert(max_numLoop >= 7000);
        tol = 0.0001;
       
    end    
          
    l21_norm = @(x)sum(sqrt(sum(x.^2, 1)));  %L21 norm of matrix.

    [T, p] = size(X);
 
    X      = X - ones(T, T) * X/T;
    
    Z          = zeros(T, q);
    Lambda     = zeros(q, p);
    %Phi        = zeros(p, 1);
    
    %% initial values of EM algorithm given by pca
    [U, S, ~] = svd(X, 'econ');  % X = U * S * V'
    %V = V';
    
    for i = 1: q
        
       Z(:, i)         = U * S(:, i); 
       var_            = var(Z(:, i));
       Z(:, i)         = Z(:, i)/sqrt(var_);
        
    end    

    %fprintf('covariance of initial values of the factors\n')
    %disp(cov(Z));
    
    Lambda = (Z'*Z)\(Z'*X);
    e      = X - Z * Lambda; 
    Phi    = diag(diag(e'*e/T));
    
    Lambda_old = Lambda;
    Phi_old    = Phi;
    para_dist  = tol * 2;
    
    n = max_numLoop;
    while  n >= 1
        
       
       % E step
       % mean_z_on_x : q * T. mean of z conditional on x
       % cov_z_on_x :  q * q. covariance of z conditional on x
       
       mean_z_on_x = Lambda * ((Lambda' * Lambda + Phi)\ (X')); 
       cov_z_on_x  = eye(q) - Lambda * ((Lambda' * Lambda + Phi)\ (Lambda'));
       
       m2_z_on_x   = mean_z_on_x * mean_z_on_x'/T + cov_z_on_x; 
       cross_prod  = mean_z_on_x * X/T;
       
       % M step
      Lambda  = m2_z_on_x \cross_prod;
      
      Phi_hat = X'*X/T - (X'* (mean_z_on_x')/T) * Lambda - Lambda'* (mean_z_on_x * X/T) + Lambda'* m2_z_on_x * Lambda;
      
      Phi     = diag(diag(Phi_hat));
       
      % convergence condition
      
      n = n - 1;
       
      para_dist = l21_norm(Lambda_old - Lambda) + l21_norm(Phi_old - Phi); 
      
      if (n < 3000) && (para_dist < tol)
          break;
      end  
      
      Lambda_old = Lambda;
      Phi_old    = Phi; 
        
    end    
    
    Z = mean_z_on_x'; 
end 