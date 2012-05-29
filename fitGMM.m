function [mu,sigma,wdist,iter,z,Pm_x] = fitGMM(vecIn,numGMM,varargin)
% fitGMM.m - EM algorithm for parameter estimation of 1d GMM
% 
% Given a 1d vector of data, the expectation-maximization (EM) algorithm
% computes the parameters of a Gaussian mixture model (GMM). More
% concretely, the EM algorithm determines the unconstrained argument of the
% maximum for P_theta(x|theta), where theta = {alpha,mu,sigma}
%
%   Inputs: 
%       vecIn           - 1d input vector of data
%       numGMM          - number of clusters
%       'minerr',eps    - specify the minimum step error
%       'maxiter',max   - specify the maximum number of iterations
%       'displayfit'    - plot resulting fit as compared to data histogram
%
%   Outputs:
%       mu              - mean of each cluster
%       sigma           - variance of each cluster
%       wdist           - probability weights of each Gaussian mixture
%       iter            - number of iterations for convergence
%       z               - MAP classification of data vector given GMM
%       Pm_x            - soft classification, probability of cluster
%
% Written by: Jonathan LeSage - jrlesage@gmail.com
% University of Texas at Austin - Department of Mechanical Engineering
% Last revision date: 5/25/2012

% Default Expectation-Maximization Parameters
minEp = 1e-4;
maxIter = 1e5;
M = length(vecIn);  dataVec = repmat(vecIn,1,numGMM);   dispRes = 0;

% ----- Import/process additional argument inputs -----------------------
n = 1;
while n <= length(varargin),
    if ischar(varargin{n}),
        switch lower(varargin{n}),
            case 'minerr'
                if isscalar(varargin{n + 1}),
                    minEp = varargin{n + 1};
                else
                    warning('war:minEp','Improper input');
                    disp('Bad term:');  disp(varargin{n + 1});
                end
                n = n + 2;
            case 'maxiter'
                if isscalar(varargin{n + 1}),
                    maxIter = varargin{n + 1};
                else
                    warning('war:maxIter','Improper input');
                    disp('Bad term:');  disp(varargin{n + 1});
                end
                n = n + 2;
            case 'displayfit'
                dispRes = 1;
                n = n + 1;
            otherwise
                warning('war:input','Input parameter not recognized');
                disp('Bad term:');  disp(varargin{n});
                n = n + 1;
        end
    else
        warning('war:input','Input parameter not recognized');
        disp('Bad term:');  disp(varargin{n});
        n = n + 1;
    end 
end

% ----- Initialize EM Algorithm -----------------------------------------
% Initial parameter estimates
mu = mean(vecIn) + std(vecIn)*randn(numGMM,1);
sigma = var(vecIn)*ones(numGMM,1);
wdist = ones(numGMM,1)/numGMM;
iter = 0;  % Iteration counter
dStep = minEp;

% ----- Iterative EM Algorithm ------------------------------------------
while (iter < maxIter && dStep >= minEp),
   
    % Temporary storage of old values
    mu0 = mu;  sigma0 = sigma;    wdist0 = wdist;
    Pm_x = zeros(M,numGMM);
    
    % Estimation-step - via Bayes rule
    % Soft clustering of data into current probability density function
    % Compute - P(M|x_i) = a_m*\phi(x_i|mu_h,sigma_h)/sum(a_m*\phi)
    for m = 1:numGMM,
        Px_theta0 = exp(-(vecIn-mu(m)).^2/(2*sigma(m)))/ ...
            (sqrt(sigma(m)*2*pi));     % PDF given current parameters
        Pm_x(:,m) = wdist(m)*Px_theta0;
    end
    Pm_x = Pm_x./repmat(sum(Pm_x,2),1,numGMM);
    
    % Maximization-step - optimization of log(P_theta)
    % Significant statistics must equate each step (maximization)
    wdist = mean(Pm_x)';
    mu = (sum(Pm_x.*dataVec)./sum(Pm_x))';
    muMat = repmat(mu',M,1);
    sigma = (sum(Pm_x.*(dataVec - muMat).^2)./sum(Pm_x))';
    
    % Update exit conditions
    iter = iter + 1;
    dStep = norm(mu0-mu) + norm(sigma0-sigma) + norm(wdist0 - wdist);
end
[~,z] = max(Pm_x,[],2);     % Maximum a posteriori classification of data

% ---- Graphically Illustrate GMM Fit -----------------------------------
if dispRes == 1,
    x = linspace(min(vecIn)*0.98,max(vecIn)*1.02,10e3)';
    pdfPlot = zeros(length(x),numGMM + 1);  pdfTemp = zeros(length(x),1);
    leg_Data = cell(1,numGMM + 2);  leg_Data{1} = 'Data';
    leg_Data{2} = 'GMM Fit';
    
    figure
    for j = 1:numGMM,
        pdfPlot(:,j+1) = wdist(j)*pdf('norm',x,mu(j),sqrt(sigma(j)));
        leg_Data{j+2} = num2str(j);   
        leg_Data{j+2} = strcat('Dist. ',leg_Data{j+2});
        pdfTemp = pdfTemp + pdfPlot(:,j+1);
    end
    pdfPlot(:,1) = pdfTemp;
    
    [n,ybar] = hist(vecIn);     b = bar(ybar,n,'hist'); hold on;
    set(b,'FaceColor',[1 1 1]); area = sum(n)*(ybar(2) - ybar(1));
    
    p = plot(x,area*pdfPlot(:,1),x,area*pdfPlot(:,2:end),'-.');             
    set(p,'LineWidth',1.5);
    legend(leg_Data,'FontSize',12)
    title('Gaussian Mixture Model Components','FontSize',15);
    ylabel('PDF - \phi(x)','FontSize',12); grid on;
end