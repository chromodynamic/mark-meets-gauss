function [mu, sigma, wdist, M, varargout] = fitGMMarkov(data,tVec,varargin)
% fitGMMarkov.m - Mixture of Gaussians Markov Chain Self-supervised Fit
% 
% Given a 1d vector of data and time, a GMM-Markov process model is fit for
% forecasting measured data. The EM algorithm clusters the data into
% respective Gaussian models, the number of which can be determined via
% iterative computation (greedy) of the Akaike information criterion with
% incremental number of clusters. The data vector is sorted into clusters
% respectively via MAP decision and the transition probabilities are
% computed over time.
%
%   Inputs: 
%       data            - 1d input vector of data
%       tVec            - 1d time vector
%       'aic','true',N  - fit GMM via AIC with maximum of N clusters
%       'numbergmm',N   - specify N number of clusters
%       'forcastdt',dt  - change timestep of Markov chain
%       'displayall'    - display all below figures
%       'displayaic'    - display AIC vector with optimal cluster number
%       'displaygmm'    - display all mixtures and overall pdf
%       'dispcompare'   - display data and GMM fit
%       'displaymarkov' - display markov transition matrix graphically
%
%   Outputs:
%       objGMM          - Gaussian mixture model object
%       M               - Markov transition matrix
%       AIC (or NlogL)  - Akaike values (or negative log likelihood)
%
% Written by: Jonathan LeSage - jrlesage@gmail.com
% University of Texas at Austin - Department of Mechanical Engineering
% Last revision date: 5/25/2012

% Default User Input Parameters
aicFit = 0; dt_forcast = 0; numGMM = 0;
disp_AIC = 0;   disp_GMMixture = 0;
disp_Compare = 0;   disp_Markov = 0;

% Determine if Statistics Toolbox is installed
try
    gmdistribution.fit([1;2;3],1);
    builtInFit = 1;
catch noPackage
    builtInFit = 0;
end

% Default Expectation-Maximization Parameters
Ngmm_max = 25;      % Maxmimum number of distributions for AIC fit
Ngmm_reinit = 3;    % Number of reinitilization of EM algorithm

if isvector(data) ~= 1 || isvector(tVec) ~= 1,
    error('er:vector','Data or time not arrays');
end

% ----- Import/process additional argument inputs -----------------------
n = 1;
while n <= length(varargin),
    if ischar(varargin{n}),
        switch lower(varargin{n}),
            case 'aic',
                if isequal(varargin{n + 1},'true'),
                    aicFit = 1;
                    if isscalar(varargin{n + 2}),
                        Ngmm_max = varargin{n + 2};
                        n = n + 1;
                    end
                elseif ischar(varargin{n + 1}),
                    aicFit = 0;
                else
                    warning('war:aicfail','Improper input');
                    disp('Bad term:');  disp(varargin{n + 1});
                end
                n = n + 2;
            case 'forcastdt',
                if isscalar(varargin{n + 1}),
                    dt_forcast = varargin{n + 1};
                else
                    warning('war:dtfail','Improper input');
                    disp('Bad term:');  disp(varargin{n + 1});
                end
                n = n + 2;
            case 'numbergmm',
                if isscalar(varargin{n + 1}),
                    numGMM = varargin{n + 1};
                else
                    warning('war:Ngmmfail','Improper input');
                    disp('Bad term:');  disp(varargin{n + 1});
                end
                n = n + 2;
            case 'displayaic',
                disp_AIC = 1;
                n = n + 1;
            case 'displaygmm',
                disp_GMMixture = 1;
                n = n + 1;
            case 'displaycompare',
                disp_Compare = 1;
                n = n + 1;
            case 'displaymarkov'
                disp_Markov = 1;
                n = n + 1;
            case 'displayall',
                disp_AIC = 1;
                disp_GMMixture = 1;
                disp_Compare = 1;
                disp_Markov = 1;
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
if aicFit == 1 && numGMM ~= 0,
    aicFit = 0;     % Artificially selecting GMM number trumps AIC 
end

% ----- Fit Gaussian Mixture Model via EM ------------------------------
switch aicFit,
    case 0,
        % Prespecified number of GM models
        if builtInFit == 1,
            for  j = 1:Ngmm_reinit,     % EM is nonconvex, reinitialize
                temp_obj = gmdistribution.fit(data,numGMM);
                mutemp{j} = temp_obj.mu;
                sigmatemp{j} = squeeze(temp_obj.Sigma);
                wtemp{j} = temp_obj.PComponents;
                ztemp{j} = cluster(temp_obj,data);
                temp_min(j) = temp_obj.NlogL;
                AICtemp(j) = temp_obj.AIC;
            end
        else
            for  j = 1:Ngmm_reinit,     % EM is nonconvex, reinitialize
                [mutemp{j},sigmatemp{j},wtemp{j},ztemp{j},temp_min(j), ...
                        AICtemp(j)] = fitGMM(data,numGMM);
            end
        end
        [~,I] = min(temp_min);
        mu = mutemp{I};     sigma = sigmatemp{I};   wdist = wtemp{I};
        idx = ztemp{I};     AIC = AICtemp(I);
        varargout{1} = temp_min(I);
    otherwise
        % AIC Criteria fitting of GMM
        AIC = zeros(Ngmm_max,1);
        for k = 1:Ngmm_max,
            if builtInFit == 1,
                for  j = 1:Ngmm_reinit,
                    temp_obj = gmdistribution.fit(data,k);
                    mutemp{j} = temp_obj.mu;
                    sigmatemp{j} = squeeze(temp_obj.Sigma);
                    wtemp{j} = temp_obj.PComponents;
                    ztemp{j} = cluster(temp_obj,data);
                    temp_min(j) = temp_obj.NlogL;
                    AICtemp(j) = temp_obj.AIC;
                end
            else
                for  j = 1:Ngmm_reinit,     % EM is nonconvex
                    [mutemp{j},sigmatemp{j},wtemp{j},ztemp{j},temp_min(j), ...
                        AICtemp(j)] = fitGMM(data,k);
                end
            end
            [~,I] = min(temp_min);        muAIC{k} = mutemp{I};  
            sigmaAIC{k} = sigmatemp{I};   wAIC{k} = wtemp{I};
            zAIC{k} = ztemp{I};           AIC(k)= AICtemp(I);
        end
        varargout{1} = AIC;   [~,I] = min(AIC);
        mu = muAIC{I};        sigma = sigmaAIC{I};   
        wdist = wAIC{I};      numGMM = I;
        idx = zAIC{I};
end
varargout{2} = idx;

% Adjust time vector for Markov model forcast
if dt_forcast ~= 0,
    dt = tVec(2) - tVec(1);
    dVec = round(dt/dt_forcast);
else
    dVec = 1;
end

% ----- Compute Markov Transition Matrix -------------------------------

% Count transitions between distributions
td= sparse(idx(1:end-1), idx(2:end), 1);    t = full(td);

% Normalize counts across rows
sumRow = sum(t,2);  sumRow(sumRow == 0) = 1;
M = t./repmat(sumRow,1,size(t,2));

% ----- Display Results ------------------------------------------------
% Plot AIC Criteria Decision
if disp_AIC == 1 && length(varargout{1}) > 1,
    figure;  plot(AIC); grid on;
    xlabel('Number of mixtures');   ylabel('AIC');
    title('GMM Fit via Akaike Information Criteria');
    hold on;
    plot(I,min(AIC),'ro');
    hold off;
end

% Plot GM Mixture Models and Complete GMM PDF
if disp_GMMixture == 1,
    x = linspace(min(data)*0.98,max(data)*1.02,10e3)';
    pdfPlot = zeros(length(x),numGMM + 1);  pdfTemp = zeros(length(x),1);
    leg_Data = cell(1,numGMM + 2);  leg_Data{1} = 'Data';
    leg_Data{2} = 'GMM Fit';
    
    figure;
    for j = 1:numGMM,
        pdfPlot(:,j+1) = wdist(j)*pdf('norm',x,mu(j),sqrt(sigma(j)));
        leg_Data{j+2} = num2str(j);   
        leg_Data{j+2} = strcat('',leg_Data{j+2});
        pdfTemp = pdfTemp + pdfPlot(:,j+1);
    end
    pdfPlot(:,1) = pdfTemp;
    
    [n,ybar] = hist(data,100);     b = bar(ybar,n,'hist'); hold on;
    set(b,'FaceColor',[1 1 1]);    area = sum(n)*(ybar(2) - ybar(1));
    
    p = plot(x,area*pdfPlot(:,1),x,area*pdfPlot(:,2:end),'-.'); grid on;        
    set(p,'LineWidth',1.5);
    legend(leg_Data,'FontSize',12)
    title('Gaussian Mixture Model Components','FontSize',15);
    ylabel('PDF - \phi(x)','FontSize',12); grid on;
end

% Plot Markov Transition Probabilities
if disp_Markov == 1,
    figure;
    imagesc(M);     colorbar;   colormap(gray);
    ylabel('Prior Distribution');   xlabel('Posteriori Distribution');
    zlabel('Transition Probability');
    title('Markov Chain Transmission Matrix');
end

% Plot Comparison between data and GMM data fit
if disp_Compare == 1 && exist('nhist','file') == 2 && builtInFit == 1,
    % Generate Fit Data for KS Density Comparison
    numRand = length(data);
    randIndex = randsample(1:length(wdist), numRand, true, wdist);
    realization = mu(randIndex) + sqrt(sigma(randIndex)).*randn(numRand,1);
    
    figure;
    % Histogram of Actual Data Vector
    [f,xi] = hist(data,round(numRand/20));  b = bar(xi,f,'hist'); 
    hold on;   set(b,'FaceColor',[1 1 1]);  area = sum(f)*(xi(2) - xi(1));
    
    % Compute Kernal Density of Data and GMM Fit
    datafit = ksdensity(data,xi);   gmmfit = ksdensity(realization,xi);
    
    % Plot Results
    p = plot(xi,area*datafit,xi,area*gmmfit,'-.'); grid on;        
    set(p,'LineWidth',1.5);     legend('Data','Data PDF','GMM PDF');
    title('GMM PDF Fit Comparison','FontSize',15);
    ylabel('PDF - \phi(x)','FontSize',12); grid on;
end

   