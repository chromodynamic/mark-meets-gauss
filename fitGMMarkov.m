function [objGMM, M, varargout] = fitGMMarkov(data,tVec,varargin)
% fitGMMarkov.m - Mixture of Gaussians Markov Chain Self-supervised Fit
% 
% Given a 1d vector of data and time, a GMM-Markov process model is fit for
% forecasting measured data. The EM algorithm clusters the data into
% respective Gaussian models, the number of which can be determined via
% iterative computation of the Akaike information criterion with
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

% Defualt Expectation-Maximization Parameters
Ngmm_max = 25;      % Maxmimum number of distributions for AIC fit
Ngmm_reinit = 3;    % Number of reinitilization of EM algorithm
options = statset('MaxIter',250);

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
        for  j = 1:Ngmm_reinit,     % EM is nonconvex, reinitialize problem
            temp_obj{j} = gmdistribution.fit(data,numGMM,'Options',options);
            temp_min(j) = temp_obj{j}.NlogL;
        end
        [~,I] = min(temp_min);
        objGMM = temp_obj{I};
        varargout{1} = objGMM.NlogL;
    otherwise
        % AIC Criteria fitting of GMM
        AIC = zeros(Ngmm_max,1);
        for k = 1:Ngmm_max
            for  j = 1:Ngmm_reinit,     % EM is nonconvex, reinitialize problem
                try
                    temp_obj{j} = gmdistribution.fit(data,k,'Options',options);
                catch ME
                    temp_obj{j} = gmdistribution.fit(data,k,'Options',options);
                end
                temp_min(j) = temp_obj{j}.NlogL;
            end
            [~,I] = min(temp_min);
            obj{k} = temp_obj{I};
            AIC(k)= obj{k}.AIC;
        end
        varargout{1} = AIC;    [~,I] = min(AIC);
        objGMM = obj{I}; 
        numGMM = I;
end

% Adjust time vector for Markov model forcast
if dt_forcast ~= 0,
    dt = tVec(2) - tVec(1);
    dVec = round(dt/dt_forcast);
else
    dVec = 1;
end

% ----- Compute Markov Transition Matrix -------------------------------

% Cluster data to most likely GMM set
% Generates interger vector with values [1,2,3] for each data point
% indicating maximum likely distribution
idx = cluster(objGMM,data(1:dVec:end));

% Count transitions between distributions
td= sparse(idx(1:end-1), idx(2:end), 1);    t = full(td);

% Normalize counts across rows
sumRow = sum(t,2);  sumRow(sumRow == 0) = 1;
M = t./repmat(sumRow,1,size(t,2));

% ----- Display Results ------------------------------------------------
% Plot AIC Criteria Decision
if disp_AIC == 1 && length(varargout{1}) > 1,
    figure;  plot(AIC);
    xlabel('Number of mixtures');   ylabel('AIC');
    title('GMM Fit via Akaike Information Criteria');
    hold on;
    plot(I,min(AIC),'ro');
    hold off;
end

% Plot GM Mixture Models and Complete GMM PDF
if disp_GMMixture == 1,
    mu = objGMM.mu;    sigmaVar = squeeze(objGMM.Sigma);
    wdist = objGMM.PComponents;
    
    x = linspace(min(data)*0.98,max(data)*1.02,10e3)';
    pdfPlot = zeros(length(x),numGMM + 1);  pdfTemp = zeros(length(x),1);
    leg_Data = cell(1,numGMM + 2);  leg_Data{1} = 'Data';
    leg_Data{2} = 'GMM Fit';
    
    figure;
    for j = 1:numGMM,
        pdfPlot(:,j+1) = wdist(j)*pdf('norm',x,mu(j),sqrt(sigmaVar(j)));
        leg_Data{j+2} = num2str(j);   
        leg_Data{j+2} = strcat('Dist. ',leg_Data{j+2});
        pdfTemp = pdfTemp + pdfPlot(:,j+1);
    end
    pdfPlot(:,1) = pdfTemp;
    
    [n,ybar] = hist(data);     b = bar(ybar,n,'hist'); hold on;
    set(b,'FaceColor',[1 1 1]); area = sum(n)*(ybar(2) - ybar(1));
    
    p = plot(x,area*pdfPlot(:,1),x,area*pdfPlot(:,2:end),'-.');             
    set(p,'LineWidth',1.5);
    legend(leg_Data,'FontSize',12)
    title('Gaussian Mixture Model Components','FontSize',15);
    ylabel('PDF - \phi(x)','FontSize',12); grid on;
end

% Plot Markov Transition Probabilities
if disp_Markov == 1,
    figure;  bar3(M);
    ylabel('Prior Distribution');   xlabel('Posteriori Distribution');
    zlabel('Transition Probability');
    title('Markov Chain Transmission Matrix');
end

% Plot Comparison between data and GMM data fit
if disp_Compare == 1 && exist('nhist','file') == 2,
   figure;
   A.Data = data;   A.GMM = random(objGMM,10e3);
   nhist(A);
   ylabel('\phi(x) - PDF'); title('GMM PDF Fit Comparison');
end
   