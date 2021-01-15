%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD - GECCO and CEC 2019 Competition: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ALGORITMH: HL_PS_VNSO
%HYBRID LEVY PARTICLE SWARM VARIABLE NEIGHBORHOOD SEARCH OPTIMIZATION
%% Developers: 
% Dharmesh A. Dabhi, Assistant Professor, M & V Patel Department of Electrical Engineering, CSPIT,
% CHARUSAT UNIVERSITY,CHANGA, Gujarat, INDIA
% Kartik S. Pandya, Professor, M & V Patel Department of Electrical Engineering, CSPIT,
% CHARUSAT UNIVERSITY,CHANGA, Gujarat, INDIA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2017 Competition & panel: Evaluating the Performance of Modern Heuristic
% Optimizers on Smart Grid Operation Problems: Variable Neighborhood Search algorithm (VNS)  
% The codes have been modified by the developers

function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = HL_PS_VNSO(HL_PS_VNS_Parameters,caseStudyData,otherParameters,xmin,xmax)
%%    
%-----This is just for notational convenience and to keep the code uncluttered.--------
I_NP         = HL_PS_VNS_Parameters.I_NP;
% F_weight     = HL_PS_VNS_Parameters.F_weight;
% F_CR         = HL_PS_VNS_Parameters.F_CR;
I_D          = numel(xmin); %Number of variables or dimension
HL_PS_VNS_Parameters.nVariables=I_D;
FVr_minbound = xmin;
FVr_maxbound = xmax;
I_itermax    = HL_PS_VNS_Parameters.I_itermax;

%Repair boundary method employed
BRM=HL_PS_VNS_Parameters.I_bnd_constr; %1: bring the value to bound violated
                               %2: repair in the allowed range

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_strategy   = HL_PS_VNS_Parameters.I_strategy; %important variable
fnc= otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----parameters-----------------------------------------------------------
% FLC modification - vectorization
minPositionsMatrix=repmat(FVr_minbound,I_NP,1);
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1);
HL_PS_VNS_Parameters.minPositionsMatrix=minPositionsMatrix;
HL_PS_VNS_Parameters.maxPositionsMatrix=maxPositionsMatrix;

% generate initial population.
%FM_pop=genpop(I_NP,I_D,minPositionsMatrix,maxPositionsMatrix);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------Evaluate the best member after initialization----------------------
% Modified by FLC
switch fnc
    case 'fitnessFun_DER'
    caseStudyData=caseStudyData(1);
end
%%
%% Vectors to store the solutions %% VNS start
%Niter = 50000 - 1;
%Niter = (deParameters.I_iterma+1) -1;
%Niter=round((deParameters.I_iterma+1)*0.61);
Niter=2; % this paremater was tuned
fitMaxVector = nan(1,Niter+1);
objMaxVector = nan(2,Niter+1);

%% Parameters
% Dimension of the problem
iter = 0;
nper = 24;
[~,D] = size(xmin);
nctrl = D/nper;
penalty = 10000;

Xmin = vec2mat(xmin,nctrl);
Xmax = vec2mat(xmax,nctrl);

%% Types of variables, this was modifed according to the 2018 competition

% Indices
Pdgind = otherParameters.ids.idsGen;      % Active power of a DG
%Qdgind = otherParameters.ids.idsQGen;     % Reactive power of a DG
Xdgind = otherParameters.ids.idsXGen;     % Binary status of a DG

V2Gind = otherParameters.ids.idsV2G;      % EV charging/discharging
DRind  = otherParameters.ids.idsLoadDR;   % Loads for demand response
ESSind = otherParameters.ids.idsStorage;  % ESS charging/discharging
MKind  = otherParameters.ids.idsMarket;   % Market
%Tapind = otherParameters.ids.idsTAP;      % DSS OLTC

% Number of controls
nDG  = length(Pdgind);
nV2G = length(V2Gind);
nDR  = length(DRind);
nESS = length(ESSind);
nMK  = length(MKind);
%nTAP = length(Tapind);

% Variable type
xtype = zeros(1,nctrl);

xtype(Pdgind(1):Pdgind(nDG))  = 1;
%xtype(Qdgind(1):Qdgind(nDG))  = 2;
xtype(Xdgind(1):Xdgind(nDG))  = 2;
xtype(V2Gind(1):V2Gind(nV2G)) = 3;
xtype(DRind(1):DRind(nDR))    = 4;
xtype(ESSind(1):ESSind(nESS)) = 5;
xtype(MKind(1):MKind(nMK))    = 6;
%xtype(Tapind(1):Tapind(nTAP)) = 8;

xtype = repmat(xtype,1,nper);

% Vector of fixed controls
xfix = zeros(1,D);

for i = 1:D
    if xmin(i) == xmax(i)
        xfix(i) = 1;
    end
end
nfix = sum(xfix); % Number of fixed variables

% Costs
% In
Loadc = caseStudyData(1).loadData.CofLoad;
MKc = caseStudyData(1).marketData.sellCofB;
ESSc_ch = caseStudyData(1).storageData.stPriceCharge;
V2Gc_ch = caseStudyData(1).v2gData.v2gCofCharge;

% OC
DGc = caseStudyData(1).genData.genCofB;
DGC = repmat(DGc,1,2);

%?energy price of external supplier S %%%%%%%%%
Loadredc = caseStudyData(1).loadData.CofCut;
Loadredc = caseStudyData(1).loadData.CofReduce;

V2Gc_dis = caseStudyData(1).v2gData.v2gCofDischarge;
ESSc_dis = caseStudyData(1).storageData.stPriceDischarge;
ENSc = caseStudyData(1).loadData.CofENS;

%% Random Initial Solution
x = zeros(1,D);
for i = 1:D
    x(i) = (xmax(i)-xmin(i))/2; 
end

for i = 1:D
    if x(i) < xmin(i)
        x(i) = xmin(i);
    elseif x(i) > xmax(i)
        x(i) = (xmax(i)-xmin(i))/2;  
    end
end

nEvals=0;

whileIter=0;

%% Random Initial Solution
x = zeros(1,D);
for i = 1:D
%     x(i) = (xmin(i) + xmax(i))/2;
    x(i) =xmax(i)/2.4;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if iter < Niter
    
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,HL_PS_VNS_Parameters.Scenarios);
        
            nEvals  = nEvals+(HL_PS_VNS_Parameters.I_NP*HL_PS_VNS_Parameters.Scenarios);

            for i=1:HL_PS_VNS_Parameters.I_NP   % Eval Initial population
                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                    obj(i)  = FIT(i); 
                    fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
            end
       
            xrep=x;
        %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
        %[fit,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
    iter = iter + 1;
    fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];
    %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',0,0,iter,fit,obj(2)-obj(1),fit+obj(1)-obj(2),obj(2),obj(1));

    FITBEST = FIT;
    OBJBEST = obj;
    fitMaxVector(iter) = FITBEST;
    objMaxVector(:,iter)= OBJBEST(1,:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fbest = fit
objbest = obj;
xbest = xrep;

%% Improve initial solution
display('Improve initial solution')

X = vec2mat(xbest,nctrl);

for i = 1:nper
    for j = 1:nctrl
        if xtype(j) == 3
              X(i,j) = Xmin(i,j) + (Xmax(i,j) - Xmin(i,j))/3;
%              X(i,j) = Xmin(i,j);
        elseif xtype(j) == 4
            X(i,j) = Xmin(i,j);
        elseif xtype(j) == 6
            X(i,j) = Xmin(i,j);
        end
    end
end

iimp = [1, 2, 4, 5, 6];
iimpn = length(iimp);

betat = nan(1,iimpn);

for kn = 1:iimpn
     for beta = 0.1:0.05:0.5
        for i = 1:nper
            for j = 1:nctrl
                if xtype(j) == iimp(kn)
                    X(i,j) = Xmin(i,j) + (Xmax(i,j) - Xmin(i,j)/2);
                 elseif xtype(j) == 4
            X(i,j) = Xmin(i,j);
        elseif xtype(j) == 6
            X(i,j) = Xmin(i,j);
                end
            end
        end

        k = 1;
        for i = 1:nper
            for j = 1:nctrl
                x(k) = X(i,j);
                k = k + 1;
            end
        end

        for i = 1:D
            if x(i) < xmin(i)
                x(i) = xmin(i);
            elseif x(i) > xmax(i)
                x(i) = xmax(i)/2;  
            end
        end    

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if iter < Niter
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,HL_PS_VNS_Parameters.Scenarios);
        
            nEvals  = nEvals+(HL_PS_VNS_Parameters.I_NP*HL_PS_VNS_Parameters.Scenarios);

            for i=1:HL_PS_VNS_Parameters.I_NP   % Eval Initial population
                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                    obj(i)  = FIT(i); 
                    fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
            end
       
            xrep=x;
            
            %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
            %[fit,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
            iter = iter + 1;
            fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];

            if FIT < FITBEST
                FITBEST = FIT;
                OBJBEST = obj;
                fitMaxVector(iter) = FITBEST;
                objMaxVector(:,iter)= OBJBEST(1,:);
            else
                fitMaxVector(iter) = FITBEST;
                objMaxVector(:,iter)= OBJBEST(1,:);
            end
        else
            break;
            1001
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if fit < fbest
            %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',0,kn,iter,fit,obj(2)-obj(1),fit+obj(1)-obj(2),obj(2),obj(1));

            fbest = fit
            nEvals
            objbest = obj;
            xbest = xrep;
            
            betat(kn) = beta;
        end
    end
    X = vec2mat(xbest,nctrl);
end


% %% Evaluate initial improved solution
% display('Evaluate initial improved solution')
% nEvals
% fbest
% 
% k = 1;
% for i = 1:nper
%     for j = 1:nctrl
%         x(k) = X(i,j);
%         k = k + 1;
%     end
% end
% 
% for i = 1:D
%     if x(i) < xmin(i)
%         x(i) = xmin(i);
%     elseif x(i) > xmax(i)
%         x(i) = xmax(i);  
%     end
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if iter < Niter
%     
%             [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,HL_PS_VNS_Parameters.Scenarios);
%         
%             nEvals  = nEvals+(HL_PS_VNS_Parameters.I_NP*HL_PS_VNS_Parameters.Scenarios);
% 
%             for i=1:deParameters.I_NP   % Eval Initial population
%                     FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                     obj(i)  = FIT(i); 
%                     fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
%             end
%        
%             xrep=x;
%         
%     %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
%     %[fit,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,x,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%     iter = iter + 1;
%     fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];
%     %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',0,0,iter,fit,obj(2)-obj(1),fit+obj(1)-obj(2),obj(2),obj(1));
% 
%     if FIT < FITBEST
%         FITBEST = FIT;
%         OBJBEST = obj;
%         fitMaxVector(iter) = FITBEST;
%         objMaxVector(:,iter)= OBJBEST(1,:);
%     else
%         fitMaxVector(iter) = FITBEST;
%         objMaxVector(:,iter)= OBJBEST(1,:);
%     end
% end



    nEvals=iter*HL_PS_VNS_Parameters.Scenarios
    fbest
    
    x=xrep; 


%%
    display('LEVY_PSO')
    HL_PS_VNS_PARAMETERS; %Function defined by the participant
    No_solutions=HL_PS_VNS_Parameters.I_NP;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Set other parameters
     otherParameters =setOtherParameters(caseStudyData,No_solutions);
otherParameters.No_eval_Scenarios=500;
    [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    LEVY_PSO(HL_PS_VNS_Parameters,caseStudyData,otherParameters,xmin,xmax,nEvals,xbest);

end