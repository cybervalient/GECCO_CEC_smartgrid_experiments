%% Proposed ALGORITHM: Gauss Mapped Variable Neighbourhood Particle Swarm Optimization (GM_VNPSO)
%% Developers: 
% Kartik S. Pandya, Professor, Dept. of Electrical Engineering, CSPIT,
% CHARUSAT, Gujarat, INDIA
% Jigar Sarda, Asst. Professor, Dept. of Electrical Engineering, CSPIT,
% CHARUSAT, Gujarat, INDIA
% Margi Shah, Independent Researcher, Gujarat, INDIA


%% References:
%The codes of VNS is available on http://sites.ieee.org/psace-mho/2017-smart-grid-operation-problems-competition-panel/.
%The codes of EPSO is available on http://www.gecad.isep.ipp.pt/WCCI2018-SG-COMPETITION/
%The Codes of VNS were modified by Sergio Rivera, srriverar@unal.edu.co,professor at UN.
% The Codes of EPSO were developed by Phillipe Vilaça Gomes
% The codes have been modified by the developers to propose GM_VNPSO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = GM_VNPSO(deParameters,caseStudyData,otherParameters,xmin,xmax)
I_NP         = deParameters.I_NP;
F_weight     = deParameters.F_weight;
F_CR         = deParameters.F_CR;
I_D          = numel(xmin); 
deParameters.nVariables=I_D;
FVr_minbound = xmin;
FVr_maxbound = xmax;
I_itermax    = deParameters.I_itermax;
BRM=deParameters.I_bnd_constr; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_strategy   = deParameters.I_strategy; 
fnc= otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

minPositionsMatrix=repmat(FVr_minbound,I_NP,1);
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1);
deParameters.minPositionsMatrix=minPositionsMatrix;
deParameters.maxPositionsMatrix=maxPositionsMatrix;


switch fnc
    case 'fitnessFun_DER'
    caseStudyData=caseStudyData(1);
end

Niter=3; % this paremater was tuned
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
Pdgind = otherParameters.ids.idsGen;     
Xdgind = otherParameters.ids.idsXGen;     
V2Gind = otherParameters.ids.idsV2G;      
DRind  = otherParameters.ids.idsLoadDR; 
ESSind = otherParameters.ids.idsStorage;  
MKind  = otherParameters.ids.idsMarket;  


% Number of controls
nDG  = length(Pdgind);
nV2G = length(V2Gind);
nDR  = length(DRind);
nESS = length(ESSind);
nMK  = length(MKind);


% Variable type
xtype = zeros(1,nctrl);

xtype(Pdgind(1):Pdgind(nDG))  = 1;
xtype(Xdgind(1):Xdgind(nDG))  = 2; 
xtype(V2Gind(1):V2Gind(nV2G)) = 3;
xtype(DRind(1):DRind(nDR))    = 4;
xtype(ESSind(1):ESSind(nESS)) = 5;
xtype(MKind(1):MKind(nMK))    = 6;


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
for i=1:D
    if xtype(i)==1
        x(i)= (rand*(xmax(i) - xmin(i))/4) ;
    else if xtype(i)==3
            x(i)=(rand*(xmax(i) - xmin(i))/4) ;
             else if xtype(i)==4
            x(i)=(rand*(xmax(i) - xmin(i))/4) ;
             else if xtype(i)==5
            x(i)=(rand*(xmax(i) - xmin(i))/4) ;
             else if xtype(i)==6
           x(i)=(rand*(xmax(i) - xmin(i))/4) ;
          else x(i)=round(rand);
                 end
                 end
                 end
        end
    end
end
       

for i = 1:D
    if x(i) < xmin(i)
        x(i) = xmin(i);
    elseif x(i) > xmax(i) 
          x(i) = xmax(i);
    end
end

nEvals=0;

whileIter=0;

%% Random Initial Solution
x = zeros(1,D);
for i = 1:D
    % x(i) = (xmin(i) + xmax(i));
    x(i) =xmax(i)/2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if iter < Niter
    
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
        
            nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

            for i=1:deParameters.I_NP   % Eval Initial population
                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                    obj(i)  = FIT(i); 
                    fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
            end
       xrep=x;
       iter = iter + 1;
    fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];
    
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
              X(i,j) = Xmin(i,j) + (Xmax(i,j) - Xmin(i,j))/6;
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
                    X(i,j) = Xmin(i,j) +(Xmax(i,j) - Xmin(i,j)/2).*rand;  % 
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
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
        
            nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

            for i=1:deParameters.I_NP   % Eval Initial population
                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                    obj(i)  = FIT(i); 
                    fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
            end
       
            xrep=x;
   
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
          
            fbest = fit
            nEvals
            objbest = obj;
            xbest = xrep;
            
            betat(kn) = beta;
        end
    end
    X = vec2mat(xbest,nctrl);
end


%% Evaluate initial improved solution
display('Evaluate initial improved solution')
nEvals
fbest

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
    
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
        
            nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

            for i=1:deParameters.I_NP   % Eval Initial population
                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                    obj(i)  = FIT(i); 
                    fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
            end
       
            xrep=x;
        
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
end

%% Cyclic Coordinated Method
display('Cyclic Coordinated Method')
fbest
nEvals

xbest = xrep;
fbest = fit;
objbest = obj;


    display('EPSOu')
    %DEparametersDEEPSO %Function defined by the participant
   EPSOuParameters %Function defined by the participant
   % No_solutions=deParameters.I_NP;
    No_solutions=epsouParameters.I_NP;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Set other parameters
     otherParameters =setOtherParameters(caseStudyData,No_solutions);

     otherParameters.No_eval_Scenarios=500;
    
     [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
     EPSOu(epsouParameters,caseStudyData,otherParameters,xmin,xmax,otherParameters.No_eval_Scenarios,xbest,nEvals);


end