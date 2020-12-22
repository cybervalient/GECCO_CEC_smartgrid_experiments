% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
% Project Code: YPEA125
% Project Title: Implementation of Cultural Algorithm in MATLAB
% Publisher: Yarpiz (www.yarpiz.com) 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
% Modified by R_Faia_v2 \GECAD/05/2017

function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    cultral_algorithm_v2(caParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)
%% Problem Definition
         % Number of Decision Variables



VarMin=low_habitat_limit;         % Decision Variables Lower Bound
VarMax=up_habitat_limit;         % Decision Variables Upper Bound

%%
%Repair boundary method employed
BRM=caParameters.I_bnd_constr; %1: bring the value to bound violated
                               %2: repair in the allowed range
%% Initialization
MaxIt=      caParameters.MaxIt;
nPop=       caParameters.nPop;   
pAccept=    caParameters.pAccept; 
nAccept=    caParameters.nAccept;
alpha=      caParameters.alpha;
beta=       caParameters.beta;
Method=     caParameters.Method;
I_D=        numel(up_habitat_limit);
VarSize=I_D;   % Decision Variables Matrix Size

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fnc= otherParameters.fnc;
% FLC modification - vectorization
minPositionsMatrix=repmat(VarMin,nPop,1);
maxPositionsMatrix=repmat(VarMax,nPop,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize Culture
Culture.Situational.Cost=inf;
Culture.Normative.Min=inf(1,VarSize);
Culture.Normative.Max=-inf(1,VarSize);
Culture.Normative.L=inf(1,VarSize);
Culture.Normative.U=inf(1,VarSize);

% Empty Individual Structure
empty_individual.Position=[];
empty_individual.Cost=[];
%------Evaluate the best member after initialization----% Initialize Population Array------------------
% Modified by FLC
switch fnc
    case 'fitnessFun_DER'
    caseStudyData=caseStudyData(1);
end

for i=1:nPop
    pop(i).Position=genpop(1,I_D,VarMin,VarMax);
    [solFitness_M(i,:),solPenalties_M(i,:),Struct_Eval(i,:)]=feval(fnc,pop(i).Position,caseStudyData, otherParameters.um,10);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The user should decide which is the criterion to optimize. 
% In this example, we optimize worse performance
%% Worse performance criterion
[S_val, worstS]=min(solFitness_M,[],2); %Choose the solution with worse performance
for i=1:nPop
     pop(i).Cost=S_val(i);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sort Population
[~, SortOrder]=sort([pop.Cost]);
pop=pop(SortOrder);

% Adjust Culture using Selected Population
spop=pop(1:nAccept);
Culture=AdjustCulture(Culture,spop);

% Update Best Solution Ever Found
BestSol=Culture.Situational;

% Array to Hold Best Costs
BestCost=zeros(MaxIt,1);



[~,I_best_index] = min(S_val); % This mean that the best individual correspond to the best worst performance
FVr_bestmemit = pop(I_best_index).Position; % best member of current iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% store other information
Best_otherInfo.idBestParticle = I_best_index;
Best_otherInfo.genCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.genCosts(1,:);
Best_otherInfo.loadDRcostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.loadDRcosts(1,:);
Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gChargeCosts(1,:);
Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(worstS(I_best_index)).otherParameters.v2gDischargeCosts(1,:);
Best_otherInfo.storageChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageChargeCosts(1,:);
Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageDischargeCosts(1,:);
Best_otherInfo.stBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.stBalance(1,:,:);
Best_otherInfo.v2gBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gBalance(1,:,:);
Best_otherInfo.penSlackBusFinal = Struct_Eval(worstS(I_best_index)).otherParameters.penSlackBus(1,:);
fitMaxVector(:,1)=[mean(solFitness_M(I_best_index,:));mean(solPenalties_M(I_best_index,:))]; %We save the mean value and mean penalty value
% The user can decide to save the mean, best, or any other value here





%% Cultural Algorithm Main Loop
it=2;
while it<MaxIt
    
    % Influnce of Culture
    for i=1:nPop
      switch Method
          case 1    
        % % 1st Method (using only Normative component)
        sigma=alpha*Culture.Normative.Size;
        pop(i).Position=pop(i).Position+sigma.*randn(I_D);
        
        case 2
        % % 2nd Method (using only Situational component)
        sigma=0.1.*(VarMax-VarMin);
        dx=sigma.*randn(1,I_D);
        ind=pop(i).Position<Culture.Situational.Position;
        dx(ind)=abs(dx(ind));
        ind=pop(i).Position>Culture.Situational.Position;
        dx(ind)=-abs(dx(ind));
        pop(i).Position=pop(i).Position+dx;
        
        case 3
          sigma=alpha*Culture.Normative.Size;
          dx=sigma.*randn(1,I_D);
          ind=pop(i).Position<Culture.Situational.Position;
          dx(ind)=abs(dx(ind));
          ind=pop(i).Position>Culture.Situational.Position;
          dx(ind)=-abs(dx(ind));
          pop(i).Position=pop(i).Position+dx;
        case 4
        % % 4th Method (using Size and Range of Normative component)
        %for j=1:I_D
         sigma=alpha*Culture.Normative.Size;
          dx=sigma.*randn(1,I_D);
          ind=pop(i).Position<Culture.Normative.Min;
          dx(ind)=abs(dx(ind));
          ind=pop(i).Position>Culture.Normative.Max;
          dx(ind)=-abs(dx(ind));
          ind=pop(i).Position==Culture.Normative.Max;
          dx(ind)=beta*(dx(ind));
          
          pop(i).Position=pop(i).Position+dx;
          %end        
      end
      
       %% Boundary Control
     % pop.Position=update(pop.Position,minPositionsMatrix,maxPositionsMatrix,BRM,FM_base);

      %pop(i).Cost=CostFunction(pop(i).Position);
       [solFitness_M(i,:),solPenalties_M(i,:),Struct_Eval(i,:)]=feval(fnc,pop(i).Position,caseStudyData, otherParameters.um,10);
       [S_val, ~]=min(solFitness_M,[],2); %Choose the solution with worse performance
       pop(i).Cost=S_val(i);
        
        
    end
    
    % Sort Population
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);

    % Adjust Culture using Selected Population
    spop=pop(1:nAccept);
    Culture=AdjustCulture(Culture,spop);

    % Update Best Solution Ever Found
    BestSol=Culture.Situational;
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    
    fitMaxVector(:,it)= [BestSol.Cost;0];
    FVr_bestmemit= BestSol.Position;
    it=it+1;
end
p1=0;
Fit_and_p=[fitMaxVector(1,it-1) p1]; %;p2;p3;p4]



% VECTORIZED THE CODE INSTEAD OF USING FOR
function pop=genpop(a,b,lowMatrix,upMatrix)
pop=unifrnd(lowMatrix,upMatrix,a,b);


