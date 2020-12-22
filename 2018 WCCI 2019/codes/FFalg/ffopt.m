%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YOEA112
% Project Title: Implementation of Firefly Algorithm (FA) in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

%Modified by Ansel Rodriguez (CICESE 24/05/2018)

function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    ffopt(ffParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)
%% Problem Definition

%CostFunction=@(x) Sphere(x);        % Cost Function
fnc= otherParameters.fnc;

nVar=numel(up_habitat_limit); %Number of variables or dimension;                 % Number of Decision Variables

VarSize=[1 nVar];       % Decision Variables Matrix Size

VarMin=low_habitat_limit;             % Decision Variables Lower Bound
VarMax= up_habitat_limit;             % Decision Variables Upper Bound

%% Firefly Algorithm Parameters

MaxIt=ffParameters.MaxIt; %1000;         % Maximum Number of Iterations

nPop=ffParameters.nPop; %25;            % Number of Fireflies (Swarm Size)

gamma=ffParameters.gamma; %1;            % Light Absorption Coefficient
beta0=ffParameters.beta0; %2;            % Attraction Coefficient Base Value
alpha=ffParameters.alpha; %0.2;          % Mutation Coefficient
alpha_damp=ffParameters.alpha_damp; %0.98;    % Mutation Coefficient Damping Ratio
m=ffParameters.m; %2;

delta=0.05*(VarMax-VarMin);     % Uniform Mutation Range


if isscalar(VarMin) && isscalar(VarMax)
    dmax = (VarMax-VarMin)*sqrt(nVar);
else
    dmax = norm(VarMax-VarMin);
end

%% Initialization

% Empty Firefly Structure
firefly.Position=[];
firefly.Cost=[];

% Initialize Population Array
pop=repmat(firefly,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;

%Create Initial Fireflies
for i=1:nPop
  pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
   
   
   %pop(i).Cost=CostFunction(pop(i).Position);
   [solFitness_M, ~,~]=feval(fnc,pop(i).Position,caseStudyData, otherParameters.otherParametersone,10);
   [S_val, ~]=min(solFitness_M,[],2); %Choose the solution with worse performance
   pop(i).Cost=S_val;
   
   
   if pop(i).Cost<=BestSol.Cost
       BestSol=pop(i);
   end
end

%%No sense vectorize
% minPositionsMatrix=repmat(low_habitat_limit,nPop,1);
% maxPositionsMatrix=repmat(up_habitat_limit,nPop,1);
% 
% % generate initial population.
% Position=genpop(nPop,nVar,minPositionsMatrix,maxPositionsMatrix);
% [solFitness_M, ~,~]=feval(fnc,Position,caseStudyData, otherParameters,10);
% [Cost, ~]=min(solFitness_M,[],2); %Choose the solution with worse performance
% [~,BestSol_index] = min(Cost); 


% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% Firefly Algorithm Main Loop
eval=100;
for it=1:MaxIt
    
    newpop=repmat(firefly,nPop,1);
    for i=1:nPop
        newpop(i).Cost = inf;
        for j=1:nPop
            if pop(j).Cost < pop(i).Cost
                
                rij=(norm(pop(i).Position-pop(j).Position)/dmax);
                beta=beta0*exp(-gamma*rij^m);
                e=delta.*unifrnd(-1,+1,VarSize);
                %e=delta*randn(VarSize);
                
                                            
                            % Original FireFly (very bad performance in this problem)
%                              newsol.Position = pop(i).Position ...
%                                 + beta*rand(VarSize).*(pop(j).Position-pop(i).Position) ...
%                                 + alpha*e;

  
                            %Fire fly with random alpha and random parameters (36 rank index more or less)
                            %Inclusion of (1+rand(1,nVar)) in the main operator
                newsol.Position = pop(i).Position ...
                                + beta*rand(VarSize).*(pop(j).Position-pop(i).Position.*(1+rand(1,nVar))) ...
                                + normrnd(alpha,1).*e;
                             
                %Boundary control 
                newsol.Position=max(newsol.Position,VarMin);
                newsol.Position=min(newsol.Position,VarMax);
                
                %newsol.Cost=CostFunction(newsol.Position);% cambiar
                [solFitness_M, ~,Struct_Eval_temp]=feval(fnc,newsol.Position,caseStudyData, otherParameters.otherParametersone,10);
                [S_val, ~]=min(solFitness_M,[],2); %Choose the solution with worse performance
                newsol.Cost=S_val;
                eval=eval+10;
   
                if newsol.Cost <= newpop(i).Cost
                    newpop(i) = newsol;
                    if newpop(i).Cost<=BestSol.Cost
                        BestSol=newpop(i);
                    end
                end
                
            end
              if eval>49999
                    break;
              end
        end
        if eval>49999
             break;
        end
    end
    
    
    % Merge
    pop=[pop
         newpop];  %#ok
    
    % Sort
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    % Truncate
    pop=pop(1:nPop);
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    %Only parameter that changes
    % Damp Mutation Coefficient
    alpha = alpha*alpha_damp;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Random parameters
    %Fire fly with random alpha and random parameters (36 rank index more or less)
    %Modification to give randomness
    gamma=0.1+rand()*0.9; %1;            % Light Absorption Coefficient
    beta0=0.1+rand()*0.9; %2;            % Attraction Coefficient Base Value
    %alpha=ffParameters.alpha; %0.2;          % Mutation Coefficient
    %alpha_damp=ffParameters.alpha_damp; %0.98;    % Mutation Coefficient Damping Ratio
    %alpha = 0.1+rand()*0.9;
    m=0.1+rand()*2; %2;
    delta=rand()*(VarMax-VarMin);     % Uniform Mutation Range
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

     %% update best results
    I_best_index = 1;
    FVr_bestmemit = pop(1).Position; % best member of current iteration
    % store fitness evolution and obj fun evolution as well
   % fitMaxVector(1,it) = pop(1).Cost;
  
   % if ismember(I_best_index,1)
        % store other info
        Best_otherInfo.idBestParticle = I_best_index;
        Best_otherInfo.genCostsFinal = Struct_Eval_temp(1).otherParameters.genCosts(I_best_index,:);
        Best_otherInfo.loadDRcostsFinal = Struct_Eval_temp(1).otherParameters.loadDRcosts(I_best_index,:);
        Best_otherInfo.v2gChargeCostsFinal = Struct_Eval_temp(1).otherParameters.v2gChargeCosts(I_best_index,:);
        Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval_temp(1).otherParameters.v2gDischargeCosts(I_best_index,:);
        Best_otherInfo.storageChargeCostsFinal = Struct_Eval_temp(1).otherParameters.storageChargeCosts(I_best_index,:);
        Best_otherInfo.storageDischargeCostsFinal = Struct_Eval_temp(1).otherParameters.storageDischargeCosts(I_best_index,:);
        Best_otherInfo.stBalanceFinal = Struct_Eval_temp(1).otherParameters.stBalance(I_best_index,:,:);
        Best_otherInfo.v2gBalanceFinal = Struct_Eval_temp(1).otherParameters.v2gBalance(I_best_index,:,:);
        Best_otherInfo.penSlackBusFinal = Struct_Eval_temp(1).otherParameters.penSlackBus(I_best_index,:);
       % fitMaxVector(:,it)= [mean(solFitness_M_temp(I_best_index,:));mean(solPenalties_M_temp(I_best_index,:))];
    
    
    if eval>49999
         %---end while ((I_iter < I_itermax) ...
        p1=0;%sum(Best_otherInfo.penSlackBusFinal);
       
        fitMaxVector= BestCost';
        Fit_and_p=[fitMaxVector(1,it) p1]; %;p2;p3;p4]
        break;
    end
    
    
    
end

% VECTORIZED THE CODE INSTEAD OF USING FOR
function pop=genpop(a,b,lowMatrix,upMatrix)
pop=unifrnd(lowMatrix,upMatrix,a,b);

% VECTORIZED THE CODE INSTEAD OF USING FOR
function p=update(p,lowMatrix,upMatrix,BRM,FM_base)
switch BRM
    case 1 %Our method
        %[popsize,dim]=size(p);
        [idx] = find(p<lowMatrix);
        p(idx)=lowMatrix(idx);
        [idx] = find(p>upMatrix);
        p(idx)=upMatrix(idx);
    case 2 %Random reinitialization
        [idx] = [find(p<lowMatrix);find(p>upMatrix)];
        replace=unifrnd(lowMatrix(idx),upMatrix(idx),length(idx),1);
        p(idx)=replace;
    case 3 %Bounce Back
      [idx] = find(p<lowMatrix);
      p(idx)=unifrnd(lowMatrix(idx),FM_base(idx),length(idx),1);
        [idx] = find(p>upMatrix);
      p(idx)=unifrnd(FM_base(idx), upMatrix(idx),length(idx),1);
end

%% Results

% figure;
% %plot(BestCost,'LineWidth',2);
% semilogy(BestCost,'LineWidth',2);
% xlabel('Iteration');
% ylabel('Best Cost');
% grid on;
