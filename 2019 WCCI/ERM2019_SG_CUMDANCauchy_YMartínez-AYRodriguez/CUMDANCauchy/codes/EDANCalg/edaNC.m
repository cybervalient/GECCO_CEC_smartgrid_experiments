%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function:         [FVr_bestmem,S_bestval,I_nfeval] = deopt(fname,S_struct)
% Author:           Rainer Storn, Ken Price, Arnold Neumaier, Jim Van Zandt
% Modified by FLC \GECAD 04/winter/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD - GECCO and CEC 2019 Competition: Evolutionary Computation in Uncertain Environments: A Smart Grid Application 
%
% TEAM UC/CISECE-UT3/UCLV
% CUMDANCauchy-C1: a Cellular EDA Designed to Solve the Energy Resource Management Problem Under Uncertainty
%
%
% Phd Student Yoan Martinez Lopez, yoan.martinez@reduc.edu.cu [1,3]
% Phd Ansel Y. Rodriguez-Gonzalez, ansel@cicese.mx [2]
% Phd Julio Madera, julio.madera@reduc.edu.cu [1]
% BSc Student Alexis Moya, alextkmoya@gmail.com [1]
% BSc Student Bismay Morgado Perez, bismaymp@gmail.com [1]
% BSc Student Miguel Betancourt Mayedo, miguel.betancourt@reduc.edu.cu [1]
%
% [1] UC (Universidad de Camaguey, Cuba)
% [2] CISECE-UT3 (Unidad de Transferencia Tecnológica Tepic del Centro de Investigación Científica y de Educación Superior de Ensenada, Mexico)
% [3] UCLV (Universidad de Central de las Villas, Cuba)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    eda(deParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)

%-----This is just for notational convenience and to keep the code uncluttered.--------
I_NP         = deParameters.I_NP;
F_weight     = deParameters.F_weight;
F_CR         = deParameters.F_CR;
I_D          = numel(up_habitat_limit); %Number of variables or dimension
deParameters.nVariables=I_D;
FVr_minbound = low_habitat_limit;
FVr_maxbound = up_habitat_limit;
I_itermax    = deParameters.I_itermax;

%Repair boundary method employed
BRM=deParameters.I_bnd_constr; %1: bring the value to bound violated
                               %2: repair in the allowed range

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_strategy   = deParameters.I_strategy; %important variable
fnc= otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----Check input variables---------------------------------------------
if (I_NP < 5)
    I_NP=5;
   fprintf(1,' I_NP increased to minimal value 5\n');
end
%if ((F_CR < 0) || (F_CR > 1))
 %  F_CR=0.5;
  % fprintf(1,'F_CR should be from interval [0,1]; set to default value 0.5\n');
%end
if (I_itermax <= 0)
   I_itermax = 200;
   fprintf(1,'I_itermax should be > 0; set to default value 200\n');
end

%-----Initialize population and some arrays-------------------------------
%FM_pop = zeros(I_NP,I_D); %initialize FM_pop to gain speed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-allocation of loop variables
fitMaxVector = nan(2,I_itermax);
% limit iterations by threshold
gen = 1; %iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----FM_pop is a matrix of size I_NPx(I_D+1). It will be initialized------
%----with random values between the min and max values of the-------------
%----parameters-----------------------------------------------------------
% FLC modification - vectorization
minPositionsMatrix=repmat(FVr_minbound,I_NP,1);
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1);
deParameters.minPositionsMatrix=minPositionsMatrix;
deParameters.maxPositionsMatrix=maxPositionsMatrix;

% generate initial population.
FM_pop=genpop(I_NP,I_D,minPositionsMatrix,maxPositionsMatrix);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------Evaluate the best member after initialization----------------------
% Modified by FLC
switch fnc
    case 'fitnessFun_DER'
    caseStudyData=caseStudyData(1);
end

[solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,FM_pop,caseStudyData, otherParameters,10);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The user should decide which is the criterion to optimize. 
% In this example, we optimize worse performance
%% Worse performance criterion
[S_val, worstS]=max(solFitness_M,[],2); %Choose the solution with worse performance
[~,I_best_index] = min(S_val); % This mean that the best individual correspond to the best worst performance
FVr_bestmemit = FM_pop(I_best_index,:); % best member of current iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% store other information
Best_otherInfo.idBestParticle = I_best_index;
Best_otherInfo.genCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.genCosts(I_best_index,:);
Best_otherInfo.loadDRcostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.loadDRcosts(I_best_index,:);
Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gChargeCosts(I_best_index,:);
Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(worstS(I_best_index)).otherParameters.v2gDischargeCosts(I_best_index,:);
Best_otherInfo.storageChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageChargeCosts(I_best_index,:);
Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageDischargeCosts(I_best_index,:);
Best_otherInfo.stBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.stBalance(I_best_index,:,:);
Best_otherInfo.v2gBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gBalance(I_best_index,:,:);
Best_otherInfo.penSlackBusFinal = Struct_Eval(worstS(I_best_index)).otherParameters.penSlackBus(I_best_index,:);
fitMaxVector(:,1)=[mean(solFitness_M(I_best_index,:));mean(solPenalties_M(I_best_index,:))]; %We save the mean value and mean penalty value
% The user can decide to save the mean, best, or any other value here

%------EDA-Minimization---------------------------------------------
%------FM_popold is the population which has to compete. It is--------
%------static through one iteration. FM_pop is the newly--------------
%------emerging population.----------------------------------------
%FVr_rot  = (0:1:I_NP-1);               % rotating index array (size I_NP)
while gen<=I_itermax %%&&  fitIterationGap >= threshold
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %[FM_ui,FM_base,~]=generate_trial(I_strategy,F_weight, F_CR, FM_pop, FVr_bestmemit,I_NP, I_D, FVr_rot);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    FM_ui = learningEDA(FM_pop,I_NP);

    %% Boundary Control
     FM_ui = update_eda(FM_ui,minPositionsMatrix,maxPositionsMatrix,BRM);  
    %Evaluation of new Pop
    [solFitness_M_temp, solPenalties_M_temp,Struct_Eval_temp]=feval(fnc,FM_ui,caseStudyData, otherParameters,10);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% The user should decide which is the criterion to optimize. 
    % In this example, we optimize worse performance
    [S_val_temp, worstS]=max(solFitness_M_temp,[],2); %Choose the solution with worse performance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Elitist Selection
    ind=find(S_val_temp<S_val);
    S_val(ind)=S_val_temp(ind);
    FM_pop(ind,:)=FM_ui(ind,:);
  
  
    %% update best results
    [S_bestval,I_best_index] = min(S_val);
    FVr_bestmemit = FM_pop(I_best_index,:); % best member of current iteration
    % store fitness evolution and obj fun evolution as well
    fitMaxVector(1,gen) = S_bestval;
  
    if ismember(I_best_index,ind)
        % store other info
        Best_otherInfo.idBestParticle = I_best_index;
        Best_otherInfo.genCostsFinal = Struct_Eval_temp(worstS(I_best_index)).otherParameters.genCosts(I_best_index,:);
        Best_otherInfo.loadDRcostsFinal = Struct_Eval_temp(worstS(I_best_index)).otherParameters.loadDRcosts(I_best_index,:);
        Best_otherInfo.v2gChargeCostsFinal = Struct_Eval_temp(worstS(I_best_index)).otherParameters.v2gChargeCosts(I_best_index,:);
        Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval_temp(worstS(I_best_index)).otherParameters.v2gDischargeCosts(I_best_index,:);
        Best_otherInfo.storageChargeCostsFinal = Struct_Eval_temp(worstS(I_best_index)).otherParameters.storageChargeCosts(I_best_index,:);
        Best_otherInfo.storageDischargeCostsFinal = Struct_Eval_temp(worstS(I_best_index)).otherParameters.storageDischargeCosts(I_best_index,:);
        Best_otherInfo.stBalanceFinal = Struct_Eval_temp(worstS(I_best_index)).otherParameters.stBalance(I_best_index,:,:);
        Best_otherInfo.v2gBalanceFinal = Struct_Eval_temp(worstS(I_best_index)).otherParameters.v2gBalance(I_best_index,:,:);
        Best_otherInfo.penSlackBusFinal = Struct_Eval_temp(worstS(I_best_index)).otherParameters.penSlackBus(I_best_index,:);
        fitMaxVector(:,gen)= [mean(solFitness_M_temp(I_best_index,:));mean(solPenalties_M_temp(I_best_index,:))];
    elseif gen>1
        fitMaxVector(:,gen)=fitMaxVector(:,gen-1);
    end
    gen=gen+1;

end %---end while ((I_iter < I_itermax) ...
p1=sum(Best_otherInfo.penSlackBusFinal);
Fit_and_p=[fitMaxVector(1,gen-1) p1]; %;p2;p3;p4]


 
% VECTORIZED THE CODE INSTEAD OF USING FOR
function pop=genpop(a,b,lowMatrix,upMatrix)
 pop=unifrnd(lowMatrix,upMatrix,a,b);

% VECTORIZED THE CODE INSTEAD OF USING FOR
 function p=update_eda(p,lowMatrix,upMatrix,BRM)
  switch BRM
    case 1 % max and min replace
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
      p(idx)=unifrnd(lowMatrix(idx),p(idx),length(idx),1);
        [idx] = find(p>upMatrix);
      p(idx)=unifrnd(p(idx), upMatrix(idx),length(idx),1);
   end


function pop_eda = learningEDA(pop_u,I_NP)
 %Cauchy's distribution
 %var_means = mean(pop_u);
 %var_sigma = std(pop_u);
 %m = length(var_means);
 %pop_eda = sin(pop_u)-exp(pop_u);%28.864
 mu = mean(pop_u);
 sd = std(pop_u);
for i=1:I_NP
 %pop_eda(i,:) = -normrnd(mu,sd).*exp(pop_u(i,:));
 %pop_eda(i,:) = - lognrnd(mu,sd).*exp(pop_u(i,:));
 %pop_eda(i,:) = - chi2rnd(2)*exp(pop_u(i,:));
 pop_eda(i,:) = -normrnd(mu,sd).*(mu - sd*tan(pi*(rand(1,1))-0.5));
 %pop_eda(i,:) = - lognrnd(mu,sd).*(mu - sd*tan(pi*(rand(1,1))-0.5));
 %pop_eda(i,:) = chi2rnd(2)*(mu - sd*tan(pi*(rand(1,1))-0.5));
end
 %pop_eda = cos(pop_u)-exp(pop_u);31.65
 %pop_eda = log(pop_u);
 %sigma = std(pop_u);
 %mu = mean(pop_u);
 %sd = std(pop_u);
 %m = length(sd);
 %mu = mean(pop_u);
 %fu = sin(mu+sd)-exp(mu+sd);
 %fu = -exp(mu+sd);
 %va = var(pop_u);
 %m = length(mu);
 %xmin = min(pop_u);
 %xmax = max(pop_u);
 %Gaussian multiva
 %vars_cov = cov(sin(pop_u)-exp(pop_u));%34.057  
 %for i=1:I_NP
  %pop_eda(i,:) = vars_cov(i,:);
  %pop_eda(i,:) = var_means - var_sigma*tan(pi*(rand(m,1))-0.5); %estimation of distribution of cauchy--- 24.862
  %pop_eda(i,:) = fu*rand(1,1); %24.262
  %pop_eda(i,:)= fu*rand(1,1);%24.193
 %end
 
