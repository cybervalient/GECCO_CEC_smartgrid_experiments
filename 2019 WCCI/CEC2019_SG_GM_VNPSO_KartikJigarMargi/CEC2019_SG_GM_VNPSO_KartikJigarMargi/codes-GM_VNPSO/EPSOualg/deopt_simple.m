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
% The Codes of EPSO were developed by Phillipe Vila�a Gomes
% The codes have been modified by the developers to propose GM_VNPSO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    deopt_simple(deParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)

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
if ((F_CR < 0) || (F_CR > 1))
   F_CR=0.5;
   fprintf(1,'F_CR should be from interval [0,1]; set to default value 0.5\n');
end
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

%------DE-Minimization---------------------------------------------
%------FM_popold is the population which has to compete. It is--------
%------static through one iteration. FM_pop is the newly--------------
%------emerging population.----------------------------------------
FVr_rot  = (0:1:I_NP-1);               % rotating index array (size I_NP)
while gen<=I_itermax %%&&  fitIterationGap >= threshold
    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [FM_ui,FM_base,~]=generate_trial(I_strategy,F_weight, F_CR, FM_pop, FVr_bestmemit,I_NP, I_D, FVr_rot);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

    %% Boundary Control
    FM_ui=update1(FM_ui,minPositionsMatrix,maxPositionsMatrix,BRM,FM_base);

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

function [FM_ui,FM_base,msg]=generate_trial(method,F_weight, F_CR, FM_pop, FVr_bestmemit,I_NP,I_D,FVr_rot)

  FM_popold = FM_pop;                  % save the old population
  FVr_ind = randperm(4);               % index pointer array
  FVr_a1  = randperm(I_NP);                   % shuffle locations of vectors
  FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);     % rotate indices by ind(1) positions
  FVr_a2  = FVr_a1(FVr_rt+1);                 % rotate vector locations
  FVr_rt  = rem(FVr_rot+FVr_ind(2),I_NP);
  FVr_a3  = FVr_a2(FVr_rt+1);                
  FM_pm1 = FM_popold(FVr_a1,:);             % shuffled population 1
  FM_pm2 = FM_popold(FVr_a2,:);             % shuffled population 2
  FM_pm3 = FM_popold(FVr_a3,:);             % shuffled population 3
  FM_mui = rand(I_NP,I_D) < F_CR;  % all random numbers < F_CR are 1, 0 otherwise
  FM_mpo = FM_mui < 0.5;    % inverse mask to FM_mui

    switch method
        case 1,
            FM_ui = FM_pm3 + F_weight*(FM_pm1 - FM_pm2);   % differential variation
            FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;     % crossover
            FM_base = FM_pm3;
            msg=' DE/rand/bin';
        case 2,
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
            %VEC by FLC
            FM_bm=repmat(FVr_bestmemit,I_NP,1);
            FM_ui = FM_popold + F_weight*(FM_bm-FM_popold) + F_weight*(FM_pm1 - FM_pm2);
            FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;
            FM_base = FM_bm;
            msg=' DE/current-to-best/1';
    end
return

