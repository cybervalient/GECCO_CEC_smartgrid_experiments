%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Developer: Kartik S. Pandya, PhD (email: kartikpandya.ee@charusat.ac.in)
%Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA

% Improved Chaotic Differential Evolutionary Particle Swarm Optimization (I-C-DEEPSO) algorithm as
% optimization engine to solve WCCI 2018 competition test bed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [globalminimizer, fitMaxVector, objMaxVector, otherParameters] = ...
    %CHAOTIC_DEEPSO(levydeepso,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit,initialSolution)
function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    CHAOTIC_DEEPSO(chaos_DEEPSO_parameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)

Xmin=low_habitat_limit;

% Particles' upper bounds.
Xmax=up_habitat_limit;
pop_size=chaos_DEEPSO_parameters.I_NP;
I_itermax=chaos_DEEPSO_parameters.I_itermax;
fnc= otherParameters.fnc;
%convert binary variables of (3) no. of DG generators and external
%suppliers into continuous
%  for j=0:23
%    for i=155+(j*2080):231+(j*2080)
% 
%         Xmin( 1, i ) = Xmin( 1, i ) - 0.4999;
%         Xmax( 1, i ) = Xmax( 1, i ) + 0.4999;
%    end
% end

%Xmin=ps.x_min;
% Particles' upper bounds.
%Xmax=ps.x_max;

%Find Levy_L constant
global Levy_constant
Levy_constant=(mean(Xmax-Xmin))/100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE strategic parameters of DEEPSO
global deepso_par;
global ff_par;
global D
memGBestMaxSize = ceil( pop_size * 0.2 );% for 100 partilces, it is 20
deepso_par.memGBestMaxSize = ceil(pop_size * 0.2 );
% %pause;
% 
         deepso_par.mutationRate = 0.5;
        deepso_par.communicationProbability = 0.5;
        deepso_par.localSearchProbability = 0.25;
        deepso_par.localSearchContinuousDiscrete = 0.75;
        ff_par.excludeBranchViolations = 0;
        
        ff_par.factor = 1;
        ff_par.numCoefFF = 4;
    
        

ff_par.avgCoefFF = zeros( 1, ff_par.numCoefFF );
ff_par.coefFF = ones( 1, ff_par.numCoefFF );
ff_par.numFFEval = 0;
%pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE generation counter
%countGen = 1;
% limit iterations by threshold
gen = 1; %iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RANDOMLY INITIALIZE CURRENT population
%if proc.system ~= 41 % if system not equal to 41 (ie 57, 118) then
    %for i = 1 + ps.D_cont : ps.D_cont + ps.n_OLTC % OLTC:14:28
        %a=1+13:13+15 means a=14 15 16 17 18.....28
       % Xmin( 1, i ) = Xmin( 1, i ) - 0.4999;
       % Xmax( 1, i ) = Xmax( 1, i ) + 0.4999;
    %end
%for i = 1 + ps.D_cont : ps.D_cont + ps.n_OLTC % OLTC:14:28
   % for i=otherParameters.ids.idsXGen(1):otherParameters.ids.idsXGen(77)
        %a=1+13:13+15 means a=14 15 16 17 18.....28
       % Xmin( 1, i ) = Xmin( 1, i ) - 0.4999;
       % Xmax( 1, i ) = Xmax( 1, i ) + 0.4999;
   % end
Vmin = -Xmax + Xmin;
Vmax = -Vmin;
nvariables=numel(Xmin);
D=nvariables; % D=3408
pos = zeros(pop_size, D);

vel = zeros( pop_size,D);
for i = 1 : pop_size
    %pos( i, : ) = Xmin + ( Xmax - Xmin ) .* rand( 1, D );
    %vel( i, : ) = Vmin + ( Vmax - Vmin ) .* rand( 1, D );
    ccrand=rand(1,D);
    ccpos=((1./ccrand)-floor(1./ccrand));
    
    pos( i, : ) = Xmin + ( Xmax - Xmin ).*ccpos;
    vel( i, : ) = Vmin + ( Vmax - Vmin ).*ccpos;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE strategic parameters of LEVY_DEEPSO
communicationProbability = deepso_par.communicationProbability;
mutationRate = deepso_par.mutationRate;
% Weights matrix
% 1 - inertia
% 2 - memory
% 3 - cooperation
% 4 - perturbation
weights = rand( pop_size, 4 );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EVALUATE the CURRENT population
%[ fit, ~, ~, pos, ~ ] = feval( fhd, ii, jj, kk, args, pos );


%[fit_superorganism, obj_superorganism, pos, otherParameters]=feval(levydeepso.fnc, pos,caseStudyData, otherParameters);

%[solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,FM_pop,caseStudyData, otherParameters,10);

[solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,10);

%[globalminimum,indexbest] = min(fit_superorganism);
%[globalminimum,indexbest] = min(fit);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UPDATE GLOBAL BEST
%[ gbestval, gbestid ] = min( solFitness_M );
[ gbestval, worstS ] = min( solFitness_M, [ ],2 );
[gbestval,gbestid] = min(gbestval);
gbest = pos( gbestid, : );
fit1=gbestval;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%change name as per requirement

I_best_index=gbestid;
FVr_bestmemit=gbest;  % position of gbest 1*3408
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Worse performance criterion
%[S_val, worstS]=max(solFitness_M,[],2); %Choose the solution with worse performance

%[~,I_best_index] = min(S_val); % This mean that the best individual correspond to the best worst performance
%FVr_bestmemit = pos(I_best_index,:); % best member of current iteration
%pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% store other information
Best_otherInfo.idBestParticle = I_best_index; % g of global best particle
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The user can decide to save the mean, best, or any other value here
% store other information
%our gbestid= dsa's indexbest
% otherParameters.idBestParticle = gbestid;
% otherParameters.pfFinal = otherParameters.pfDB(:,gbestid);
% otherParameters.genCostsFinal = otherParameters.genCosts(gbestid,:);
% otherParameters.loadDRcostsFinal = otherParameters.loadDRcosts(gbestid,:);
% otherParameters.v2gChargeCostsFinal = otherParameters.v2gChargeCosts(gbestid,:);
% otherParameters.v2gDischargeCostsFinal =otherParameters.v2gDischargeCosts(gbestid,:);
% otherParameters.storageChargeCostsFinal = otherParameters.storageChargeCosts(gbestid,:);
% otherParameters.storageDischargeCostsFinal = otherParameters.storageDischargeCosts(gbestid,:);
% otherParameters.stBalanceFinal = otherParameters.stBalance(gbestid,:,:);
% otherParameters.v2gBalanceFinal = otherParameters.v2gBalance(gbestid,:,:);
% otherParameters.pensVoltageUFinal =  otherParameters.pensVoltageU(gbestid,:);
% otherParameters.pensVoltageLFinal = otherParameters.pensVoltageL(gbestid,:);
% otherParameters.pensMaxSLinesFinal = otherParameters.pensMaxSLines(gbestid,:);
% otherParameters.penSlackBusFinal = otherParameters.penSlackBus(gbestid,:);
% objMaxVector(:,1)=obj_superorganism(gbestid,:);
%pause;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number particles saved in the memory of the DEEPSO
memGBestMaxSize = deepso_par.memGBestMaxSize;

memGBestSize = 1;
% Memory of the DEEPSO
memGBest( memGBestSize, : ) = gbest;

memGBestFit( 1, memGBestSize ) = gbestval;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % limit iterations by threshold
% fitIterationGap = inf;
% %noIterationsToGap = dsaParameters.noIterationsToGap;
% noIterationsToGap=levydeepso.noIterationsToGap;
% epk = 1;
% epoch=levydeepso.maxIterations;
% %%%%%%%
%while 1
 while gen<=I_itermax
% while epk<=epoch %&&  fitIterationGap >= levydeepso.threshold   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % COPY CURRENT population
    copyPos = pos;
    copyVel = vel;
    copyWeights = weights;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE MEMORY
    tmpMemGBestSize = memGBestSize + pop_size;
    %pause;
    tmpMemGBestFit = cat( 2, memGBestFit, fit1' ); %1*21, 1+20 A=1 2   B=5 6
                                                  %  3 4     7 8
                                            %C=cat(1,A,B)   C=cat(2,A,B)
                                             %B bottom       %B side ma 
                                                  %C=1 2   C=1 2 5 6
                                                  %  3 4     3 4 7 8
                                                  %  5 6
                                                  %  7 8
                                          
    
     tmpMemGBest = cat( 1, memGBest, pos );% it is 21*49920, memGBest:1*49920, pos:20*49920
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if rand() > deepso_par.localSearchProbability;
   
        for i = 1 : pop_size
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DEEPSO movement rule
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW VELOCITY for the particles of the CURRENT population
          %  vel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest, fit1( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability );
            vel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest,  vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability );
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW POSITION for the particles of the CURRENT population
            [ pos( i, : ), vel( i, : ) ] = COMPUTE_NEW_POS( pos( i, : ), vel( i, : ) );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DEEPSO movement rule
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % MUTATE WEIGHTS of the particles of the COPIED population
            copyWeights( i, : ) = MUTATE_WEIGHTS( weights( i, : ), mutationRate );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW VELOCITY for the particles of the COPIED population
          %  copyVel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest, fit1( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ), communicationProbability );
           copyVel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest,   copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ), communicationProbability );
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW POSITION for the particles of the COPIED population
            [ copyPos( i, : ), copyVel( i, : ) ] = COMPUTE_NEW_POS( copyPos( i, : ), copyVel( i, : ) );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the COPIED population
        [ copyPos, copyVel ] = ENFORCE_POS_LIMITS( copyPos, Xmin, Xmax, copyVel, Vmin, Vmax );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the CURRENT population
        [ pos, vel ] = ENFORCE_POS_LIMITS( pos, Xmin, Xmax, vel, Vmin, Vmax );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the COPIED population
       % [ copyFit, ~, ~, copyPos, ~ ] = feval( fhd, ii, jj, kk, args, copyPos );
      % [copyFit, obj_superorganism, copyPos, otherParameters]=feval(levydeepso.fnc, copyPos,caseStudyData, otherParameters);
        [copyFit, solPenalties_M,Struct_Eval]=feval(fnc,copyPos,caseStudyData, otherParameters,10);
       copyFit=min(copyFit, [ ], 2);
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the CURRENT population
        %[ fit, ~, ~, pos, ~ ] = feval( fhd, ii, jj, kk, args, pos );
       % [fit1, obj_superorganism, pos, otherParameters]=feval(levydeepso.fnc, pos,caseStudyData, otherParameters);
       [fit1, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,10);
       fit1=min(fit1, [ ], 2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CREATE NEW population to replace CURRENT population
        selParNewSwarm = ( copyFit < fit1 );
        for i = 1 : pop_size
            if selParNewSwarm( i )
                fit1( i ) = copyFit( i );
                pos( i, : ) = copyPos( i, : );
                vel( i, : ) = copyVel( i, : );
                weights( i, : ) = copyWeights( i, : );
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Local Search
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW POSITION for the particles of the COPIED population using LOCAL SEARCH
        pos( i, : ) = LOCAL_SEARCH( pos( i, : ), Xmin, Xmax );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the CURRENT population
        [ pos, vel ] = ENFORCE_POS_LIMITS( pos, Xmin, Xmax, vel, Vmin, Vmax );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the CURRENT population
       % [ fit, ~, ~, pos, ~ ] = feval( fhd, ii, jj, kk, args, pos );
      % [fit1, obj_superorganism, pos, otherParameters]=feval(levydeepso.fnc, pos,caseStudyData, otherParameters);
       [fit1, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,10); 
      fit1= min( fit1, [ ],2 );
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE GLOBAL BEST
   % [ tmpgbestval, gbestid ] = min( fit1 );
    
    [ tmpgbestval, worstS] = min( fit1, [ ],2 );
    [tmpgbestval,gbestid] =min(tmpgbestval);
    
    if tmpgbestval < gbestval
        gbestval = tmpgbestval;
        gbest = pos( gbestid, : );
        % UPDATE MEMORY DEEPSO
        if memGBestSize < memGBestMaxSize
            memGBestSize = memGBestSize + 1;
            memGBest( memGBestSize, : ) = gbest;
            memGBestFit( 1, memGBestSize ) = gbestval;
        else
            [ ~, tmpgworstid ] = max( memGBestFit );
            memGBest( tmpgworstid, : ) = gbest;
            memGBestFit( 1, tmpgworstid ) = gbestval;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    globalminimum =gbestval;
%    fitMaxVector(epk) = globalminimum;
%    globalminimizer=gbest;
% [ gbestval, worstS ] = min( solFitness_M, [ ],2 );
% [~,gbestid] = min(gbestval);
% gbest = pos( gbestid, : );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%change name as per requirement

I_best_index=gbestid;
%pause;
FVr_bestmemit=gbest;  % position of gbest 1*3408
   %%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% store other information
Best_otherInfo.idBestParticle = I_best_index; % g of global best particle
Best_otherInfo.genCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.genCosts(I_best_index,:);
Best_otherInfo.loadDRcostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.loadDRcosts(I_best_index,:);
Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gChargeCosts(I_best_index,:);
Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(worstS(I_best_index)).otherParameters.v2gDischargeCosts(I_best_index,:);
Best_otherInfo.storageChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageChargeCosts(I_best_index,:);
Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageDischargeCosts(I_best_index,:);
Best_otherInfo.stBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.stBalance(I_best_index,:,:);
Best_otherInfo.v2gBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gBalance(I_best_index,:,:);
Best_otherInfo.penSlackBusFinal = Struct_Eval(worstS(I_best_index)).otherParameters.penSlackBus(I_best_index,:);
%fitMaxVector(:,1)=[mean(solFitness_M(I_best_index,:));mean(solPenalties_M(I_best_index,:))]; %We save the mean value and mean penalty value
fitMaxVector(:,1)=[mean(fit1(I_best_index,:));mean(solPenalties_M(I_best_index,:))];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update results
    %[globalminimum,indexbest]=min(fit_superorganism);
    %globalminimizer=superorganism(indexbest,:);
    
    % store fitness evolution and obj fun evolution as well
   % fitMaxVector(epk) = globalminimum;
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% store other information
%our gbestid= dsa's indexbest
% otherParameters.idBestParticle = gbestid;
% otherParameters.pfFinal = otherParameters.pfDB(:,gbestid);
% otherParameters.genCostsFinal = otherParameters.genCosts(gbestid,:);
% otherParameters.loadDRcostsFinal = otherParameters.loadDRcosts(gbestid,:);
% otherParameters.v2gChargeCostsFinal = otherParameters.v2gChargeCosts(gbestid,:);
% otherParameters.v2gDischargeCostsFinal =otherParameters.v2gDischargeCosts(gbestid,:);
% otherParameters.storageChargeCostsFinal = otherParameters.storageChargeCosts(gbestid,:);
% otherParameters.storageDischargeCostsFinal = otherParameters.storageDischargeCosts(gbestid,:);
% otherParameters.stBalanceFinal = otherParameters.stBalance(gbestid,:,:);
% otherParameters.v2gBalanceFinal = otherParameters.v2gBalance(gbestid,:,:);
% otherParameters.pensVoltageUFinal =  otherParameters.pensVoltageU(gbestid,:);
% otherParameters.pensVoltageLFinal = otherParameters.pensVoltageL(gbestid,:);
% otherParameters.pensMaxSLinesFinal = otherParameters.pensMaxSLines(gbestid,:);
% otherParameters.penSlackBusFinal = otherParameters.penSlackBus(gbestid,:);
% objMaxVector(:,1)=obj_superorganism(gbestid,:);
%     
   % epk
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % RE-CALCULATES NEW COEFFICIENTS for the fitness function
   % CALC_COEFS_FF();
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE generation counter
   % countGen = countGen + 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if epk >= minEpochs
       % fitIterationGap = abs(fitMaxVector(epk)-mean(fitMaxVector(epk-1:-1:epk-1-noIterationsToGap)));
  %  end
  if gen>1
  fitMaxVector(:,gen)=fitMaxVector(:,gen-1);
  end
    gen=gen+1;
    %epk=epk+1;
    %%%%%%%%%%
   % if proc.finish
      %  return;
    %end
    %%%%%%%%%%%%%%
 end
p1=sum(Best_otherInfo.penSlackBusFinal);
Fit_and_p=[fitMaxVector(1,gen-1) p1]; %;p2;p3;p4]
end