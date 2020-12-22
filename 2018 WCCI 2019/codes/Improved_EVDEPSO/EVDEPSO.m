%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Kartik S. Pandya, PhD (email: kartikpandya.ee@charusat.ac.in)
%Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA
%Co-Author: Dharmesh A. Dabhi, PhD(Pursuing) (email: dharmeshdabhi.ee@charusat.ac.in)
%Assistant Professor, Dept. of Electrical Engg., CSPIT, CHRUSAT, Gujarat, INDIA

% Enhanced Velocity Differential Evolutionary Particle Swarm Optimization (EVDEPSO) algorithm as
% optimization engine to solve WCCI 2018 competition test bed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    EVDEPSO(EVDEPSO_parameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)

Xmin=low_habitat_limit;

Xmax=up_habitat_limit;

% INITIALIZE strategic parameters of EVDEPSO
global EVDEPSO_par;
global ff_par;
global D
global pop_size
pop_size=EVDEPSO_parameters.I_NP;
memGBestMaxSize = ceil( pop_size * 0.2 );
% for 100 partilces, it is 20
EVDEPSO_par.memGBestMaxSize = ceil(pop_size * 0.2 );
EVDEPSO_par.mutationRate = 0.7;
EVDEPSO_par.localSearchProbability = 0.3;
EVDEPSO_par.localSearchContinuousDiscrete = 0.9;
ff_par.excludeBranchViolations = 0;
ff_par.factor = 1;
ff_par.numCoefFF = 4;
ff_par.avgCoefFF = zeros( 1, ff_par.numCoefFF );
ff_par.coefFF = ones( 1, ff_par.numCoefFF );
ff_par.numFFEval = 0;
fnc=otherParameters.fnc;
I_itermax=EVDEPSO_parameters.I_itermax;
%pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE generation counter
%countGen = 1;
% limit iterations by threshold
gen = 1; %iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Vmin = -Xmax + Xmin;
Vmax = -Vmin;
nvariables=numel(Xmin);
D=nvariables; % D=3408
pos = zeros(pop_size, D);

vel = zeros( pop_size,D);
for i = 1 : pop_size
     pos( i, : ) = Xmin + ( Xmax - Xmin );%INITIALIZATION OF POSITION OF EACH PARTICLES
    vel( i, : ) = Vmin + ( Vmax - Vmin );%INITIALIZATION OF VELOCITY OF EACH PARTICLES
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE strategic parameters of EVDEPSO
mutationRate = EVDEPSO_par.mutationRate;
% Weights matrix
% 1 - inertia
% 2 - memory
% 3 - cooperation
% 4 - perturbation
weights = rand( pop_size, 4 );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch fnc
    case 'fitnessFun_DER'
    caseStudyData=caseStudyData(1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EVALUATE the CURRENT population
[solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,10);
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
FVr_bestmemit=gbest;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% store other information
Best_otherInfo.idBestParticle = I_best_index; % g of global best particle
Best_otherInfo.genCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.genCosts(I_best_index,:);
Best_otherInfo.loadDRcostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.loadDRcosts(I_best_index,:);
Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gChargeCosts(I_best_index,:);
Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(worstS(I_best_index)).otherParameters.v2gDischargeCosts(I_best_index,:);
Best_otherInfo.storageChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageChargeCosts(I_best_index,:);Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageDischargeCosts(I_best_index,:);
Best_otherInfo.stBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.stBalance(I_best_index,:,:);
Best_otherInfo.v2gBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gBalance(I_best_index,:,:);
Best_otherInfo.penSlackBusFinal = Struct_Eval(worstS(I_best_index)).otherParameters.penSlackBus(I_best_index,:);
fitMaxVector(:,1)=[mean(solFitness_M(I_best_index,:));mean(solPenalties_M(I_best_index,:))]; %We save the mean value and mean penalty value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number particles saved in the memory of the EVDEPSO
memGBestMaxSize = EVDEPSO_par.memGBestMaxSize;

memGBestSize = 1;
% Memory of the EVDEPSO
memGBest( memGBestSize, : ) = gbest;

memGBestFit( 1, memGBestSize ) = gbestval;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % limit iterations by threshold
%while 1
 while gen<=I_itermax

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
    tmpMemGBestFit = cat( 2, memGBestFit, fit1' ); %1*12, 2+10  A=1 2   B=5 6
                                                               %  3 4     7 8
                                            %C=cat(1,A,B)   C=cat(2,A,B)
                                             %B bottom       %B side ma 
                                                  %C=1 2   C=1 2 5 6
                                                  %  3 4     3 4 7 8
                                                  %  5 6
                                                  %  7 8
                                          
    
     tmpMemGBest = cat( 1, memGBest, pos );
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if rand() > EVDEPSO_par.localSearchProbability;
%    
        for i = 1 : pop_size
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % EVDEPSO movement rule

            % COMPUTE NEW VELOCITY for the particles of the CURRENT population
 vel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest,  vel( i, : ), Vmin, Vmax, weights( i, : ) );
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW POSITION for the particles of the CURRENT population
            [ pos( i, : ), vel( i, : ) ] = COMPUTE_NEW_POS( pos( i, : ), vel( i, : ) );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           
            % MUTATE WEIGHTS of the particles of the COPIED population
            copyWeights( i, : ) = MUTATE_WEIGHTS( weights( i, : ), mutationRate );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW VELOCITY for the particles of the COPIED population
           copyVel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest,   copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ) );
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
        [copyFit, solPenalties_M,Struct_Eval]=feval(fnc,copyPos,caseStudyData, otherParameters,10);
       copyFit=min(copyFit, [ ], 2);
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the CURRENT population
       [fit1, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,10);
       fit1=min(fit1, [ ], 2);
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
        % Local Search
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else
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
       [fit1, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,10); 
      fit1= min( fit1, [ ],2 );
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
%change name as per requirement

I_best_index=gbestid;
%pause;
FVr_bestmemit=gbest;  
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
    % UPDATE generation counter
  if gen>1
  fitMaxVector(:,gen)=fitMaxVector(:,gen-1);
  end
    gen=gen+1;
end
p1=sum(Best_otherInfo.penSlackBusFinal);
Fit_and_p=[fitMaxVector(1,gen-1) p1];
end
