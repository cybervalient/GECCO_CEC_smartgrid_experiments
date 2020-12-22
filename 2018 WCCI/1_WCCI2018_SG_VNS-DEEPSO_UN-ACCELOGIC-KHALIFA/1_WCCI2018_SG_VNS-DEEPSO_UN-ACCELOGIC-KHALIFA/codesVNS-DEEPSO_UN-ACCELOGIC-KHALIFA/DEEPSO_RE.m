%% TEAM: UN-ACCELOGIC-KHALIFA
% Cooperation of Universidad Nacional de Colombia (UN), ACCELOGIC and Khalifa University
%% TEAM MEMBERS: 
% Sergio Rivera, srriverar@unal.edu.co, professor at UN
% Pedro Garcia, pjgarciag@unal.edu.co, PhD Student at UN and Researcher at Servicio Nacional de Aprendizaje SENA Ocaña, Colombia
% Julian Cantor, jrcantorl@unal.edu.co, Graduated from UN
% Juan Gonzalez, juan.gonzalez@accelogic.com, Chief Scientist at ACCELOGIC
% Rafael Nunez, rafael.nunez@accelogic.com, Vice President Research and Development at ACCELOGIC
% Camilo Cortes, caacortesgu@unal.edu.co, professor at UN
% Alejandra Guzman, maguzmanp@unal.edu.co, professor at UN
% Ameena Al Sumaiti  ameena.alsumaiti@ku.ac.ae, professor at Khalifa University
%% ALGORITMH: VNS-DEEPSO
% Combination of Variable Neighborhood Search algorithm (VNS) and Differential Evolutionary Particle Swarm Optimization (DEEPSO)
%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/

function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    DEEPSO_RE(deParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit,iter,xbest)
%%    
%-----This is just for notational convenience and to keep the code uncluttered.--------
I_NP         = deParameters.I_NP;
F_weight     = deParameters.F_weight;
F_CR         = deParameters.F_CR;
I_D          = numel(up_habitat_limit); %Number of variables or dimension
deParameters.nVariables=I_D;
FVr_minbound = low_habitat_limit;
FVr_maxbound = up_habitat_limit;
I_itermax    = deParameters.I_itermax;
I_iterma    = deParameters.I_iterma;

%Repair boundary method employed
BRM=deParameters.I_bnd_constr; %1: bring the value to bound violated
                               %2: repair in the allowed range

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_strategy   = deParameters.I_strategy; %important variable
fnc= otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----Check input variables---------------------------------------------
% if (I_NP < 5)
%    I_NP=5;
%    fprintf(1,' I_NP increased to minimal value 5\n');
% end
% if ((F_CR < 0) || (F_CR > 1))
%    F_CR=0.5;
%    fprintf(1,'F_CR should be from interval [0,1]; set to default value 0.5\n');
% end
% if (I_itermax <= 0)
%    I_itermax = 200;
%    fprintf(1,'I_itermax should be > 0; set to default value 200\n');
% end

%-----Initialize population and some arrays-------------------------------
%FM_pop = zeros(I_NP,I_D); %initialize FM_pop to gain speed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-allocation of loop variables
fitMaxVector = nan(2,I_iterma);
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
%FM_pop=genpop(I_NP,I_D,minPositionsMatrix,maxPositionsMatrix);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------Evaluate the best member after initialization----------------------
% Modified by FLC
switch fnc
    case 'fitnessFun_DER'
    caseStudyData=caseStudyData(1);
end
%%

% INITIALIZE strategic parameters of DEEPSO
    global deepso_par;
    global ff_par;
    [deepso_par,ff_par,set_par] = DEEPSO_SETTINGS(low_habitat_limit,up_habitat_limit,deParameters);
    nEvals=iter;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE generation counter
countGen = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RANDOMLY INITIALIZE CURRENT population

Vmin = -set_par.Xmax + set_par.Xmin;
Vmax = -Vmin;
pos = zeros( set_par.pop_size, set_par.D );
vel = zeros( set_par.pop_size, set_par.D );
for i = 1 : set_par.pop_size
    pos( i, : ) = set_par.Xmin + ( set_par.Xmax - set_par.Xmin ) .* rand( 1, set_par.D );
    vel( i, : ) = Vmin + ( Vmax - Vmin ) .* rand( 1, set_par.D );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE strategic parameters of DEEPSO
communicationProbability = deepso_par.communicationProbability;
mutationRate = deepso_par.mutationRate;
% Weights matrix
% 1 - inertia
% 2 - memory
% 3 - cooperation
% 4 - perturbation
weights = rand( set_par.pop_size, 4 );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EVALUATE the CURRENT population

pos(1,:)=xbest;
pos(2,:)=low_habitat_limit;
% pos(3,:)=up_habitat_limit;
% pos(4,:)=low_habitat_limit+((up_habitat_limit-low_habitat_limit)/2);

[solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,deParameters.Scenarios);

            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);

for i=1:set_par.pop_size   % Eval Initial population
            fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:));
end

%[ fit, ~, ~, pos, ~ ] = feval( fhd, ii, jj, kk, args, pos );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UPDATE GLOBAL BEST
[ gbestval, gbestid ] = min( fit );
gbest = pos( gbestid, : );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number particles saved in the memory of the DEEPSO
memGBestMaxSize = deepso_par.memGBestMaxSize;
memGBestSize = 1;
% Memory of the DEEPSO
memGBest( memGBestSize, : ) = gbest;
memGBestFit( 1, memGBestSize ) = gbestval;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% The user should decide which is the criterion to optimize. 
% In this example, we optimize worse performance
% Worse performance criterion
[S_val, worstS]=max(solFitness_M,[],2); %Choose the solution with worse performance

%[~,I_best_index] = min(S_val); % This mean that the best individual correspond to the best worst performance
I_best_index=gbestid;
FVr_bestmemit = pos(I_best_index,:); % best member of current iteration
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

%%
nEvals+(2*(deParameters.I_NP*deParameters.Scenarios))
set_par.nEvals_Max

while nEvals+(2*(deParameters.I_NP*deParameters.Scenarios))<set_par.nEvals_Max
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % COPY CURRENT population
    copyPos = pos;
    copyVel = vel;
    copyWeights = weights;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE MEMORY
    tmpMemGBestSize = memGBestSize + set_par.pop_size;
    tmpMemGBestFit = cat( 2, memGBestFit, fit );
    tmpMemGBest = cat( 1, memGBest, pos );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if rand() > deepso_par.localSearchProbability;
        for i = 1 : set_par.pop_size
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DEEPSO movement rule
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW VELOCITY for the particles of the CURRENT population
            vel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability, set_par.D);
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
            copyVel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ), communicationProbability, set_par.D );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW POSITION for the particles of the COPIED population
            [ copyPos( i, : ), copyVel( i, : ) ] = COMPUTE_NEW_POS( copyPos( i, : ), copyVel( i, : ) );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the COPIED population
        [ copyPos, copyVel ] = ENFORCE_POS_LIMITS( copyPos, set_par.Xmin, set_par.Xmax, copyVel, Vmin, Vmax, set_par.pop_size, set_par.D  );
        %copyPos=update1(copyPos,minPositionsMatrix,maxPositionsMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the CURRENT population
        [ pos, vel ] = ENFORCE_POS_LIMITS( pos, set_par.Xmin, set_par.Xmax, vel, Vmin, Vmax, set_par.pop_size, set_par.D  );
        %pos=update1(pos,minPositionsMatrix,maxPositionsMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the COPIED population
        
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,copyPos,caseStudyData, otherParameters,deParameters.Scenarios);

            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);

            for i=1:set_par.pop_size   % Eval Initial population
                        copyFit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:));
            end
           
        %[ copyFit, ~, ~, copyPos, ~ ] = feval( fhd, ii, jj, kk, args, copyPos );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the CURRENT population
        
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,deParameters.Scenarios);

            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);

            for i=1:set_par.pop_size   % Eval Initial population
                        fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:));
            end
      
        %[ fit, ~, ~, pos, ~ ] = feval( fhd, ii, jj, kk, args, pos );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CREATE NEW population to replace CURRENT population
        selParNewSwarm = ( copyFit < fit );
        for i = 1 : set_par.pop_size
            if selParNewSwarm( i )
                fit( i ) = copyFit( i );
                pos( i, : ) = copyPos( i, : );
                vel( i, : ) = copyVel( i, : );
                weights( i, : ) = copyWeights( i, : );
                
                %C(i)=copyC(i);
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Local Search
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW POSITION for the particles of the COPIED population using LOCAL SEARCH
        for i=1 : set_par.pop_size
        pos( i, : ) = LOCAL_SEARCH( pos( i, : ), set_par.Xmin, set_par.Xmax , set_par.D);
        end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the CURRENT population
        [ pos, vel ] = ENFORCE_POS_LIMITS( pos, set_par.Xmin, set_par.Xmax, vel, Vmin, Vmax , set_par.pop_size, set_par.D  );
        %pos=update1(pos,minPositionsMatrix,maxPositionsMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW POSITION for the particles of the COPIED population using LOCAL SEARCH
        for i=1 : set_par.pop_size
        copyPos( i, : ) = LOCAL_SEARCH( copyPos( i, : ), set_par.Xmin, set_par.Xmax , set_par.D);
        end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % ENFORCE search space limits of the COPIED population
        [ copyPos, copyVel ] = ENFORCE_POS_LIMITS( copyPos, set_par.Xmin, set_par.Xmax, copyVel, Vmin, Vmax, set_par.pop_size, set_par.D  );
        %copyPos=update1(copyPos,minPositionsMatrix,maxPositionsMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the COPIED population
        
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,copyPos,caseStudyData, otherParameters,deParameters.Scenarios);

            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);

            for i=1:set_par.pop_size   % Eval Initial population
                        copyFit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:));
            end
           
        %[ copyFit, ~, ~, copyPos, ~ ] = feval( fhd, ii, jj, kk, args, copyPos );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the CURRENT population
        
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters,deParameters.Scenarios);

            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);

            for i=1:set_par.pop_size   % Eval Initial population
                        fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:));
            end
      
        %[ fit, ~, ~, pos, ~ ] = feval( fhd, ii, jj, kk, args, pos );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CREATE NEW population to replace CURRENT population
        selParNewSwarm = ( copyFit < fit );
        for i = 1 : set_par.pop_size
            if selParNewSwarm( i )
                fit( i ) = copyFit( i );
                pos( i, : ) = copyPos( i, : );
                vel( i, : ) = copyVel( i, : );
                weights( i, : ) = copyWeights( i, : );
                
                %C(i)=copyC(i);
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE GLOBAL BEST
    [ tmpgbestval, gbestid ] = min( fit );
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
%% The user should decide which is the criterion to optimize. 
% In this example, we optimize worse performance
% Worse performance criterion
[S_val, worstS]=max(solFitness_M,[],2); %Choose the solution with worse performance

%[~,I_best_index] = min(S_val); % This mean that the best individual correspond to the best worst performance
I_best_index=gbestid;
FVr_bestmemit = pos(I_best_index,:); % best member of current iteration
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
gen=gen + 1;
fitMaxVector(:,gen)=[mean(solFitness_M(I_best_index,:));mean(solPenalties_M(I_best_index,:))]; %We save the mean value and mean penalty value
% The user can decide to save the mean, best, or any other value here
%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % RE-CALCULATES NEW COEFFICIENTS for the fitness function
    CALC_COEFS_FF();
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE generation counter
    countGen = countGen + 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [qw,er]=min(fit);
        disp(countGen);
        disp(min(fit))
        disp(gbestval)
        %disp(qw);
        disp(nEvals);
        %disp(C(er));
    %%%%%%%%%%%%%%
end
p1=sum(Best_otherInfo.penSlackBusFinal);
Fit_and_p=[fitMaxVector(1,gen-1) p1]; %;p2;p3;p4]

% er=size(fitMaxVector);
% err=er(1,2);
% err0=err-400;
% 
% fitMaxVector=fitMaxVector(:,err0:err);
           
end			 
           

    
  
               
 

