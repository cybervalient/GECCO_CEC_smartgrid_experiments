% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2014 OPF problems (Competition & panel): Differential Evolutionary Particle Swarm Optimization (DEEPSO)  
% http://sites.ieee.org/psace-mho/panels-and-competitions-2014-opf-problems/

function [Fit_and_p,FVr_bestmemit, fitMaxVector] = ...
    DEEPSO_RE(deParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fnc= otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INITIALIZE strategic parameters of DEEPSO
    global deepso_par;
    global ff_par;
    [deepso_par,ff_par,set_par,NoCor] = DEEPSO_SETTINGS(low_habitat_limit,up_habitat_limit,deParameters);
    nEvals=0;

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

% pos(1,:)=xbest;
% pos(1,:)=low_habitat_limit;
% pos(1,:)=up_habitat_limit;
% pos(2,:)=low_habitat_limit+((up_habitat_limit-low_habitat_limit)/2);

            iter=0;
                        
            solFit=zeros(deParameters.I_NP,1);
            for ui=1:NoCor
            
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters);
            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);
            solFit=solFit+solFitness_M;
            
            end
            solFit=solFit/NoCor;
            
            iter=iter+1;
            
for i=1:set_par.pop_size   % Eval Initial population
            fit(i)  = solFit(i);
end

%[ fit, ~, ~, pos, ~ ] = feval( fhd, ii, jj, kk, args, pos );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UPDATE GLOBAL BEST
[ gbestval, gbestid ] = min( fit );
gbest = pos( gbestid, : );
[ gworstval, gworstid ] = max( fit );
gworst = pos( gworstid, : );
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
fitMaxVector(iter)=gbestval; %We save the mean value and mean penalty value
% The user can decide to save the mean, best, or any other value here

%%
nEvals+(2*(deParameters.I_NP*deParameters.Scenarios))
set_par.nEvals_Max

while nEvals+(2*(deParameters.I_NP*deParameters.Scenarios*NoCor))-1<set_par.nEvals_Max
    
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
            %vel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability, set_par.D);
            vel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability, set_par.D);
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
            %copyVel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ), communicationProbability, set_par.D );
            copyVel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ), communicationProbability, set_par.D );
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
                    
            solFit=zeros(deParameters.I_NP,1);
            for ui=1:NoCor
            
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,copyPos,caseStudyData, otherParameters);
            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);
            solFit=solFit+solFitness_M;
            
            end
            solFit=solFit/NoCor;
                        
            for i=1:set_par.pop_size   % Eval Initial population
                        copyFit(i)  = solFit(i);
            end
           
        %[ copyFit, ~, ~, copyPos, ~ ] = feval( fhd, ii, jj, kk, args, copyPos );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the CURRENT population
           
            solFit=zeros(deParameters.I_NP,1);
            for ui=1:NoCor
            
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters);
            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);
            solFit=solFit+solFitness_M;
            
            end
            solFit=solFit/NoCor;
            
            for i=1:set_par.pop_size   % Eval Initial population
                        fit(i)  = solFit(i);
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
%         for i=1 : set_par.pop_size
%         pos( i, : ) = LOCAL_SEARCH( pos( i, : ), set_par.Xmin, set_par.Xmax , set_par.D);
%         end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
            pos = learningEDA(pos,deParameters.I_NP);
        
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
        
        copyPos = learningEDA(copyPos,deParameters.I_NP);
        
        % ENFORCE search space limits of the COPIED population
        [ copyPos, copyVel ] = ENFORCE_POS_LIMITS( copyPos, set_par.Xmin, set_par.Xmax, copyVel, Vmin, Vmax, set_par.pop_size, set_par.D  );
        %copyPos=update1(copyPos,minPositionsMatrix,maxPositionsMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the COPIED population
        
            solFit=zeros(deParameters.I_NP,1);
            for ui=1:NoCor
            
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,copyPos,caseStudyData, otherParameters);
            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);
            solFit=solFit+solFitness_M;
            
            end
            solFit=solFit/NoCor;

            for i=1:set_par.pop_size   % Eval Initial population
                        copyFit(i)  = solFit(i);
            end
           
        %[ copyFit, ~, ~, copyPos, ~ ] = feval( fhd, ii, jj, kk, args, copyPos );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % EVALUATE the CURRENT population
            solFit=zeros(deParameters.I_NP,1);
            for ui=1:NoCor
            
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters);
            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);
            solFit=solFit+solFitness_M;
            
            end
            solFit=solFit/NoCor;

            for i=1:set_par.pop_size   % Eval Initial population
                        fit(i)  = solFit(i);
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
    %
    [ gworstval, gworstid ] = max( fit );
    gworst = pos( gworstid, : );

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The user should decide which is the criterion to optimize. 
iter=iter+1;
fitMaxVector(iter)=gbestval; %We save the mean value and mean penalty value
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
%p1=sum(Best_otherInfo.penSlackBusFinal);
Fit_and_p=[fitMaxVector(iter) 0]; %;p2;p3;p4]
FVr_bestmemit=gbest;
           
	


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
           

    
  
               
 

