%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ensembled HHO-DEEPSO
% Harris's hawk optimizer (HHO): In this algorithm, Harris' hawks try to catch the rabbit.
% Differential Evolutionary Particle Swarm Optimization (DEEPSO)

% based on http://www.evo-ml.com/2019/03/02/hho/
% based on https://ieeexplore.ieee.org/document/6855877

function [Fit_and_p,FVr_bestmemit, fitMaxVector]=HHOcompetition(deParameters,caseStudyData,otherParameters,lowerB,upperB,Select_testbed)

disp('HHO is now tackling your problem')
lb=lowerB;
ub=upperB;

N=3;
nEvals=0;
NuRun=10;

T=round(50000/(NuRun*N*deParameters.Scenarios));
T=6;%HHO

fnc= otherParameters.fnc;

DD=size(lowerB);
dim=DD(1,2);

% initialize the location and Energy of the rabbit
Rabbit_Location=zeros(1,dim);
Rabbit_Energy=inf;

%Initialize the locations of Harris' hawks
X=initialization(N,dim,ub,lb);

CNVG=zeros(1,T);

t=0; % Loop counter

while t<T
%while    nEvals+(N*(NuRun*deParameters.I_NP*deParameters.Scenarios))
    for i=1:size(X,1)
        % Check boundries
        FU=X(i,:)>ub;FL=X(i,:)<lb;X(i,:)=(X(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
        % fitness of locations
        [fitness,nEvals]=fobj(X(i,:),NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
       
        % Update the location of Rabbit
        if fitness<Rabbit_Energy
            Rabbit_Energy=fitness;
            Rabbit_Location=X(i,:);
        end
    end
    
    E1=2*(1-(t/T)); % factor to show the decreaing energy of rabbit
    % Update the location of Harris' hawks
    for i=1:size(X,1)
        E0=2*rand()-1; %-1<E0<1
        Escaping_Energy=E1*(E0);  % escaping energy of rabbit
        
        if abs(Escaping_Energy)>=1
            %% Exploration:
            % Harris' hawks perch randomly based on 2 strategy:
            
            q=rand();
            rand_Hawk_index = floor(N*rand()+1);
            X_rand = X(rand_Hawk_index, :);
            if q<0.5
                % perch based on other family members
                X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*X(i,:));
            elseif q>=0.5
                % perch on a random tall tree (random site inside group's home range)
                X(i,:)=(Rabbit_Location(1,:)-mean(X))-rand()*((ub-lb)*rand+lb);
            end
            
        elseif abs(Escaping_Energy)<1
            %% Exploitation:
            % Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
            
            %% phase 1: surprise pounce (seven kills)
            % surprise pounce (seven kills): multiple, short rapid dives by different hawks
            
            r=rand(); % probablity of each event
            
            if r>=0.5 && abs(Escaping_Energy)<0.5 % Hard besiege
                X(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X(i,:));
            end
            
            if r>=0.5 && abs(Escaping_Energy)>=0.5  % Soft besiege
                Jump_strength=2*(1-rand()); % random jump strength of the rabbit
                X(i,:)=(Rabbit_Location-X(i,:))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));
            end
            
            %% phase 2: performing team rapid dives (leapfrog movements)
            if r<0.5 && abs(Escaping_Energy)>=0.5 % Soft besiege % rabbit try to escape by many zigzag deceptive motions
                
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));
                
                [AA,nEvals]=fobj(X1,NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
                [BB,nEvals]=fobj(X(i,:),NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
                
                if AA<BB % improved move?
                    X(i,:)=X1;
                else % hawks perform levy-based short rapid dives around the rabbit
                    X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:))+rand(1,dim).*Levy(dim);
                    
                    [AA,nEvals]=fobj(X2,NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
                    [BB,nEvals]=fobj(X(i,:),NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
                    
                    
                    if AA<BB%, % improved move?
                        X(i,:)=X2;
                    end
                end
            end
            
            if r<0.5 && abs(Escaping_Energy)<0.5 % Hard besiege % rabbit try to escape by many zigzag deceptive motions
                % hawks try to decrease their average location with the rabbit
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X));
                
                [AA,nEvals]=fobj(X1,NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
                [BB,nEvals]=fobj(X(i,:),NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
                
                if AA<BB % improved move?
                    X(i,:)=X1;
                else % Perform levy-based short rapid dives around the rabbit
                    X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X))+rand(1,dim).*Levy(dim);
                    
                    [AA,nEvals]=fobj(X2,NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
                    [BB,nEvals]=fobj(X(i,:),NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals);
                    
                    if AA<BB%, % improved move?
                        X(i,:)=X2;
                    end
                end
            end
            %%
        end
    end
    t=t+1
    nEvals
    Rabbit_Energy
    CNVG(t)=Rabbit_Energy;
end

solutions=Rabbit_Location;
gen=t;

gbest=solutions;

disp('Ensembled DEEPSO')
%DEEPSO parameters
if Select_testbed==1
    deParameters.I_NP= 5; 
    deParameters.NuRun=5;
    deParameters.Scenarios=10;
else
    deParameters.I_NP= 50; 
    deParameters.NuRun=1;
    deParameters.Scenarios=1;
end

NoCor=deParameters.NuRun;
No_solutions=deParameters.I_NP; %Notice that some algorithms are limited to one individual
 
low_habitat_limit=lowerB;
up_habitat_limit=upperB;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set other parameters
otherParameters =setOtherParameters(caseStudyData,No_solutions,Select_testbed);

% INITIALIZE strategic parameters of DEEPSO
    global deepso_par;
    global ff_par;
    [deepso_par,ff_par,set_par] = DEEPSO_SETTINGS1(low_habitat_limit,up_habitat_limit,deParameters,gbest);
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

            xbest=solutions;

            pos(1,:)=xbest;

            iter=0;
                        
            solFit=zeros(deParameters.I_NP,1);
            for ui=1:NoCor
            
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters);
            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);
            solFit=solFit+solFitness_M;
            
            end
            solFit=solFit/NoCor;
            
            gen=gen+1;
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
fitMaxVector(1,gen)=gbestval; %We save the mean value and mean penalty value
% The user can decide to save the mean, best, or any other value here

%%

while nEvals+(2*(deParameters.I_NP*deParameters.Scenarios*NoCor))-1<15300%set_par.nEvals_Max
    
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
    
    if rand() > deepso_par.localSearchProbability
        for i = 1 : set_par.pop_size
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DEEPSO movement rule
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW VELOCITY for the particles of the CURRENT population
            vel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability, set_par.D);
            %vel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest,vel( i, : ), Vmin, Vmax, weights( i, : ));
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
            %copyVel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest,copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ));
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
                 pos = learningEDA(pos,deParameters.I_NP);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the CURRENT population
        [ pos, vel ] = ENFORCE_POS_LIMITS( pos, set_par.Xmin, set_par.Xmax, vel, Vmin, Vmax , set_par.pop_size, set_par.D  );
        %pos=update1(pos,minPositionsMatrix,maxPositionsMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW POSITION for the particles of the COPIED population using LOCAL SEARCH
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
gen=gen+1;
iter=iter+1;
fitMaxVector(1,gen)=gbestval; %We save the mean value and mean penalty value
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

disp('Ensembled DEEPSO second stage')
%DEEPSO parameters
if Select_testbed==1
    deParameters.I_NP= 3; 
    deParameters.NuRun=10;
    deParameters.Scenarios=10;
else
    deParameters.I_NP= 50; 
    deParameters.NuRun=1;
    deParameters.Scenarios=1;
end

NoCor=deParameters.NuRun;
No_solutions=deParameters.I_NP; %Notice that some algorithms are limited to one individual
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set other parameters
otherParameters =setOtherParameters(caseStudyData,No_solutions,Select_testbed);

% INITIALIZE strategic parameters of DEEPSO
%     global deepso_par;
%     global ff_par;
    [deepso_par,ff_par,set_par] = DEEPSO_SETTINGS2(low_habitat_limit,up_habitat_limit,deParameters,gbest);
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

            %xbest=FVr_bestmemit;

            pos(1,:)=gbest;

            iter=0;
                        
            solFit=zeros(deParameters.I_NP,1);
            for ui=1:NoCor
            
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters);
            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);
            solFit=solFit+solFitness_M;
            
            end
            solFit=solFit/NoCor;
            
            gen=gen+1;
            iter=iter+1;

            fit=[];
            copyFit=[];
            tmpMemGBestSize=[];
            tmpMemGBestFit=[];
            tmpMemGBest=[];
            
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
gbestval
nEvals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number particles saved in the memory of the DEEPSO
memGBestMaxSize = deepso_par.memGBestMaxSize;
memGBestSize = 1;
% Memory of the DEEPSO
memGBest( memGBestSize, : ) = gbest;
memGBestFit( 1, memGBestSize ) = gbestval;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% The user should decide which is the criterion to optimize. 
fitMaxVector(1,gen)=gbestval; %We save the mean value and mean penalty value
% The user can decide to save the mean, best, or any other value here

%%

while nEvals+(2*(deParameters.I_NP*deParameters.Scenarios*NoCor))-1<22300%set_par.nEvals_Max
    
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
    
    if rand() > deepso_par.localSearchProbability
        for i = 1 : set_par.pop_size
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DEEPSO movement rule
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW VELOCITY for the particles of the CURRENT population
            vel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability, set_par.D);
            %vel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest,vel( i, : ), Vmin, Vmax, weights( i, : ));
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
            %copyVel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest,copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ));
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
             pos = learningEDA(pos,deParameters.I_NP);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the CURRENT population
        [ pos, vel ] = ENFORCE_POS_LIMITS( pos, set_par.Xmin, set_par.Xmax, vel, Vmin, Vmax , set_par.pop_size, set_par.D  );
        %pos=update1(pos,minPositionsMatrix,maxPositionsMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW POSITION for the particles of the COPIED population using LOCAL SEARCH
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
gen=gen+1;
iter=iter+1;
fitMaxVector(1,gen)=gbestval; %We save the mean value and mean penalty value
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

disp('Ensembled DEEPSO third stage')
%DEEPSO parameters
if Select_testbed==1
    deParameters.I_NP= 3; 
    deParameters.NuRun=10;
    deParameters.Scenarios=10;
else
    deParameters.I_NP= 50; 
    deParameters.NuRun=1;
    deParameters.Scenarios=1;
end

NoCor=deParameters.NuRun;
No_solutions=deParameters.I_NP; %Notice that some algorithms are limited to one individual
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set other parameters
otherParameters =setOtherParameters(caseStudyData,No_solutions,Select_testbed);

% INITIALIZE strategic parameters of DEEPSO
%     global deepso_par;
%     global ff_par;
    [deepso_par,ff_par,set_par] = DEEPSO_SETTINGS3(low_habitat_limit,up_habitat_limit,deParameters,gbest);
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

            %xbest=FVr_bestmemit;

            pos(1,:)=gbest;

            iter=0;
                        
            solFit=zeros(deParameters.I_NP,1);
            for ui=1:NoCor
            
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,pos,caseStudyData, otherParameters);
            nEvals  = nEvals+(set_par.pop_size*deParameters.Scenarios);
            solFit=solFit+solFitness_M;
            
            end
            solFit=solFit/NoCor;
            
            gen=gen+1;
            iter=iter+1;

            fit=[];
            copyFit=[];
            tmpMemGBestSize=[];
            tmpMemGBestFit=[];
            tmpMemGBest=[];
            
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
gbestval
nEvals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number particles saved in the memory of the DEEPSO
memGBestMaxSize = deepso_par.memGBestMaxSize;
memGBestSize = 1;
% Memory of the DEEPSO
memGBest( memGBestSize, : ) = gbest;
memGBestFit( 1, memGBestSize ) = gbestval;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% The user should decide which is the criterion to optimize. 
fitMaxVector(1,gen)=gbestval; %We save the mean value and mean penalty value
% The user can decide to save the mean, best, or any other value here

%%
nEvals+(2*(deParameters.I_NP*deParameters.Scenarios*NoCor))
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
    
    if rand() > deepso_par.localSearchProbability
        for i = 1 : set_par.pop_size
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % DEEPSO movement rule
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % COMPUTE NEW VELOCITY for the particles of the CURRENT population
            vel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability, set_par.D);
            %vel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest,vel( i, : ), Vmin, Vmax, weights( i, : ));
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
            %copyVel( i, : ) = EVDEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest,copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ));
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
            pos = learningEDA(pos,deParameters.I_NP);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENFORCE search space limits of the CURRENT population
        [ pos, vel ] = ENFORCE_POS_LIMITS( pos, set_par.Xmin, set_par.Xmax, vel, Vmin, Vmax , set_par.pop_size, set_par.D  );
        %pos=update1(pos,minPositionsMatrix,maxPositionsMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW POSITION for the particles of the COPIED population using LOCAL SEARCH
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
gen=gen+1;
iter=iter+1;
fitMaxVector(1,gen)=gbestval; %We save the mean value and mean penalty value
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

fitMaxVectorfinal=fitMaxVector(end-60:end);
fitMaxVector=[];
fitMaxVector=fitMaxVectorfinal;

Fit_and_p=[fitMaxVector(end) 0]; %;p2;p3;p4]
FVr_bestmemit=gbest;

end

% ___________________________________
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end

function pop_eda = learningEDA(pop_u,I_NP)
 %Cauchy's distribution
 mu = mean(pop_u);
 sd = std(pop_u);
for i=1:I_NP
 pop_eda(i,:) = -normrnd(mu,sd).*(mu - sd*tan(pi*(rand(1,1))-0.5));
end
end
