function [ new_vel ] = DEEPSO_COMPUTE_NEW_VEL( pos, gbest,   vel, Vmin, Vmax, weights, communicationProbability )
% Computes new velocity according to the DEEPSO movement rule
global ps
global proc
global Levy_constant
global D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute inertial term
inertiaTerm = weights( 1 ) * vel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select subset of particles to sample myBestPos from
% Get the index of the best particles ever visited that have a fitness less
% than or equal to the fitness of particle i
% tmpMemoryVect = zeros( 1, numGBestSaved );
% tmpMemoryVectSize = 0;  % change from 0 to 1 on 21 march
% for i = 1 : numGBestSaved
%     if memGBestFit( 1, i ) <= fit1
%          tmpMemoryVectSize = tmpMemoryVectSize + 1;
%         tmpMemoryVect( 1, tmpMemoryVectSize ) = i;
%     end
% end
% tmpMemoryVect = tmpMemoryVect( 1, 1:tmpMemoryVectSize );
% % Sample every entry of myBestPos using the subset - Pb-rnd
% myBestPos = zeros( 1, D );
% tmpIndexMemoryVect = randsample( tmpMemoryVectSize, D, true );
% for i = 1 : D
%     myBestPos( 1, i ) = memGBest( tmpMemoryVect( 1, tmpIndexMemoryVect( i ) ), i );
% end
% pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute memory term
%memoryTerm = weights( 2 ) * ( myBestPos - pos );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cooperation term
% Sample normally distributed number to perturbate the best position
cooperationTerm = weights( 3 ) * ( gbest * ( 1 + weights( 4 ) * normrnd( 0, 1 ) ) - pos );
communicationProbabilityMatrix = rand( 1, D ) < communicationProbability;
cooperationTerm = cooperationTerm .* communicationProbabilityMatrix;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply levy flight for each particle to enhance global search.
%Levy exponent and coefficient
beta=3/2; %scalar
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);%scalar

%for j=1:proc.pop_size  
    %s=pos(1:D); 
    % This is a simple way of implementing Levy flights
    % For standard random walks, use step=1;
    %% Levy flights by Mantegna's algorithm
    u=randn(1,D)*sigma;
    v=randn(1,D);
    step=u./abs(v).^(1/beta); 
  % In the next equation, the difference factor (s-best) means that 
    % when the solution is the best solution, it remains unchanged.     
    %stepsize=0.31*step.*(pos-gbest);
    
    % stepsize=unifrnd(0.1, 0.31, 1,1)*step.*(pos-gbest);
    stepsize=Levy_constant*step.*(pos-gbest);
    levystep=stepsize;
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute velocity
new_vel = inertiaTerm + cooperationTerm + levystep; %+ memoryTerm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check velocity limits using chaos

ccrand=rand(1,D);
ccpos=((1./ccrand)-floor(1./ccrand));
new_vel = ( new_vel > Vmax ) .*(Vmin + ( Vmax - Vmin ).*ccpos)+ ( new_vel <= Vmax ) .* new_vel;
new_vel = ( new_vel < Vmin ) .* (Vmin + ( Vmax - Vmin ).*ccpos) + ( new_vel >= Vmin ) .* new_vel;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end