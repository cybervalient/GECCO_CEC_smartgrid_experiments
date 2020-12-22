%% TEAM: UN-UF-GERS-JEA
% Cooperation: Universidad Nacional de Colombia, University of Florida, GERS USA and JEA
%% TEAM MEMBERS: 
% Pedro Garcia, pjgarciag@unal.edu.co, PhD Student at UN 
% Diego Rodriguez, diego.rodriguez@gers.com.co, International Studies Manager at GERS USA and PhD Student at UN
% David Alvarez, dlalvareza@unal.edu.co, Postdoc at UN
% Sergio Rivera, srriverar@unal.edu.co, Professor at UN and Fulbright Scholar
% Camilo Cortes, caacortesgu@unal.edu.co, Professor at UN
% Alejandra Guzman, maguzmanp@unal.edu.co, Professor at UN
% Arturo Bretas, arturo@ece.ufl.edu, Professor at UF
% Julio Romero, romeje@jea.com, Chief Innovation and Transformation Officer at JEA
%% ALGORITMH: VNS-DEEPSO
% Combination of Variable Neighborhood Search algorithm (VNS) and Differential Evolutionary Particle Swarm Optimization (DEEPSO)
%% 
% THIS SCRIPT IS BASED ON THE WINNER CODES IN THE TEST BED 2 ON THE
% IEEE 2017 and 2018 Competition & panel: Evaluating the Performance of Modern Heuristic
% Optimizers on Smart Grid Operation Problems

function [ new_vel ] = DEEPSO_COMPUTE_NEW_VEL( pos, gbest, fit, numGBestSaved, memGBestFit, memGBest, vel, Vmin, Vmax, weights, communicationProbability,D )
% Computes new velocity according to the DEEPSO movement rule
%global ps

%global D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute inertial term
inertiaTerm = weights( 1 ) * vel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select subset of particles to sample myBestPos from
% Get the index of the best particles ever visited that have a fitness less
% than or equal to the fitness of particle i
tmpMemoryVect = zeros( 1, numGBestSaved );
tmpMemoryVectSize = 0;
for i = 1 : numGBestSaved
    if memGBestFit( 1, i ) <= fit
        tmpMemoryVectSize = tmpMemoryVectSize + 1;
        tmpMemoryVect( 1, tmpMemoryVectSize ) = i;
    end
end
tmpMemoryVect = tmpMemoryVect( 1, 1:tmpMemoryVectSize );
% Sample every entry of myBestPos using the subset - Pb-rnd
myBestPos = zeros( 1, D );
tmpIndexMemoryVect = randsample( tmpMemoryVectSize, D, true );
for i = 1 : D
    myBestPos( 1, i ) = memGBest( tmpMemoryVect( 1, tmpIndexMemoryVect( i ) ), i );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute memory term

memoryTerm = weights( 2 ) * ( myBestPos - pos );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cooperation term
% Sample normally distributed number to perturbate the best position
cooperationTerm = weights( 3 ) * ( gbest * ( 1 + weights( 4 ) * normrnd( 0, 1 ) ) - pos );
communicationProbabilityMatrix = rand( 1, D ) < communicationProbability;
cooperationTerm = cooperationTerm .* communicationProbabilityMatrix;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute velocity
new_vel = inertiaTerm + memoryTerm + cooperationTerm;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check velocity limits
new_vel = ( new_vel > Vmax ) .* Vmax + ( new_vel <= Vmax ) .* new_vel;
new_vel = ( new_vel < Vmin ) .* Vmin + ( new_vel >= Vmin ) .* new_vel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end