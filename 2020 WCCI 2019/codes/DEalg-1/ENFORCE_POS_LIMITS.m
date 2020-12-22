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

function [ new_pos, new_vel ] = ENFORCE_POS_LIMITS( pos, Xmin, Xmax, vel, Vmin, Vmax, pop_size, D )
% Enforces search space limits
%global proc
%global ps
new_pos = pos;
new_vel = vel;
for i = 1 : pop_size
    for j = 1 : D
        if new_pos( i, j ) < Xmin( j )
            new_pos( i, j ) = Xmin( j );
            if new_vel( i, j ) < 0
                new_vel( i, j ) = -new_vel( i, j );
            end
        elseif new_pos( i, j ) > Xmax( j )
            new_pos( i, j ) = Xmax( j );
            if new_vel( i, j ) > 0
                new_vel( i, j ) = -new_vel( j );
            end
        end
        % Check velocity in case of asymmetric velocity limits
        if new_vel( i, j ) < Vmin( j )
            new_vel( i, j ) = Vmin( j );
        elseif new_vel( i, j ) > Vmax( j )
            new_vel( i, j ) = Vmax( j );
        end
    end
end
end