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

function [ mutated_pos ] = LOCAL_SEARCH_DISCRETE( pos, Xmin, Xmax )
tmpPos = rounding( pos );
minPos = rounding( Xmin );
maxPos = rounding( Xmax );
if rand() > 0.5
    if tmpPos < maxPos
        mutated_pos = tmpPos + 1;
    else
        mutated_pos = tmpPos - 1;
    end
else
    if tmpPos > minPos
        mutated_pos = tmpPos - 1;
    else
        mutated_pos = tmpPos + 1;
    end
end
end