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

function CALC_COEFS_FF()
% Calculates new coefficients for the fitness function
global ff_par;
maxCoef = max( ff_par.avgCoefFF( 1:ff_par.numCoefFF - ff_par.excludeBranchViolations ) );
for i = 1 : ff_par.numCoefFF - ff_par.excludeBranchViolations
    if ff_par.avgCoefFF( i ) > 0
        if ff_par.coefFF( i ) == maxCoef
            ff_par.coefFF( i ) = 1;
        else
            ff_par.coefFF( i ) = maxCoef / ff_par.avgCoefFF( i );
        end
    end
end
ff_par.coefFF = round( log10( ff_par.coefFF( 1:ff_par.numCoefFF - ff_par.excludeBranchViolations ) ) );
for i = 1 : ff_par.numCoefFF - ff_par.excludeBranchViolations
    if ff_par.coefFF( i ) == Inf || ff_par.coefFF( i ) == -Inf
        ff_par.coefFF( i ) = 1;
    end
    ff_par.coefFF( i ) = 10 ^ ff_par.coefFF( i );
end
if ff_par.excludeBranchViolations == 1
    ff_par.coefFF( ff_par.numCoefFF ) = max( ff_par.coefFF( 1:ff_par.numCoefFF - ff_par.excludeBranchViolations ) );
end
end