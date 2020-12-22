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