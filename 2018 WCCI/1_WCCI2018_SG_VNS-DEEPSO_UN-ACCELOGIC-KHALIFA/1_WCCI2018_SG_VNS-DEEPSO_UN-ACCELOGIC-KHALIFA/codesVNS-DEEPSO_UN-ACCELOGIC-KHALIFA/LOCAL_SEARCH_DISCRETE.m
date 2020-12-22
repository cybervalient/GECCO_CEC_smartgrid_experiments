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