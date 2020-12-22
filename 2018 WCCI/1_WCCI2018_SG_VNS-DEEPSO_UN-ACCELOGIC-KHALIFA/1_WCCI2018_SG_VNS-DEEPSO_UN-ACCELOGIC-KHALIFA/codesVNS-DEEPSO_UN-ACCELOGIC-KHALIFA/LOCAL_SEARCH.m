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

function [ new_pos ] = LOCAL_SEARCH( pos, Xmin, Xmax , D)
% Mutates the integer part of the particle
% global proc
% global ps
global deepso_par
new_pos = pos;
% switch proc.system
%     case 41
%         % Select which type of variables will be mutated
%         prob = deepso_par.localSearchContinuousDiscrete;
%         if rand() > prob
%             prob = 1 / ( ps.n_gen_VS + 1 );
%             for i = 1 : ps.n_gen_VS
%                 tmpDim = i;
%                 if rand() < prob
%                     new_pos( tmpDim ) = LOCAL_SEARCH_CONTINUOUS( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
%                 end
%             end
%             if rand() < prob
%                 tmpDim = ps.n_gen_VS + ps.n_OLTC + 1;
%                 new_pos( tmpDim ) = LOCAL_SEARCH_CONTINUOUS( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
%             end
%         else
%             prob = 1 / ( ps.n_OLTC + 1 );
%             for i = 1 : ps.n_OLTC;
%                 tmpDim = ps.n_gen_VS + i;
%                 if rand() < prob
%                     new_pos( tmpDim ) = LOCAL_SEARCH_DISCRETE( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
%                 end
%             end
%             if rand() < prob
%                 tmpDim = ps.n_gen_VS + ps.n_OLTC + ps.n_SH;
%                 new_pos( tmpDim ) = LOCAL_SEARCH_DISCRETE( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
%             end
%         end
%     otherwise
        prob = deepso_par.localSearchContinuousDiscrete;
%         if rand() > prob
            prob = 1 / D;
            for i = 1 : D;
                tmpDim = i;
                if rand() < prob
                    new_pos( tmpDim ) = LOCAL_SEARCH_CONTINUOUS( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
                end
            end
%         else
%             prob = 1 / ps.D_disc;
%             for i = 1 : ps.D_disc;
%                 tmpDim = ps.D_cont + i;
%                 if rand() < prob
%                     new_pos( tmpDim ) = LOCAL_SEARCH_DISCRETE( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
%                 end
%             end
%         end
% end
end