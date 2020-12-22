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