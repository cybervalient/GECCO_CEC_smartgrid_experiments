function [ new_pos ] = LOCAL_SEARCH( pos, Xmin, Xmax )
% Mutates the integer part of the particle
global proc
global ps
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
        %prob = deepso_par.localSearchContinuousDiscrete;
         localSearchContinuousDiscrete = 0.75;
         prob = localSearchContinuousDiscrete;
        if rand() > prob
            %prob = 1 / ps.D_cont;
            prob = 1 / 168;% first active power continuous variables, prob=0.0000208
           % for i = 1 : 48072;
           for j=0:23
             for i=1+(j*142):7+(j*142);% 1-7, 143-149, 285-291  continuous var
                tmpDim = i;
                if rand() > prob
                    new_pos( tmpDim ) = LOCAL_SEARCH_CONTINUOUS( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
                end
             end
           end
           
           
           for j=0:23
             for i=15+(j*142):142+(j*142);% remaining continuous variables.15-142, 157-284, 299-426
               
             tmpDim = i;
                if rand() > prob
                    new_pos( tmpDim ) = LOCAL_SEARCH_CONTINUOUS( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
                end
             end
           end
        else
            prob = 1 / 168; % no of discrete variables,0.005952
            %for i = 1 : ps.D_disc;
              for j=0:23
                  for i=8+(j*142):14+(j*142) % discrete 8-14, 150: 156, 292:298 generator binaries connected
                 % for i=155+(j*2080):231+(j*2080) % discrete variables
                   tmpDim=i;
                if rand() > prob
                    new_pos( tmpDim ) = LOCAL_SEARCH_DISCRETE( new_pos( tmpDim ), Xmin( tmpDim ), Xmax( tmpDim ) );
                end
                  end
              end
end
end

% for j=0:23
  % for i=155+(j*2080):231+(j*2080)

      %  Xmin( 1, i ) = Xmin( 1, i ) - 0.4999;
      %  Xmax( 1, i ) = Xmax( 1, i ) + 0.4999;
  % end
%end