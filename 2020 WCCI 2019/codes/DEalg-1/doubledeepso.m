function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo
] = ...
    doubledeepso(deParameters,Select_testbed,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)
%Guarantee same initial population
rand('state',otherParameters.iRuns) 

I_D= numel(up_habitat_limit); %Number of variables or dimension
lb =low_habitat_limit;
ub = up_habitat_limit;
fnc=  otherParameters.fnc;
grouping = deParameters.grouping;
deepso_NP = deParameters.deepso_NP;
deepso2_iter=deParameters.deepso2_iter;
deepso_iter=deParameters.deepso_iter;
communicationProbability = deParameters.communicationProbability;
mutationRate = deParameters.mutationRate;
I_itermax = deParameters.I_itermax;
fitMaxVector = nan(1,I_itermax);
memGBestMaxSize = deParameters.memGBestMaxSize;
temp_control = deParameters.temp;
[seps, nonseps, FECount,bestx] = half_RDG3(caseStudyData,otherParameters, lb,ub,grouping);
[linked_seps,linked_nonseps] = linking(seps, nonseps,lb,ub);
gbestval2 = 1000;
gbestval = 1000;
minPositionsMatrix=repmat(low_habitat_limit,deepso_NP,1);
maxPositionsMatrix=repmat(up_habitat_limit,deepso_NP,1);
deepso_idx = linked_seps;
deepso_idx2 = linked_nonseps;

%Guarantee same initial population
real_pos=genpop(deepso_NP,I_D,minPositionsMatrix,maxPositionsMatrix);%NxD
real_pos(1,:) = bestx;




globalmin_params=setOtherParameters(caseStudyData,1,Select_testbed);
deepso_params =setOtherParameters(caseStudyData,2*deepso_NP,Select_testbed);
deepso_params2 =setOtherParameters(caseStudyData,deepso_NP,Select_testbed);

if Select_testbed == 1
   globalmin_params.No_eval_Scenarios=500;
   deepso_params.No_eval_Scenarios=10;
   deepso_params2.No_eval_Scenarios=10;
else
    globalmin_params.No_eval_Scenarios=1;
    deepso_params.No_eval_Scenarios=1;
    deepso_params2.No_eval_Scenarios=1;
end






initialflag1 = 0;
initialflag = 0;





gen = 0;

while FECount<50000
    
        %%&&  fitIterationGap >= threshold
        if initialflag == 0
            deepso_D = numel(deepso_idx);
            Xmin = lb(deepso_idx);
            Xmax = ub(deepso_idx);
            Vmin= -ub(deepso_idx) + lb(deepso_idx);
            Vmax = -Vmin;
            vel = zeros(deepso_NP, deepso_D );
            pos = zeros( deepso_NP, deepso_D );
            weights = rand( deepso_NP, 4 );

            for i = 1 : deepso_NP
                vel( i, : ) = Vmin + ( Vmax - Vmin ) .* rand( 1, deepso_D);
                pos(i,:) = real_pos(i,(deepso_idx));
            end
            
            FECount = FECount + size(real_pos,1)*deepso_params2.No_eval_Scenarios;
            if FECount>50000
                break
            end
            
            fit = evalutate(fnc,real_pos,caseStudyData,deepso_params2,Select_testbed);
            %[fit,penalty,c,d,e,~]=feval(fnc,real_pos,caseStudyData,deepso_params2);
            %fit = fit'+std(e');
            



            [ ~, gbestid ] = min( fit );
            gbest = pos( gbestid, : );
            FECount = FECount + globalmin_params.No_eval_Scenarios;
            if FECount>50000
                break
            end

            %[globalfit,penalty,c,d,e,~]=feval(fnc,real_pos(gbestid,:),caseStudyData,globalmin_params);
            %globalfit = globalfit'+std(e');
            %gbestval = globalfit;
            gbestval = evalutate(fnc,real_pos(gbestid,:),caseStudyData,globalmin_params,Select_testbed);
            x1 = real_pos(gbestid,:);
            memGBestSize = 1;
            % Memory of the DEEPSO
            memGBest( memGBestSize, : ) = gbest;
            memGBestFit( 1, memGBestSize ) = gbestval;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            initialflag =1;
        end
    
    smallgen = 0;
    while (smallgen<deepso_iter)
        copyPos = pos;
        copyVel = vel;
        copyWeights = weights;
        tmpMemGBestSize = memGBestSize;
        tmpMemGBestFit = memGBestFit;
        tmpMemGBest = memGBest;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i = 1 : deepso_NP

                vel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, vel( i, : ), Vmin, Vmax, weights( i, : ), communicationProbability, deepso_D,temp_control);

                [ pos( i, : ), vel( i, : ) ] = COMPUTE_NEW_POS( pos( i, : ), vel( i, : ) );

                copyWeights( i, : ) = MUTATE_WEIGHTS( weights( i, : ), mutationRate );

                copyVel( i, : ) = DEEPSO_COMPUTE_NEW_VEL( copyPos( i, : ), gbest, fit( i ), tmpMemGBestSize, tmpMemGBestFit, tmpMemGBest, copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ), communicationProbability, deepso_D,temp_control );

                [ copyPos( i, : ), copyVel( i, : ) ] = COMPUTE_NEW_POS( copyPos( i, : ), copyVel( i, : ) );

            end

            [ copyPos, copyVel ] = ENFORCE_POS_LIMITS( copyPos, Xmin, Xmax, copyVel, Vmin, Vmax, deepso_NP, deepso_D );

            [ pos, vel ] = ENFORCE_POS_LIMITS( pos, Xmin, Xmax, vel, Vmin, Vmax,deepso_NP, deepso_D );

            
            real_copypos = real_pos;
            for i = 1 : deepso_NP
                real_pos(i,(deepso_idx)) = pos(i,:);
                real_copypos(i,(deepso_idx)) = copyPos(i,:);  
            end
            
            eval_pos =cat(1,real_pos,real_copypos); 
            
            FECount = FECount + size(eval_pos,1)*deepso_params.No_eval_Scenarios;
            if FECount>50000
                break
            end
            tempfit = evalutate(fnc,eval_pos,caseStudyData,deepso_params,Select_testbed);
            
            
            %[tempfit,penalty,c,d,e,~]=feval(fnc,eval_pos,caseStudyData,deepso_params);
            %tempfit =tempfit'+std(e');
            
            fit = tempfit(1:deepso_NP);
            copyFit = tempfit(deepso_NP+1:end);
            selParNewSwarm = ( copyFit < fit );
            for i = 1 : deepso_NP
                if selParNewSwarm( i )
                    fit( i ) = copyFit( i );
                    pos( i, : ) = copyPos( i, : );
                    vel( i, : ) = copyVel( i, : );
                    weights( i, : ) = copyWeights( i, : );

                end
            end

            [ ~, gbestid ] = min( fit );

            
            FECount = FECount + globalmin_params.No_eval_Scenarios;
            if FECount>50000
                break
            end                                                                                                                                                                  
            real_pos(gbestid,(deepso_idx)) = pos(gbestid,:);
            %[globalfit,penalty,c,d,e, ~]=feval(fnc,real_pos(gbestid,:),caseStudyData,globalmin_params);
            %globalfit = globalfit'+std(e');
            %tmpgbestval = globalfit;
            
            tmpgbestval = evalutate(fnc,real_pos(gbestid,:),caseStudyData,globalmin_params,Select_testbed);
            
            if tmpgbestval < gbestval
                gbestval = tmpgbestval;
                gbest = pos( gbestid, : );
                x1 = real_pos(gbestid,:);
                
                
                % UPDATE MEMORY DEEPSO
                if memGBestSize < memGBestMaxSize
                    memGBestSize = memGBestSize + 1;
                    memGBest( memGBestSize, : ) = gbest;
                    memGBestFit( 1, memGBestSize ) = gbestval;
                else
                    [ ~, tmpgworstid ] = max( memGBestFit );
                    memGBest( tmpgworstid, : ) = gbest;
                    memGBestFit( 1, tmpgworstid ) = gbestval;
                end
            end

            bestx(deepso_idx) =  gbest;
            smallgen = smallgen+1;


    end
        
        
    real_pos = repmat(bestx,deepso_NP,1);

    if initialflag1 == 0
        
        deepso_D2 = numel(deepso_idx2);
        Xmin2 = lb(deepso_idx2);
        Xmax2 = ub(deepso_idx2);
        Vmin2= -ub(deepso_idx2) + lb(deepso_idx2);
        Vmax2 = -Vmin2;
        vel2 = zeros(deepso_NP, deepso_D2 );
        pos2 = zeros( deepso_NP, deepso_D2 );
        weights2 = rand( deepso_NP, 4 );
        
        for i = 1 : deepso_NP
            vel2( i, : ) = Vmin2 + ( Vmax2 - Vmin2 ) .* rand( 1, deepso_D2);
            pos2(i,:) =real_pos(i,(deepso_idx2));
        
        end
        
        FECount = FECount + size(real_pos,1)*deepso_params2.No_eval_Scenarios;
        if FECount>50000
             break
        end

        %[fit2,penalty,c,d,e, ~]=feval(fnc,real_pos,caseStudyData,deepso_params2);
        
        %fit2 = fit2'+std(e');
        
        fit2 = evalutate(fnc,real_pos,caseStudyData,deepso_params2,Select_testbed);

        
        [ gbestval2, gbestid ] = min( fit2 );
        gbest2 = pos2( gbestid, : );
        
        FECount = FECount + globalmin_params.No_eval_Scenarios;
        if FECount>50000
            break
        end

        %[globalfit2,penalty,c,d,e, ~]=feval(fnc,real_pos(gbestid,:),caseStudyData,globalmin_params);
        %globalfit2 = globalfit2'+std(e');
        %gbestval2 = globalfit2;
        
        gbestval2 = evalutate(fnc,real_pos(gbestid,:),caseStudyData,globalmin_params,Select_testbed);
        x2 = real_pos(gbestid,:);
        memGBestSize2 = 1;
        % Memory of the DEEPSO
        memGBest2( memGBestSize2, : ) = gbest2;
        memGBestFit2( 1, memGBestSize2 ) = gbestval2;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        initialflag1 =1;
    end
    
    smallgen = 0;
    while (smallgen<deepso2_iter)
        copyPos2 = pos2;
        copyVel2 = vel2;
        copyWeights2 = weights2;
        tmpMemGBestSize2 = memGBestSize2;
        tmpMemGBestFit2 = memGBestFit2;
        tmpMemGBest2 = memGBest2;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i = 1 : deepso_NP
                %size(pos2( i, : ))
                %size(gbest2)
                %size(fit2( i ))
                vel2( i, : ) = DEEPSO_COMPUTE_NEW_VEL( pos2( i, : ), gbest2, fit2( i ), tmpMemGBestSize2, tmpMemGBestFit2, tmpMemGBest2, vel2( i, : ), Vmin2, Vmax2, weights2( i, : ), communicationProbability, deepso_D2,temp_control);

                [ pos2( i, : ), vel2( i, : ) ] = COMPUTE_NEW_POS( pos2( i, : ), vel2( i, : ) );

                copyWeights2( i, : ) = MUTATE_WEIGHTS( weights2( i, : ), mutationRate );

                copyVel2( i, : ) = DEEPSO_COMPUTE_NEW_VEL( copyPos2( i, : ), gbest2, fit2( i ), tmpMemGBestSize2, tmpMemGBestFit2, tmpMemGBest2, copyVel2( i, : ), Vmin2, Vmax2, copyWeights2( i, : ), communicationProbability, deepso_D2,temp_control );

                [ copyPos2( i, : ), copyVel2( i, : ) ] = COMPUTE_NEW_POS( copyPos2( i, : ), copyVel2( i, : ) );

            end

            [ copyPos2, copyVel2 ] = ENFORCE_POS_LIMITS( copyPos2, Xmin2, Xmax2, copyVel2, Vmin2, Vmax2, deepso_NP, deepso_D2 );

            [ pos2, vel2 ] = ENFORCE_POS_LIMITS( pos2, Xmin2, Xmax2, vel2, Vmin2, Vmax2,deepso_NP, deepso_D2 );

            
            real_copypos = real_pos;
            for i = 1 : deepso_NP
                real_pos(i,(deepso_idx2)) = pos2(i,:);
                real_copypos(i,(deepso_idx2)) = copyPos2(i,:);  
            end
            
            eval_pos =cat(1,real_pos,real_copypos); 
            
            FECount = FECount + size(eval_pos,1)*deepso_params.No_eval_Scenarios;
            if FECount>50000
                
                break
            end
            
            %[tempfit,penalty,c,d,e, ~]=feval(fnc,eval_pos,caseStudyData,deepso_params);
            %tempfit =tempfit'+std(e');
            tempfit = evalutate(fnc,eval_pos,caseStudyData,deepso_params,Select_testbed);
            
            
            fit2 = tempfit(1:deepso_NP);
            copyFit2 = tempfit(deepso_NP+1:end);
            selParNewSwarm = ( copyFit2 < fit2 );
            for i = 1 : deepso_NP
                if selParNewSwarm( i )
                    fit2( i ) = copyFit2( i );
                    pos2( i, : ) = copyPos2( i, : );
                    vel2( i, : ) = copyVel2( i, : );
                    weights2( i, : ) = copyWeights2( i, : );

                end
            end

            [ ~, gbestid ] = min( fit2 );

            
            FECount = FECount + globalmin_params.No_eval_Scenarios;
            if FECount>50000
                break
            end                                                                                                                                                                  
            real_pos(gbestid,(deepso_idx2)) = pos2(gbestid,:);
            %[globalfit2,penalty,c,d,e, ~]=feval(fnc,real_pos(gbestid,:),caseStudyData,globalmin_params);
            
            %globalfit2 = globalfit2'+std(e');
            %tmpgbestval = globalfit2;
            
            tmpgbestval = evalutate(fnc,real_pos(gbestid,:),caseStudyData,globalmin_params,Select_testbed);
            
            
            if tmpgbestval < gbestval2
                gbestval2 = tmpgbestval;
                gbest2 = pos2( gbestid, : );
                x2 = real_pos(gbestid,:);
                
                
                % UPDATE MEMORY DEEPSO
                if memGBestSize2 < memGBestMaxSize
                    memGBestSize2 = memGBestSize2 + 1;
                    memGBest2( memGBestSize2, : ) = gbest2;
                    memGBestFit2( 1, memGBestSize2 ) = gbestval2;
                else
                    [ ~, tmpgworstid ] = max( memGBestFit2 );
                    memGBest2( tmpgworstid, : ) = gbest2;
                    memGBestFit2( 1, tmpgworstid ) = gbestval2;
                end
            end

            bestx(deepso_idx2) =  gbest2;
            smallgen = smallgen+1;


    end
    
    real_pos = repmat(bestx,deepso_NP,1);
    gen = gen+1;
    %fprintf('Fitness value: %f\n',gbestval2)%gbestval )%best_val )
    %fprintf('Generation: %d\n',gen)
    fitMaxVector(1,gen) = gbestval2;        
    if FECount>50000
        break
    end
end


if gbestval <= gbestval2
    final_val = gbestval;
    final_x = x1;
else
    final_val = gbestval2;
    final_x = x2;
end

FVr_bestmemit = final_x;
Fit_and_p=[final_val 0]; %;p2;p3;p4]


function pop=genpop(a,b,lowMatrix,upMatrix)
pop=unifrnd(lowMatrix,upMatrix,a,b);

function val=evalutate(fnc,eval_pos,caseStudyData,deepso_params,Select_testbed)
    if Select_testbed == 1

        [val,~,~,~,e, ~]=feval(fnc,eval_pos,caseStudyData,deepso_params);
        val =val'+std(e');
    else
        [val,~]=feval(fnc,eval_pos,caseStudyData,deepso_params);
    end

            
function p=update(p,lowMatrix,upMatrix)
        [idx] = find(p<lowMatrix);
        p(idx)=lowMatrix(idx);
        [idx] = find(p>upMatrix);
        p(idx)=upMatrix(idx);




    