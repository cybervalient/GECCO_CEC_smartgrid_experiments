function [Fit_and_p , sol , fitVector , Best_otherInfo  ] = upso( deParameters,caseStudyData,otherParameters,lowerB,upperB)
%% basic version of upso
%Authors; Christoforos N. Bekos, Paschalis A. Gkaidatzis
%% define static context
nD = deParameters.numOfDim ;
nP = deParameters.numOfParticles ;
matrix_r = deParameters.matrix_r;
max_iterations = deParameters.max_iterations;
mode = deParameters.mode;
topology = deParameters.topology;
u = 0:0.1:1;
uf = u(1+mod((1:nP)-1,length(u)));
urG = ones(nP,nD);
urL = ones(nP,nD);
%% set objective function
% objective = @(x) sum(exp(x) + sind(x)) ;
%% define particles 
X = rand(nP , nD).*repmat((upperB-lowerB),nP,1)+repmat(lowerB,nP,1);
V = rand(nP , nD).*repmat((upperB-lowerB),nP,1)+repmat(lowerB,nP,1);
c1 = 2.05;
c2 = 2.05;
x = (2/abs(2-(c1+c2)-sqrt((c1+c2)^2-4*(c1+c2))));
pBest = zeros(nP,1);
currentScore = zeros(nP,1);
lBest = zeros(nP,1);
lbpos = zeros(nP,1);
solbp = rand(nP , nD).*repmat((upperB-lowerB),nP,1)+repmat(lowerB,nP,1);
solbl = zeros(size(X));
[solFitness_M,~, ~] = feval( otherParameters.fnc,X,caseStudyData,otherParameters,deParameters.Neval);
for j=1:nP
	pBest(j) = getObjectiveValue( solFitness_M(j,:),deParameters.Neval,2) ;
end
[gBest, gBestPos] = min(pBest);
Old_gBest = gBest;
gBestSol = X(gBestPos,:);
indexes = 1:nP;
indexes = [indexes , indexes];
% gBestHistory = zeros(max_iterations,3408);
% gBestHistory(1,:) = X(gBestPos,:);
bestFitValues = solFitness_M(gBestPos,:);
fitVector = nan(2,max_iterations);  
for i=1:max_iterations    
    r = matrix_r(i);
    %% bound checking
    rep = repmat(lowerB,nP,1);
    X(X < rep) = rep(X < rep);
    rep = repmat(upperB,nP,1);
    X(X > rep) = rep(X > rep);    
	if(strcmp(mode,'lol_mode'))
        Neval = floor(50000 / (max_iterations*nP)) - 1;
        if(Neval > 100)
            Neval = 100;
        end
    else
        Neval = deParameters.Neval;
	end    
    if(i~= max_iterations)
        [solFitness_M,solPenalties_M, Struct_Eval] = feval( otherParameters.fnc,X,caseStudyData,otherParameters,Neval);
        for j=1:nP
            pen = getObjectiveValue( solFitness_M(j,:),deParameters.Neval,2) ;
            currentScore(j) = pen;
            if(pen < pBest(j))
                pBest(j) = pen;
                solbp(j,:) = X(j,:);
            end
        end
    else
        [solFitness_M,solPenalties_M, Struct_Eval] = feval( otherParameters.fnc,X,caseStudyData,otherParameters,100);
        for j=1:nP
            pen = getObjectiveValue( solFitness_M(j,:),100,1) ;
            if(pen < pBest(j))
                pBest(j) = pen;
                solbp(j,:) = X(j,:);
            end
        end
    end
    [curBest, curBestPos] = min(pBest);
	fitVector(i,:) = curBest;
    if(curBest < gBest)
        gBest = curBest;
        gBestPos = curBestPos;
        gBestSol = X(gBestPos,:);
        bestFitValues = solFitness_M(gBestPos,:);
    end
    if(topology == 2)
        % ring topology 
        csts = currentScore;
        mcs = sum(csts)/nP;
        ftn = exp(-csts./mcs);
        for xi=1:r:floor(nP/r)
            p = ftn(xi:xi+r)./sum(ftn(xi:xi+r));
            local_pos = zeros(100,1);
            c = 1;
            for j=1:length(p)
                for ii=1:floor(p(j)*100)
                    local_pos(c) = j;
                    c = c + 1;
                end
            end
            random = floor((sum(floor(p.*100))-1)*rand(1,1)) + 1;
            random = local_pos(random);
            localBestPos = random + xi - 1;
            solbl(xi:xi+r,:) = repmat(X(localBestPos,:),r+1,1);
        end        
    else
        for xi=1:nP
            tmp = zeros(nP,1);
            for xj=1:nP
                if(xj ~= xi)
                    % calculate distance between particle xi and all the others
                    tmp(i) = sum( (X(xi,:)-X(xj,:)).^2 );
                end
            end
            tmp(xi) = max(tmp);
            % find particle which is closer to xi
            tmp = tmp(1:nP);
            [~,closer_pos] = min(tmp);
            if( currentScore(lbpos(xi)) > currentScore(closer_pos))
                lbpos(xi) = closer_pos;
                solbl(xi,:) = X(closer_pos,:);
            end
        end
    end
	GVel = x*(V+c1.*rand(nP,nD).*(solbp-X)+c2.*rand(nP,nD).*(repmat(gBestSol,nP,1)-X));
	LVel = x*(V+c1.*rand(nP,nD).*(solbp-X)+c2.*rand(nP,nD).*(solbl-X));        
    if(deParameters.version == 1)
        urG(uf<=0.5,:) = rand(sum(uf<=0.5),nD);
        urL(uf>0.5,:)  = rand(sum(uf>0.5),nD);
        V = repmat(uf',1,nD).*urG.*GVel+(1-repmat(uf',1,nD)).*urL.*LVel;
    elseif(deParameters.version > 1 )
        if(deParameters.version == 2)
            %% linear
            u=i/max_iterations;   
        elseif(deParameters.version == 3)
            %% square
             u=(i/max_iterations)^2;
        elseif(deParameters.version == 4)
            %Square Root
            u=sqrt(i/max_iterations);
        elseif(deParameters.version == 5)
            mq=max_iterations/10;
            u=mod(i,mq)/mq;
        elseif(deParameters.version == 6)    
            u=exp(i*log(2)/max_iterations)-1;
        elseif(deParameters.version == 7)        
             u=1/(1+exp(-0.01*(i-max_iterations/2)));
        elseif(deParameters.version == 8)             
             mq=max_iterations/10;
             u=abs(sin(2*pi*i/mq));
        elseif(deParameters.version == 9)
            alfa =  1;
            dBest = min(currentScore) - Old_gBest;
            Old_gBest = min(currentScore);
            u= alfa*(i/max_iterations)^2 + (1-alfa)*(1 - 1/(exp(-dBest)*exp(abs(dBest)) ) );
        end
        if(u<=0.5)
        	V = u.*rand(nP,nD).*GVel+(1-u).*LVel;
        else
        	V = u.*GVel+(1-u).*rand(nP,nD).*LVel;
        end
    end
    Vbin = V(:,2:6:end);
    Vbin = 1./(1+exp(-Vbin));
    r3 = rand(size(Vbin));
    Xbin = X(:,2:6:end);
    Xbin(r3 < Vbin) = 1;    
	X = X + V;
    X(:,2:6:end) = round(Xbin);
end
%% store results
% Fit_and_p : array of size 1x2 with the best fitness and penalties values
Fit_and_p = [gBest, 0];
% sol : vector of size: 1 x noVariables with the best candidate solution found by your algorithm
sol = gBestSol;
% fitVector : array of size: 2xnoIterations with the value of the fitness and penalties over the iterations
%% store results - gia ta teleutaia 2
best_particle = gBestPos;
[~,best_scenario] = min(bestFitValues);
Best_otherInfo.idBestParticle = best_particle;
Best_otherInfo.genCostsFinal = Struct_Eval(best_scenario).otherParameters.genCosts(best_particle,:);
Best_otherInfo.loadDRcostsFinal = Struct_Eval(best_scenario).otherParameters.loadDRcosts(best_particle,:);
Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(best_scenario).otherParameters.v2gChargeCosts(best_particle,:);
Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(best_scenario).otherParameters.v2gDischargeCosts(best_particle,:);
Best_otherInfo.storageChargeCostsFinal = Struct_Eval(best_scenario).otherParameters.storageChargeCosts(best_particle,:);
Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(best_scenario).otherParameters.storageDischargeCosts(best_particle,:);
Best_otherInfo.stBalanceFinal = Struct_Eval(best_scenario).otherParameters.stBalance(best_particle,:,:);
Best_otherInfo.v2gBalanceFinal = Struct_Eval(best_scenario).otherParameters.v2gBalance(best_particle,:,:);
Best_otherInfo.penSlackBusFinal = Struct_Eval(best_scenario).otherParameters.penSlackBus(best_particle,:);


end

