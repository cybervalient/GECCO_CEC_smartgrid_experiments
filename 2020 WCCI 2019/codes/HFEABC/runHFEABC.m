
function [Fit_and_p,GlobalParams,fitMaxVector] = ...
    runHFEABC(mabcParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit)

%/* Control Parameters of ABC algorithm*/
% NP = mabcParameters.NP; %/* The number of colony size (employed bees+onlooker bees)*/
FoodNumber = mabcParameters.FoodNumber; %/*The number of food sources equals the half of the colony size*/
limit = mabcParameters.limit; %/*A food source which could not be improved through "limit" trials is abandoned by its employed bee*/
maxCycle = mabcParameters.maxCycle; %/*The number of cycles for foraging {a stopping criteria}*/
D = numel(up_habitat_limit);%/*The number of parameters of the problem to be optimized*/
lb = low_habitat_limit;
ub = up_habitat_limit;
% C = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I_strategy = mabcParameters.I_strategy; %important variable
fnc = otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
fitMaxVector = nan(1,maxCycle);
ITER = nan(1,maxCycle);
% /*All food sources are initialized */
minPositionsMatrix=repmat(low_habitat_limit,FoodNumber,1);
maxPositionsMatrix=repmat(up_habitat_limit,FoodNumber,1);

% generate initial population.
rand('state',otherParameters.iRuns) %Guarantee same initial population
Foods = unifrnd(minPositionsMatrix,maxPositionsMatrix,FoodNumber,D);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------Evaluate the best member after initialization----------------------
[ObjVal, ~]=feval(fnc,Foods,caseStudyData,otherParameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iter=1;

%/*The best food source is memorized*/
% BestInd=find(ObjVal==min(ObjVal));
% BestInd=BestInd(end);
% GlobalMin=ObjVal(BestInd);
% GlobalParams=Foods(BestInd,:);
BestInd=find(ObjVal==min(ObjVal));
BestInd=BestInd(end);
GlobalMin=ObjVal(BestInd);
GlobalParams=Foods(BestInd,:);
fitMaxVector(1,iter) = GlobalMin;
ITER(iter) = iter;
% reset trial counters
trial=zeros(1,FoodNumber);

while ((iter <= maxCycle)),

%%%%%%%%% EMPLOYED BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:(FoodNumber)

%         %/*The parameter to be changed is determined randomly*/ ds
%         Param2Change=fix(rand*D)+1;
% 
%         %/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
%         neighbour=fix(rand*(FoodNumber))+1;
% 
%         %/*Randomly selected solution must be different from the solution i*/        
%         while(neighbour==i)
%             neighbour=fix(rand*(FoodNumber))+1;
%         end;

        
%         %/*The best food source so far:
%         ind=find(ObjVal==min(ObjVal));
%         ind=ind(end);
%         if (ObjVal(ind)<GlobalMin)
%             GlobalMin=ObjVal(ind);
%             GlobalParams=Foods(ind,:);
%         end
        
%         sol=Foods(i,:);
        %  /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
%         sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change))*(rand-0.5)*2;


        mu = mean(Foods);
        sd = std(Foods);
        sol = -normrnd(mu,sd).*(mu - sd*tan(pi*(rand(1,1))-0.5));
      
        %  /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
        ind=find(sol<lb);
        sol(ind)=lb(ind);
        ind=find(sol>ub);
        sol(ind)=ub(ind);

        %evaluate new solution
        [ObjValSol, ~]=feval(fnc,sol,caseStudyData, otherParameters.one);

       % Elitist Selection
       if (ObjValSol<ObjVal(i)) %/*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
           Foods(i,:)=sol;
           ObjVal(i)=ObjValSol;
           trial(i)=0;
       else
           trial(i)=trial(i)+1; %/*if the solution i can not be improved, increase its trial counter*/
       end;

    end;

%%%%%%%%%%%%%%%%%%%%%%%% CalculateProbabilities %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%/* A food source is chosen with the probability which is proportioal to its quality*/
%/*Different schemes can be used to calculate the probability values*/
%/*For example prob(i)=ObjVal(i)/sum(ObjVal)*/
%/*or in a way used in the method below prob(i)=a*ObjVal(i)/max(ObjVal)+b*/
%/*probability values are calculated by using ObjVal values and normalized by dividing maximum ObjVal value*/

    prob=ObjVal./sum(ObjVal);
  
%%%%%%%%%%%%%%%%%%%%%%%% ONLOOKER BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    i=1;
    t=0;
    while(t<FoodNumber)
        if(rand<prob(i))
            
            t=t+1;
            %/*The parameter to be changed is determined randomly*/
%             Param2Change=fix(rand*D)+1;

            %/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
%             neighbour=fix(rand*(FoodNumber))+1;
% 
%             %/*Randomly selected solution must be different from the solution i*/        
%             while(neighbour==i)
%                 neighbour=fix(rand*(FoodNumber))+1;
%             end;
           
            %/*The best food source so far:
            ind=find(ObjVal==min(ObjVal));
            ind=ind(end);
            if (ObjVal(ind)<GlobalMin)
                GlobalMin=ObjVal(ind);
                GlobalParams=Foods(ind,:);
            end
            
            
%             sol=Foods(i,:);
%             L = iter/maxCycle;
%             fg = (1 - L)^(1/D);
%             fb = log(L+1)/log(2);
%             %  /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
% 
%             sol= Foods(ind,:)+ fg * (Foods(i,:)-Foods(neighbour,:))*(rand-0.5)*2 ...
%             + rand*C*fb*(Foods(ind,:) - Foods(neighbour,:));
            mu = mean(Foods);
            sd = std(Foods);
            sol = -normrnd(mu,sd).*(mu - sd*tan(pi*(rand(1,1))-0.5));
            
            %  /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
            ind=find(sol<lb);
            sol(ind)=lb(ind);
            ind=find(sol>ub);
            sol(ind)=ub(ind);

            %evaluate new solution
            [ObjValSol, ~]=feval(fnc,sol,caseStudyData, otherParameters.one);

            % /*a greedy selection is applied between the current solution i and its mutant*/
            if (ObjValSol<ObjVal(i)) %/*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
                Foods(i,:)=sol;
                ObjVal(i)=ObjValSol;
                trial(i)=0;
            else
                trial(i)=trial(i)+1; %/*if the solution i can not be improved, increase its trial counter*/
            end;
        end;
    
        i=i+1;
        if (i==(FoodNumber)+1) 
            i=1;
        end;   
    end; 


    %/*The best food source is memorized*/
    ind=find(ObjVal==min(ObjVal));
    ind=ind(end);
    if (ObjVal(ind)<GlobalMin)
        GlobalMin=ObjVal(ind);
        GlobalParams=Foods(ind,:);
    end;
         
         
%%%%%%%%%%%% SCOUT BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%/*determine the food sources whose trial counter exceeds the "limit" value. 
%In Basic ABC, only one scout is allowed to occur in each cycle*/

%     ind=find(trial==max(trial));
%     ind=ind(end);
%     if (trial(ind)>limit)
%         trial(ind)=0;
%         sol = Foods(ind,:) + (iter / maxCycle) * (rand - 0.5) * 2 * Foods(ind,:);
%         %evaluate new solution
%         [ObjValSol, ~] = feval(fnc,sol,caseStudyData, otherParameters.one);
%         Foods(ind,:) = sol;
%         ObjVal(ind) = ObjValSol;
%     end;

    for i = 1:FoodNumber
        if (trial(i)>limit)
            trial(i)=0;
            sol = Foods(i,:) + (iter / maxCycle) * (rand - 0.5) * 2 * Foods(i,:);
%             sol=(ub-lb).*rand(1,D)+lb;
            indx=find(sol<lb);
            sol(indx)=lb(indx);
            indx=find(sol>ub);
            sol(indx)=ub(indx);

            %evaluate new solution
            [ObjValSol, ~] = feval(fnc,sol,caseStudyData, otherParameters.one);
            
            Foods(i,:)=sol;
            ObjVal(i)=ObjValSol;

        end;
    end;



% fprintf('Iter=%d ObjVal=%g\n',iter,GlobalMin);

iter=iter+1;
ITER(iter) = iter;
fitMaxVector(1,iter) = GlobalMin;

end % End of ABC
plot(ITER, fitMaxVector);
grid on;
hold on;
Fit_and_p=[fitMaxVector(1,iter) 0];

