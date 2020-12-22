function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    pso_competition(PSOparameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit,nEvals,xbest)
%% Problem Definition

nVar=numel(up_habitat_limit);            % Number of Decision Variables
VarMin=low_habitat_limit;         % Lower Bound of Variables
VarMax=up_habitat_limit;         % Upper Bound of Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fnc= otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PSO Parameters

MaxIt=                      PSOparameters.MaxIt;      % Maximum Number of Iterations
nPop=                       PSOparameters.nPop;        % Population Size (Swarm Size)
wmin=                       PSOparameters.wmin;            % Inertia Weight
wmax=                       PSOparameters.wmax;
c1min=                      PSOparameters.c1min;         % Personal Learning Coefficient
c1max=                      PSOparameters.c1max;
c2min=                      PSOparameters.c2min;         % Global Learning Coefficient
c2max=                      PSOparameters.c2max;
strategy=                   PSOparameters.strategy;
BRM=                        PSOparameters.BRM;
inertia=                    PSOparameters.inertia;
velocity_update_func=       PSOparameters.velocity_update_func;
MatrixSize=[nPop nVar];                     %Repair boundary method employed
% pre-allocation of loop variables
fitMaxVector = nan(2,MaxIt);
%% Initialization
Position=[];
Cost=[];
Velocity=[];
LocalBest.Position=[];
LocalBest.Cost=[];
GlobalBest.Position=[];
GlobalBest.Cost=[];
r1=repmat(rand(nPop,1),1,nVar);
r2=repmat(rand(nPop,1),1,nVar);
minPositionsMatrix=repmat(VarMin,nPop,1);
maxPositionsMatrix=repmat(VarMax,nPop,1);
% generate initial population
Position=genpop(nPop,nVar,VarMin,VarMax);

Position(1,:)=xbest;
Position(2,:)=VarMin;

Velocity=zeros(nPop,nVar);
%------Evaluate the best member after initialization----------------------
% Modified by FLC
switch fnc
    case 'fitnessFun_DER'
    caseStudyData=caseStudyData(1);
end
[solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,Position,caseStudyData, otherParameters,500);

nEvals=nEvals+ (nPop*500);

for i= 1:nPop
    fitt(i)= mean(solFitness_M(i,:))+std(solFitness_M(i,:));
end
disp(min(fitt));

[gbestval,gebstid]=min(fitt);
gbest=Position(gebstid,:);

% Worse performance criterion
[Cost(:,1), worstS]=max(solFitness_M,[],2); %Choose the solution with worse performance
LocalBest.Position=Position;
LocalBest.Cost(:,1)=Cost;
inx=Cost<LocalBest.Cost;
LocalBest.Cost(inx)=Cost(inx);

[gbestvall,idk]=min(LocalBest.Cost);
GlobalBest.Position=LocalBest.Position(idk,:);
GlobalBest.Cost=LocalBest.Cost(idk,1);

FVr_bestmemit = GlobalBest.Position; % best member of current iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% store other information
Best_otherInfo.idBestParticle = idk;
Best_otherInfo.genCostsFinal = Struct_Eval(worstS(idk)).otherParameters.genCosts(idk,:);
Best_otherInfo.loadDRcostsFinal = Struct_Eval(worstS(idk)).otherParameters.loadDRcosts(idk,:);
Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(worstS(idk)).otherParameters.v2gChargeCosts(idk,:);
Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(worstS(idk)).otherParameters.v2gDischargeCosts(idk,:);
Best_otherInfo.storageChargeCostsFinal = Struct_Eval(worstS(idk)).otherParameters.storageChargeCosts(idk,:);
Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(worstS(idk)).otherParameters.storageDischargeCosts(idk,:);
Best_otherInfo.stBalanceFinal = Struct_Eval(worstS(idk)).otherParameters.stBalance(idk,:,:);
Best_otherInfo.v2gBalanceFinal = Struct_Eval(worstS(idk)).otherParameters.v2gBalance(idk,:,:);
Best_otherInfo.penSlackBusFinal = Struct_Eval(worstS(idk)).otherParameters.penSlackBus(idk,:);
fitMaxVector(:,1)=[mean(solFitness_M(idk,:));mean(solPenalties_M(idk,:))]; %We save the mean value and mean penalty value
% The user can decide to save the mean, best, or any other value here
FVr_rot  = (0:1:nPop-1);               % rotating index array (size nPop)
%% PSO Main Loop

% while nEvals+ (nPop*500)<50000

    
    copyPosition=Position;
    copyVel=Velocity;
    
for it=2:MaxIt
    %Inertia update
    ccc=GlobalBest(it-1).Position;
    cc=GlobalBest(it-1).Cost;
    w=inertia_update(inertia,it,MaxIt,wmin,wmax,nPop,nVar);
    % Update coeficients
    [c1,c2]=coef_update(c1min,c1max,c2min,c2max,it,MaxIt);
    % Velocity Limits
    [VelMax,VelMin]=velocity_update(VarMax,VarMin,nPop,it,MaxIt,velocity_update_func);
    % random criation
    [r1,r2]=random_criation(nPop,r1(idk,1),r2(idk,1),nVar);
    % Update Velocity
    switch strategy
        case 1
            Velocity=w.*Velocity+...
                    c1*r1.*(LocalBest.Position-Position)+...
                    c2*r2.*(repmat(GlobalBest(it-1).Position,[nPop,1])-Position);
        case 2
            Velocity=w.*Velocity.*(1+randn([nPop, nVar])).*rand([nPop, nVar])+...
                     c1*r1.*(LocalBest.Position-Position).*randi([0,1],nPop,nVar)+...
                     c2*r2.*(repmat(GlobalBest(it-1).Position,[nPop,1]).*(1+randn([nPop, nVar]))-Position);
        case 3
            Velocity=w.*Velocity.*(1+randn([nPop, nVar])).*rand([nPop, nVar])+...
                     c2*r2.*(repmat(GlobalBest(it-1).Position,[nPop,1]).*(1+randn([nPop, nVar]))-Position);
        case 4
            idp= GlobalBest(it-1).Cost./LocalBest.Cost;
            P_peturbation=0.95-((0.95-0.95)/MaxIt)*it;
            %jj=(idp>=P_peturbation);
           % jjj=sum(idp>=P_peturbation);
            jjjjj=GlobalBest(it-1).Position-(GlobalBest(it-1).Position.*randn([1, nVar]));
           % jjjjjjj=[sum(idp>=P_peturbation), 1];
           % jjjj=repmat(GlobalBest(it-1).Position-(GlobalBest(it-1).Position.*randn([1, nVar])),[sum(idp>=P_peturbation), 1]);
            
            
            Velocity(idp<P_peturbation,:)=w(idp<P_peturbation,:).*Velocity(idp<P_peturbation,:).*(1+randn([sum(idp<P_peturbation), nVar])).*rand([sum(idp<P_peturbation), nVar])+...
                     c2*r2(idp<P_peturbation,:).*(repmat(GlobalBest(it-1).Position,[sum(idp<P_peturbation),1]).*(1+randn([sum(idp<P_peturbation), nVar]))-Position(idp<P_peturbation,:));
            Velocity(idp>=P_peturbation,:)=genpop(sum(idp>=P_peturbation),nVar,jjjjj,jjjjj);
    end
    %end 
    % Apply Velocity Limits
    Velocity =max(Velocity,VelMin);
    Velocity =min(Velocity,VelMax);
    % Update Position
    Position =Position +0.08* Velocity;
    
    
    % Velocity Mirror Effect
    IsOutside=(Position<minPositionsMatrix | Position>maxPositionsMatrix);
    Velocity(IsOutside)=-Velocity(IsOutside);
    % Apply Position Limits Boundary Control
    Position=update(Position,minPositionsMatrix,maxPositionsMatrix,BRM);
    % Evaluation
    [solFitness_M_temp, solPenalties_M_temp,Struct_Eval_temp]=feval(fnc,Position,caseStudyData, otherParameters,500);
    
    for i= 1:nPop
    fitttt(i)= mean(solFitness_M_temp(i,:))+std(solFitness_M_temp(i,:));
    end
disp(min(fitttt));
    
selparnewswarm=(fitttt<fitt);
for i= 1:nPop
    if selparnewswarm(i)
        fitt(i)=fitttt(i);
        Position(i,:)=copyPosition(i,:);
        Velocity(i,:)=copyVel(i,:);
    end
end

[tmpgbestval,gbestid]=min(fitt);
if tmpgbestval < gbestval
    gbestval=tmpgbestval;
    gbest=Position(gebstid,:);
end

    % In this example, we optimize worse performance
    [Cost(:,1), worstS]=max(solFitness_M_temp,[],2); %Choose the solution with worse performance
    % Update Personal Best
    ind=Cost<LocalBest.Cost;
    LocalBest.Cost(ind)=Cost(ind);
    LocalBest.Position(ind,:)=Position(ind,:);
    
    
    % Update Global Best
    [~,idk]=min(LocalBest.Cost);
    GlobalBest(it).Position=LocalBest.Position(idk,:);
    GlobalBest(it).Cost=LocalBest.Cost(idk,1);
    FVr_bestmemit = GlobalBest(it).Position; % best member of current iteration
    
    
    % store fitness evolution and obj fun evolution as well
    fitMaxVector(1,it) = GlobalBest(it).Cost;
    if ismember(idk,ind)
        % store other info
        Best_otherInfo.idBestParticle = idk;
        Best_otherInfo.genCostsFinal = Struct_Eval_temp(worstS(idk)).otherParameters.genCosts(idk,:);
        Best_otherInfo.loadDRcostsFinal = Struct_Eval_temp(worstS(idk)).otherParameters.loadDRcosts(idk,:);
        Best_otherInfo.v2gChargeCostsFinal = Struct_Eval_temp(worstS(idk)).otherParameters.v2gChargeCosts(idk,:);
        Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval_temp(worstS(idk)).otherParameters.v2gDischargeCosts(idk,:);
        Best_otherInfo.storageChargeCostsFinal = Struct_Eval_temp(worstS(idk)).otherParameters.storageChargeCosts(idk,:);
        Best_otherInfo.storageDischargeCostsFinal = Struct_Eval_temp(worstS(idk)).otherParameters.storageDischargeCosts(idk,:);
        Best_otherInfo.stBalanceFinal = Struct_Eval_temp(worstS(idk)).otherParameters.stBalance(idk,:,:);
        Best_otherInfo.v2gBalanceFinal = Struct_Eval_temp(worstS(idk)).otherParameters.v2gBalance(idk,:,:);
        Best_otherInfo.penSlackBusFinal = Struct_Eval_temp(worstS(idk)).otherParameters.penSlackBus(idk,:);
         
        fitMaxVector(:,it)= [mean(solFitness_M_temp(idk,:));mean(solPenalties_M_temp(idk,:))];
    elseif it>1
        fitMaxVector(:,it)=fitMaxVector(:,it-1);
    end 
    it=it+1;
end
   disp(min(fitt))
p1=sum(Best_otherInfo.penSlackBusFinal);
Fit_and_p=[fitMaxVector(1,it-1) p1]; %;p2;p3;p4]

%% Adition function
% VECTORIZED THE CODE INSTEAD OF USING FOR
function pop=genpop(a,b,VarMin,VarMax)
for i=1:a
pop(i,:)=VarMin+(VarMax-VarMin).*rand(1,b);
end
%pop=unifrnd(lowMatrix,upMatrix,a,b);

% INERTIA CALCULATION
function w=inertia_update(inertia,it,MaxIt,wmin,wmax,nPop,nVar)
switch inertia
    case 1
        w=wmax-((wmax-wmin)/MaxIt)*it;
%     case 2
%         w=1.1+(GlobalBest_Position/mean(LocalBest_Position));
end
w=repmat(w,[nPop,nVar]);
% Update coeficients
function [c1,c2]=coef_update(c1min,c1max,c2min,c2max,it,MaxIt)
c1=c1max-((c1max-c1min)/MaxIt)*it; % decreasing
c2=c2min+((c2max-c2min)/MaxIt)*it; % increasing

% Velocity Limits
function [VelMax,VelMin]=velocity_update(VarMax,VarMin,nPop,it,MaxIt,velocity_update_func)
switch velocity_update_func
    case 1
        f=0.5-((0.5-0.1)/MaxIt)*it;
        VelMax=f*(VarMax-VarMin);
        VelMin=-VelMax;
end
VelMax=repmat(VelMax,[nPop,1]);
VelMin=repmat(VelMin,[nPop,1]);

% random criation
function[r1,r2]=random_criation(nPop,r1mean,r2mean,nVar)
r1pd=makedist('Normal','mu',r1mean,'sigma',0.001);
r2pd=makedist('Normal','mu',r2mean,'sigma',0.001);
r1=repmat(random(r1pd,[nPop,1]),1,nVar);
r2=repmat(random(r2pd,[nPop,1]),1,nVar);

%  VECTORIZED THE CODE INSTEAD OF USING FOR
function Position=update(Position,minPositionsMatrix,maxPositionsMatrix,BRM)
switch BRM
    case 1 % max and min replace
        Position = max(Position,minPositionsMatrix);
        Position = min(Position,maxPositionsMatrix);
    case 2 %Random reinitialization
        IsOutside=[find(Position<minPositionsMatrix);find(Position>maxPositionsMatrix)];
        Position(IsOutside)=unifrnd(minPositionsMatrix(IsOutside),maxPositionsMatrix(IsOutside),length(IsOutside),1);
    
end
