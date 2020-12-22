function [BestSol] = pso(PSOparameters)
%% Problem Definition

CostFunction=@(x) Sphere(x);        % Cost Function
nVar=10;            % Number of Decision Variables
VarMin=-10.*ones(1,nVar);         % Lower Bound of Variables
VarMax= 10.*ones(1,nVar);         % Upper Bound of Variables

%% PSO Parameters

MaxIt=          PSOparameters.MaxIt;      % Maximum Number of Iterations
nPop=           PSOparameters.nPop;        % Population Size (Swarm Size)
wmin=           PSOparameters.wmin;            % Inertia Weight
wmax=           PSOparameters.wmax;
c1min=          PSOparameters.c1min;         % Personal Learning Coefficient
c1max=          PSOparameters.c1max;
c2min=          PSOparameters.c2min;         % Global Learning Coefficient
c2max=          PSOparameters.c2max;
strategy=       PSOparameters.strategy;     % strategy for boundary control 
MatrixSize=[nPop nVar];
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
Position=genpop(nPop,nVar,minPositionsMatrix,maxPositionsMatrix);
Velocity=zeros(nPop,nVar);
Cost(:,1)=CostFunction(Position);
LocalBest.Position=Position;
LocalBest.Cost(:,1)=Cost;
inx=Cost<LocalBest.Cost;
LocalBest.Cost(inx)=Cost(inx);
[~,idk]=min(LocalBest.Cost);
GlobalBest.Position=LocalBest.Position(idk,:);
GlobalBest.Cost=LocalBest.Cost(idk,1);

%% PSO Main Loop

for it=2:MaxIt
    %Inertia update
    w=inertia(it,MaxIt,wmin,wmax);
    % Update coeficients
    [c1,c2]=coef_update(c1min,c1max,c2min,c2max,it,MaxIt);
    % Velocity Limits
    [VelMax,VelMin]=velocity_update(VarMax,VarMin,nPop,it,MaxIt);
    % random criation
    [r1,r2]=random_criation(nPop,r1(idk,1),r2(idk,1),nVar);
    % Update Velocity
    Velocity=w*Velocity+...
             c1.*r1.*(LocalBest.Position-Position)+...
             c2.*r2.*(repmat(GlobalBest(it-1).Position,[nPop,1])-Position);
    % Apply Velocity Limits
    Velocity =max(Velocity,VelMin);
    Velocity =min(Velocity,VelMax);
    % Update Position
    Position =Position + Velocity;
    % Velocity Mirror Effect
    IsOutside=(Position<minPositionsMatrix | Position>maxPositionsMatrix);
    Velocity(IsOutside)=-Velocity(IsOutside);
    % Apply Position Limits
    switch strategy
        case 1
            Position = max(Position,minPositionsMatrix);
            Position = min(Position,maxPositionsMatrix);
        case 2
            IsOutsidemax=(Position>maxPositionsMatrix);
            
            IsOutsidemin=(Position<minPositionsMatrix);
        case 3
            fator=sum(IsOutside,2);
            penalty=fator.*100;
    end
    % Evaluation
    Cost(:,1)=CostFunction(Position);
    if strategy==3
        Cost=Cost+penalty;
    end
    % Update Personal Best
    ind=Cost<LocalBest.Cost;
    LocalBest.Cost(ind)=Cost(ind);
    LocalBest.Position(ind,:)=Position(ind,:);
    % Update Global Best
    [~,idk]=min(LocalBest.Cost);
    GlobalBest(it).Position=LocalBest.Position(idk,:);
    GlobalBest(it).Cost=LocalBest.Cost(idk,1);
    it=it+1;
end

BestSol.Best=GlobalBest(end).Cost;
BestSol.Position=GlobalBest(end).Position;

%% Adition function
% VECTORIZED THE CODE INSTEAD OF USING FOR
function pop=genpop(a,b,lowMatrix,upMatrix)
pop=unifrnd(lowMatrix,upMatrix,a,b);

% INERTIA CALCULATION
function w=inertia(it,MaxIt,wmin,wmax)
w=wmax-((wmax-wmin)/MaxIt)*it;

% Update coeficients
function [c1,c2]=coef_update(c1min,c1max,c2min,c2max,it,MaxIt)
c1=c1max-((c1max-c1min)/MaxIt)*it; % decreasing
c2=c2min+((c2max-c2min)/MaxIt)*it; % increasing

% Velocity Limits
function [VelMax,VelMin]=velocity_update(VarMax,VarMin,nPop,it,MaxIt)
f=0.1-((0.1-0.1)/MaxIt)*it;
VelMax=f*(VarMax-VarMin);
VelMin=-VelMax;
VelMax=repmat(VelMax,[nPop,1]);
VelMin=repmat(VelMin,[nPop,1]);

% random criation
function[r1,r2]=random_criation(nPop,r1mean,r2mean,nVar)
r1pd=makedist('Normal','mu',r1mean,'sigma',0.001);
r2pd=makedist('Normal','mu',r2mean,'sigma',0.001);
r1=repmat(random(r1pd,[nPop,1]),1,nVar);
r2=repmat(random(r2pd,[nPop,1]),1,nVar);

