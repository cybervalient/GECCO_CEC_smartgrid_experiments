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
% IEEE 2017 Competition & panel: Evaluating the Performance of Modern Heuristic
% Optimizers on Smart Grid Operation Problems: Variable Neighborhood Search algorithm (VNS)  

function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = VNS2(deParameters,caseStudyData,otherParameters,xmin,xmax)
%%    
%-----This is just for notational convenience and to keep the code uncluttered.--------
I_NP         = deParameters.I_NP;
F_weight     = deParameters.F_weight;
F_CR         = deParameters.F_CR;
I_D          = numel(xmin); %Number of variables or dimension
deParameters.nVariables=I_D;
FVr_minbound = xmin;
FVr_maxbound = xmax;
I_itermax    = deParameters.I_itermax;

%Repair boundary method employed
BRM=deParameters.I_bnd_constr; %1: bring the value to bound violated
                               %2: repair in the allowed range

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_strategy   = deParameters.I_strategy; %important variable
fnc= otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----parameters-----------------------------------------------------------
% FLC modification - vectorization
minPositionsMatrix=repmat(FVr_minbound,I_NP,1);
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1);
deParameters.minPositionsMatrix=minPositionsMatrix;
deParameters.maxPositionsMatrix=maxPositionsMatrix;

% generate initial population.
%FM_pop=genpop(I_NP,I_D,minPositionsMatrix,maxPositionsMatrix);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------Evaluate the best member after initialization----------------------
% Modified by FLC
switch fnc
    case 'fitnessFun_DER'
    caseStudyData=caseStudyData(1);
end
%%
%% Vectors to store the solutions %% VNS start
%Niter = 50000 - 1;
%Niter = (deParameters.I_iterma+1) -1;
%Niter=round((deParameters.I_iterma+1)*0.61);
Niter=243; % this paremater was tuned
fitMaxVector = nan(1,Niter+1);
objMaxVector = nan(2,Niter+1);

%% Parameters
% Dimension of the problem
iter = 0;
nper = 24;
[~,D] = size(xmin);
nctrl = D/nper;
penalty = 10000;

Xmin = vec2mat(xmin,nctrl);
Xmax = vec2mat(xmax,nctrl);

%% Types of variables, this was modifed according to the 2018 competition

% Indices
Pdgind = otherParameters.ids.idsGen;      % Active power of a DG
%Qdgind = otherParameters.ids.idsQGen;     % Reactive power of a DG
Xdgind = otherParameters.ids.idsXGen;     % Binary status of a DG

V2Gind = otherParameters.ids.idsV2G;      % EV charging/discharging
DRind  = otherParameters.ids.idsLoadDR;   % Loads for demand response
ESSind = otherParameters.ids.idsStorage;  % ESS charging/discharging
MKind  = otherParameters.ids.idsMarket;   % Market
%Tapind = otherParameters.ids.idsTAP;      % DSS OLTC

% Number of controls
nDG  = length(Pdgind);
nV2G = length(V2Gind);
nDR  = length(DRind);
nESS = length(ESSind);
nMK  = length(MKind);
%nTAP = length(Tapind);

% Variable type
xtype = zeros(1,nctrl);

xtype(Pdgind(1):Pdgind(nDG))  = 1;
%xtype(Qdgind(1):Qdgind(nDG))  = 2;
xtype(Xdgind(1):Xdgind(nDG))  = 2;
xtype(V2Gind(1):V2Gind(nV2G)) = 3;
xtype(DRind(1):DRind(nDR))    = 4;
xtype(ESSind(1):ESSind(nESS)) = 5;
xtype(MKind(1):MKind(nMK))    = 6;
%xtype(Tapind(1):Tapind(nTAP)) = 8;

xtype = repmat(xtype,1,nper);

% Vector of fixed controls
xfix = zeros(1,D);

for i = 1:D
    if xmin(i) == xmax(i)
        xfix(i) = 1;
    end
end
nfix = sum(xfix); % Number of fixed variables

% Costs
% In
Loadc = caseStudyData(1).loadData.CofLoad;
MKc = caseStudyData(1).marketData.sellCofB;
ESSc_ch = caseStudyData(1).storageData.stPriceCharge;
V2Gc_ch = caseStudyData(1).v2gData.v2gCofCharge;

% OC
DGc = caseStudyData(1).genData.genCofB;
DGC = repmat(DGc,1,2);

%?energy price of external supplier S %%%%%%%%%
Loadredc = caseStudyData(1).loadData.CofCut;
Loadredc = caseStudyData(1).loadData.CofReduce;

V2Gc_dis = caseStudyData(1).v2gData.v2gCofDischarge;
ESSc_dis = caseStudyData(1).storageData.stPriceDischarge;
ENSc = caseStudyData(1).loadData.CofENS;

%% Random Initial Solution
x = zeros(1,D);
for i = 1:D
    x(i) = xmin(i) + rand*(xmax(i) - xmin(i));
end

for i = 1:D
    if x(i) < xmin(i)
        x(i) = xmin(i);
    elseif x(i) > xmax(i)
        x(i) = xmax(i);  
    end
end

nEvals=0;

whileIter=0;

%% Random Initial Solution
x = zeros(1,D);
for i = 1:D
    x(i) = (xmin(i) + xmax(i))/2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if iter < Niter
    
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
        
            nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

            for i=1:deParameters.I_NP   % Eval Initial population
                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                    obj(i)  = FIT(i); 
                    fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
            end
       
            xrep=x;
        %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
        %[fit,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
    iter = iter + 1;
    fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];
    %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',0,0,iter,fit,obj(2)-obj(1),fit+obj(1)-obj(2),obj(2),obj(1));

    FITBEST = FIT;
    OBJBEST = obj;
    fitMaxVector(iter) = FITBEST;
    objMaxVector(:,iter)= OBJBEST(1,:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fbest = fit
objbest = obj;
xbest = xrep;

%% Improve initial solution
display('Improve initial solution')

X = vec2mat(xbest,nctrl);

for i = 1:nper
    for j = 1:nctrl
        if xtype(j) == 3
            X(i,j) = Xmin(i,j) + (Xmax(i,j) - Xmin(i,j))/2;
        elseif xtype(j) == 4
            X(i,j) = Xmin(i,j);
        elseif xtype(j) == 6
            X(i,j) = Xmin(i,j);
        end
    end
end

iimp = [1, 2, 4, 5, 6];
iimpn = length(iimp);

betat = nan(1,iimpn);

for kn = 1:iimpn
    for beta = 0:0.1:1
        for i = 1:nper
            for j = 1:nctrl
                if xtype(j) == iimp(kn)
                    X(i,j) = Xmin(i,j) + beta*(Xmax(i,j) - Xmin(i,j));
                end
            end
        end

        k = 1;
        for i = 1:nper
            for j = 1:nctrl
                x(k) = X(i,j);
                k = k + 1;
            end
        end

        for i = 1:D
            if x(i) < xmin(i)
                x(i) = xmin(i);
            elseif x(i) > xmax(i)
                x(i) = xmax(i);  
            end
        end    

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if iter < Niter
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
        
            nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

            for i=1:deParameters.I_NP   % Eval Initial population
                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                    obj(i)  = FIT(i); 
                    fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
            end
       
            xrep=x;
            
            %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
            %[fit,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
            iter = iter + 1;
            fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];

            if FIT < FITBEST
                FITBEST = FIT;
                OBJBEST = obj;
                fitMaxVector(iter) = FITBEST;
                objMaxVector(:,iter)= OBJBEST(1,:);
            else
                fitMaxVector(iter) = FITBEST;
                objMaxVector(:,iter)= OBJBEST(1,:);
            end
        else
            break;
            1001
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if fit < fbest
            %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',0,kn,iter,fit,obj(2)-obj(1),fit+obj(1)-obj(2),obj(2),obj(1));

            fbest = fit
            nEvals
            objbest = obj;
            xbest = xrep;
            
            betat(kn) = beta;
        end
    end
    X = vec2mat(xbest,nctrl);
end


%% Evaluate initial improved solution
display('Evaluate initial improved solution')
nEvals
fbest

k = 1;
for i = 1:nper
    for j = 1:nctrl
        x(k) = X(i,j);
        k = k + 1;
    end
end

for i = 1:D
    if x(i) < xmin(i)
        x(i) = xmin(i);
    elseif x(i) > xmax(i)
        x(i) = xmax(i);  
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if iter < Niter
    
            [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
        
            nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

            for i=1:deParameters.I_NP   % Eval Initial population
                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                    obj(i)  = FIT(i); 
                    fit(i)  = mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:));
            end
       
            xrep=x;
        
    %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
    %[fit,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,x,Xmax,Xmin,penalty,caseStudyData,otherParameters);
    iter = iter + 1;
    fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];
    %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',0,0,iter,fit,obj(2)-obj(1),fit+obj(1)-obj(2),obj(2),obj(1));

    if FIT < FITBEST
        FITBEST = FIT;
        OBJBEST = obj;
        fitMaxVector(iter) = FITBEST;
        objMaxVector(:,iter)= OBJBEST(1,:);
    else
        fitMaxVector(iter) = FITBEST;
        objMaxVector(:,iter)= OBJBEST(1,:);
    end
end

%% Cyclic Coordinated Method
display('Cyclic Coordinated Method')
fbest
nEvals

xbest = xrep;
fbest = fit;
objbest = obj;

% Convert x into Xr
% 1- Convert x into X
X = vec2mat(xbest,nctrl);

k = 1;
for i = 1:nper
    for j = 1:nctrl
        x(k) = X(i,j);
        k = k + 1;
    end
end

for i = 1:D
    if x(i) < xmin(i)
        x(i) = xmin(i);
        %pause
    elseif x(i) > xmax(i)
        x(i) = xmax(i);  
        %pause
    end
end

% 2 - Convert X into Xr
%    1    2    3    4   5    6       7    8
% [Pdg, Qdg, Xdg, V2G, DR, ESS, Market, TAP] 8x24 = 120

% 2 - Convert X into Xr
%    1    2    3   4    5       6      
% [Pdg, Qdg, V2G, DR, ESS, Market] 6x24 = ???

Xr = zeros(nper,6);
Xr(:,1) = sum(X(:,Pdgind(1):Pdgind(nDG))')';
Xr(:,2) = sum(X(:,Xdgind(1):Xdgind(nDG))')';
Xr(:,3) = sum(X(:,V2Gind(1):V2Gind(nV2G))')';
Xr(:,4) = sum(X(:,DRind(1):DRind(nDR))')';
Xr(:,5) = sum(X(:,ESSind(1):ESSind(nESS))')';
Xr(:,6) = sum(X(:,MKind(1):MKind(nMK))')';

% Lower and upper limits
Xrmin = zeros(nper,6);
Xrmin(:,1) = sum(Xmin(:,Pdgind(1):Pdgind(nDG))')';
Xrmin(:,2) = sum(Xmin(:,Xdgind(1):Xdgind(nDG))')';
Xrmin(:,3) = sum(Xmin(:,V2Gind(1):V2Gind(nV2G))')';
Xrmin(:,4) = sum(Xmin(:,DRind(1):DRind(nDR))')';
Xrmin(:,5) = sum(Xmin(:,ESSind(1):ESSind(nESS))')';
Xrmin(:,6) = sum(Xmin(:,MKind(1):MKind(nMK))')';

Xrmax = zeros(nper,6);
Xrmax(:,1) = sum(Xmax(:,Pdgind(1):Pdgind(nDG))')';
Xrmax(:,2) = sum(Xmax(:,Xdgind(1):Xdgind(nDG))')';
Xrmax(:,3) = sum(Xmax(:,V2Gind(1):V2Gind(nV2G))')';
Xrmax(:,4) = sum(Xmax(:,DRind(1):DRind(nDR))')';
Xrmax(:,5) = sum(Xmax(:,ESSind(1):ESSind(nESS))')';
Xrmax(:,6) = sum(Xmax(:,MKind(1):MKind(nMK))')';

% Fibonacci sequence
F_fb = zeros(1,100);
F_fb(1) = 1;
F_fb(2) = 1;
for i_fb = 3:100
    F_fb(i_fb) = F_fb(i_fb-1) + F_fb(i_fb-2);
end
    
Xrbest = Xr;
Xbest = X;

% Number of repetitions
WW = 1;

for ww = 1:WW    
    ww
    % Sequence for the CCM
    if ww == 1
        seq = [1, 2, 3, 4, 5, 6];
    else
        seq = [1, 2, 3, 4, 5, 6];
    end
    nvXr = length(seq);
    
    for ii = 1:nper
        for w = 1:nvXr         
            %% Repair the best solution
            x = xbest;                
            for i = 1:D
                if x(i) < xmin(i)
                    x(i) = xmin(i);
                elseif x(i) > xmax(i)
                    x(i) = xmax(i);  
                end
                
                if abs(x(i) - xmin(i)) < 10^-3
                    x(i) = xmin(i);
                elseif abs(xmax(i) - x(i)) < 10^-3
                    x(i) = xmax(i);
                end
            end            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if iter < Niter
                    [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
        
                    nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

                    for i=1:deParameters.I_NP   % Eval Initial population
                        FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                        obj(i)  = FIT(i); 
                        frep(i) =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
                    end
       
                    xrep=x;

                %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
                %[frep,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
                iter = iter + 1;
                fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];

                if FIT < FITBEST
                    FITBEST = FIT;
                    OBJBEST = obj;
                    fitMaxVector(iter) = FITBEST;
                    objMaxVector(:,iter)= OBJBEST(1,:);
                else
                    fitMaxVector(iter) = FITBEST;
                    objMaxVector(:,iter)= OBJBEST(1,:);
                end
            else
                break;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if frep < fbest
                fbest = frep
                nEvals
                101
                objbest = obj;
                xbest = xrep;
                Xbest = vec2mat(xbest,nctrl);

                Xrbest = zeros(nper,6);
                Xrbest(:,1) = sum(Xbest(:,Pdgind(1):Pdgind(nDG))')';
                Xrbest(:,2) = sum(Xbest(:,Xdgind(1):Xdgind(nDG))')';
                Xrbest(:,3) = sum(Xbest(:,V2Gind(1):V2Gind(nV2G))')';
                Xrbest(:,4) = sum(Xbest(:,DRind(1):DRind(nDR))')';
                Xrbest(:,5) = sum(Xbest(:,ESSind(1):ESSind(nESS))')';
                Xrbest(:,6) = sum(Xbest(:,MKind(1):MKind(nMK))')';
                
            end
            
            Xr = Xrbest;
            X = Xbest;            
            %% Lambda and mu calculation
            a_fb = Xrmin(ii,seq(w));
            b_fb = Xrmax(ii,seq(w));
            
            flag = 0;
            if a_fb == b_fb
                flag = 1;
            end
            
            if flag == 0
%                 if (abs(b_fb - a_fb)/1000) < (10^-3)
                    l_fb = abs(b_fb - a_fb)/10;
%                 else
%                     l_fb = 10^-3;
%                 end

                n_fb = 1; 
                while F_fb(n_fb) <= (b_fb - a_fb)/l_fb
                    n_fb = n_fb + 1;                
                end

                lambda_fb = a_fb + (F_fb(n_fb-1)/F_fb(n_fb+1))*(b_fb - a_fb);
                mu_fb = a_fb + (F_fb(n_fb)/F_fb(n_fb+1))*(b_fb - a_fb);

                %%
                x_lb = Xr;
                x_lb(ii,seq(w)) = lambda_fb;            
                Xlb = X;

                % Convert x_lb into Xlb
                if seq(w) == 1      % Pdg
                    % Sequence of the cheapest generators
                    [~,cdg] = sort(DGc(ii,:));                
                    % Available active power to be distributed to the cheapest generators
                    pav = x_lb(ii,seq(w)) - Xrmin(ii,seq(w));

                    % Fix all variables in the lower limit
                    for j = Pdgind(1):Pdgind(nDG)
                        Xlb(ii,j) = Xmin(ii,j);
                    end
                    
                    for j = Xdgind(1):Xdgind(nDG)
                        Xlb(ii,j) = Xmin(ii,j);
                    end

                    for j = 1:nDG
                        k = cdg(j);                        
                        Xlb(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                        if (Xmax(ii,k) - Xmin(ii,k) > 0) && (pav >= Xmax(ii,k) - Xmin(ii,k))
                            Xlb(ii,k) = Xmax(ii,k);
                            Xlb(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                            
                            pav = pav - (Xmax(ii,k) - Xmin(ii,k));
                        elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (pav > 0) && (pav < Xmax(ii,k) - Xmin(ii,k))
                            Xlb(ii,k) = Xmin(ii,k) + pav;
                            Xlb(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                            
                            pav = 0;                        
                        elseif pav == 0
                            break;                        
                        end
                    end 
%                 elseif seq(w) == 2  % Qdg
%                     % Sequence of the cheapest generators
%                     [~,cdg] = sort(DGc(ii,:));
%                     cdg = cdg + nDG;
%                     % Available reactive power to be distributed to the cheapest generators
%                     qav = x_lb(ii,seq(w)) - Xrmin(ii,seq(w));
% 
%                     % Fix all variables in the lower limit
%                     for j = Qdgind(1):Qdgind(nDG)
%                         Xlb(ii,j) = Xmin(ii,j);
%                     end
% 
%                     for j = 1:nDG
%                         k = cdg(j);
%                         if (Xmax(ii,k) - Xmin(ii,k) > 0) && (qav >= Xmax(ii,k) - Xmin(ii,k))
%                             Xlb(ii,k) = Xmax(ii,k);
%                             qav = qav - (Xmax(ii,k) - Xmin(ii,k));
%                         elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (qav > 0) && (qav < Xmax(ii,k) - Xmin(ii,k))
%                             Xlb(ii,k) = Xmin(ii,k) + qav;
%                             qav = 0;                        
%                         elseif qav == 0
%                             break;                        
%                         end
%                     end
                elseif seq(w) == 2  % Xdg
                    % Sequence of the cheapest generators
                    [~,cdg] = sort(DGc(ii,:));
                    cdg = cdg + 2*nDG;

                    % Number of operating generators to be distributed to the cheapest generators
                    xav = x_lb(ii,seq(w)) - Xrmin(ii,seq(w));

                    % Fix all variables in the lower limit
                    for j = Xdgind(1):Xdgind(nDG)
                        Xlb(ii,j) = Xmin(ii,j);
                    end

                    for j = 1:nDG
                        k = cdg(j);
                        if (Xmax(ii,k) - Xmin(ii,k) > 0) && (xav >= Xmax(ii,k) - Xmin(ii,k))
                            Xlb(ii,k) = Xmax(ii,k);
                            xav = xav - (Xmax(ii,k) - Xmin(ii,k));
                        elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (xav > 0) && (xav < Xmax(ii,k) - Xmin(ii,k))
                            Xlb(ii,k) = Xmax(ii,k);
                            xav = 0;                        
                        elseif xav == 0
                            break;                        
                        end
                    end              
                elseif seq(w) == 3  % V2G
                    pav2g = (x_lb(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));
                    if pav2g > 1
                        pav2g = 1;
                    elseif pav2g < 0
                        pav2g = 0;
                    end                  
                    for j = V2Gind(1):V2Gind(nV2G)
                        Xlb(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pav2g;                    
                    end
                elseif seq(w) == 4  % DR
                    pavdr = (x_lb(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));
                    if pavdr > 1
                        pavdr = 1;
                    elseif pavdr < 0
                        pavdr = 0;
                    end                  
                    for j = DRind(1):DRind(nDR)
                        Xlb(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pavdr;                    
                    end              
                elseif seq(w) == 5  % ESS
                    pavESS = (x_lb(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));                
                    if pavESS > 1
                        pavESS = 1;
                    elseif pavESS < 0
                        pavESS = 0;
                    end               

                    for j = ESSind(1):ESSind(nESS)
                        Xlb(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pavESS;                    
                    end
                elseif seq(w) == 6  % Market
                    Xlb(ii,MKind(1)) = x_lb(ii,seq(w));                 
                end

                % Convert Xlb in xlb
                xlb = zeros(1,D);
                k = 1;
                for i = 1:nper
                    for j = 1:nctrl
                        xlb(k) = Xlb(i,j);
                        k = k + 1;
                    end
                end

                for i = 1:D
                    if xlb(i) < xmin(i)
                        %%fprintf('LI %4d',i);
                        xlb(i) = xmin(i);
                        %pause
                    elseif xlb(i) > xmax(i)
                        %%fprintf('LS %4d',i);
                        xlb(i) = xmax(i);  
                        %pause
                    end
                end            

                % Lambda evaluation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if iter < Niter
                   [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xlb,caseStudyData, otherParameters,deParameters.Scenarios);
        
                    nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

                    for i=1:deParameters.I_NP   % Eval Initial population
                        FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                        olb(i)  = FIT(i); 
                        flb(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
                    end
       
                    xrep=xlb;
                    
                    %[FIT,olb,xrep,otherParameters] = fitnessFun_DER(xlb,caseStudyData,otherParameters);
                    %[flb,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(olb,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
                    iter = iter + 1;
                    fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];

                    if FIT < FITBEST
                        FITBEST = FIT;
                        OBJBEST = olb;
                        fitMaxVector(iter) = FITBEST;
                        objMaxVector(:,iter)= OBJBEST(1,:);
                    else
                        fitMaxVector(iter) = FITBEST;
                        objMaxVector(:,iter)= OBJBEST(1,:);
                    end
                else
                    break;
                    1002
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                if flb < fbest
                    %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,flb,olb(2)-olb(1),flb+olb(1)-olb(2),olb(2),olb(1));

                    fbest = flb
                    nEvals
                    102
                    objbest = olb;
                    xbest = xlb;

                    Xbest = Xlb;
                    Xrbest = x_lb;
                end

                %%
                x_mu = Xr;
                x_mu(ii,seq(w)) = mu_fb;
                Xmu = X;

                % Convert x_mu into Xmu
                if seq(w) == 1      % Pdg
                    % Sequence of the cheapest generators
                    [~,cdg] = sort(DGc(ii,:));                
                    % Available active power to be distributed to the cheapest generators
                    pav = x_mu(ii,seq(w)) - Xrmin(ii,seq(w));

                    % Fix all variables in the lower limit
                    for j = Pdgind(1):Pdgind(nDG)
                        Xmu(ii,j) = Xmin(ii,j);
                    end
                    
                    for j = Xdgind(1):Xdgind(nDG)
                        Xmu(ii,j) = Xmin(ii,j);
                    end

                    for j = 1:nDG
                        k = cdg(j);                        
                        Xmu(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                        if (Xmax(ii,k) - Xmin(ii,k) > 0) && (pav >= Xmax(ii,k) - Xmin(ii,k))
                            Xmu(ii,k) = Xmax(ii,k);                            
                            Xmu(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                            
                            pav = pav - (Xmax(ii,k) - Xmin(ii,k));
                        elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (pav > 0) && (pav < Xmax(ii,k) - Xmin(ii,k))
                            Xmu(ii,k) = Xmin(ii,k) + pav;
                            Xmu(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                            
                            pav = 0;                        
                        elseif pav == 0
                            break;                        
                        end
                    end 
%                 elseif seq(w) == 2  % Qdg
%                     % Sequence of the cheapest generators
%                     [~,cdg] = sort(DGc(ii,:));
%                     cdg = cdg + nDG;
%                     % Available reactive power to be distributed to the cheapest generators
%                     qav = x_mu(ii,seq(w)) - Xrmin(ii,seq(w));
% 
%                     % Fix all variables in the lower limit
%                     for j = Qdgind(1):Qdgind(nDG)
%                         Xmu(ii,j) = Xmin(ii,j);
%                     end
% 
%                     for j = 1:nDG
%                         k = cdg(j);
%                         if (Xmax(ii,k) - Xmin(ii,k) > 0) && (qav >= Xmax(ii,k) - Xmin(ii,k))
%                             Xmu(ii,k) = Xmax(ii,k);
%                             qav = qav - (Xmax(ii,k) - Xmin(ii,k));
%                         elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (qav > 0) && (qav < Xmax(ii,k) - Xmin(ii,k))
%                             Xmu(ii,k) = Xmin(ii,k) + qav;
%                             qav = 0;                        
%                         elseif qav == 0
%                             break;                        
%                         end
%                     end
                elseif seq(w) == 2  % Xdg
                    % Sequence of the cheapest generators
                    [~,cdg] = sort(DGc(ii,:));
                    cdg = cdg + 2*nDG;

                    % Number of operating generators to be distributed to the cheapest generators
                    xav = x_mu(ii,seq(w)) - Xrmin(ii,seq(w));

                    % Fix all variables in the lower limit
                    for j = Xdgind(1):Xdgind(nDG)
                        Xmu(ii,j) = Xmin(ii,j);
                    end

                    for j = 1:nDG
                        k = cdg(j);
                        if (Xmax(ii,k) - Xmin(ii,k) > 0) && (xav >= Xmax(ii,k) - Xmin(ii,k))
                            Xmu(ii,k) = Xmax(ii,k);
                            xav = xav - (Xmax(ii,k) - Xmin(ii,k));
                        elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (xav > 0) && (xav < Xmax(ii,k) - Xmin(ii,k))
                            Xmu(ii,k) = Xmax(ii,k);
                            xav = 0;                        
                        elseif xav == 0
                            break;                        
                        end
                    end
                elseif seq(w) == 3  % V2G
                    pav2g = (x_mu(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));
                    if pav2g > 1
                        pav2g = 1;
                    elseif pav2g < 0
                        pav2g = 0;
                    end                  
                    for j = V2Gind(1):V2Gind(nV2G)
                        Xmu(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pav2g;                    
                    end
                elseif seq(w) == 4  % DR
                    pavdr = (x_mu(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));
                    if pavdr > 1
                        pavdr = 1;
                    elseif pavdr < 0
                        pavdr = 0;
                    end                  
                    for j = DRind(1):DRind(nDR)
                        Xmu(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pavdr;                    
                    end
                elseif seq(w) == 5  % ESS
                    pavESS = (x_mu(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));                
                    if pavESS > 1
                        pavESS = 1;
                    elseif pavESS < 0
                        pavESS = 0;
                    end               

                    for j = ESSind(1):ESSind(nESS)
                        Xmu(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pavESS;                    
                    end
                elseif seq(w) == 6  % Market
                    Xmu(ii,MKind(1)) = x_mu(ii,seq(w));
                end

                % Convert Xmu in xmu
                xmu = zeros(1,D);
                k = 1;
                for i = 1:nper
                    for j = 1:nctrl
                        xmu(k) = Xmu(i,j);
                        k = k + 1;
                    end
                end

                for i = 1:D
                    if xmu(i) < xmin(i)
                        %%fprintf('LI %4d',i);
                        xmu(i) = xmin(i);
                        %pause
                    elseif xmu(i) > xmax(i)
                        %%fprintf('LS %4d',i);
                        xmu(i) = xmax(i);  
                        %pause
                    end
                end            

                % Mu evaluation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if iter < Niter
                    [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xmu,caseStudyData, otherParameters,deParameters.Scenarios);
        
                    nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

                    for i=1:deParameters.I_NP   % Eval Initial population
                        FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                        omu(i)  = FIT(i); 
                        fmu(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
                    end
       
                    xrep=xmu;
                    %[FIT,omu,xrep,otherParameters] = fitnessFun_DER(xmu,caseStudyData,otherParameters);
                    %[fmu,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(omu,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
                    iter = iter + 1;
                    fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];

                    if FIT < FITBEST
                        FITBEST = FIT;
                        OBJBEST = omu;
                        fitMaxVector(iter) = FITBEST;
                        objMaxVector(:,iter)= OBJBEST(1,:);
                    else
                        fitMaxVector(iter) = FITBEST;
                        objMaxVector(:,iter)= OBJBEST(1,:);
                    end
                else
                    break;
                    1003
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                if fmu < fbest
                    %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,fmu,omu(2)-omu(1),fmu+omu(1)-omu(2),omu(2),omu(1));

                    fbest = fmu
                    nEvals
                    103
                    objbest = omu;
                    xbest = xmu;

                    Xbest = Xmu;
                    Xrbest = x_mu;                
                end

                %%
                for k_fb = 1:n_fb-2
                    %%
                    if flb > fmu
                        a_fb = lambda_fb;

                        lambda_fb = mu_fb;
                        mu_fb = a_fb + (F_fb(n_fb-k_fb)/F_fb(n_fb+1-k_fb))*(b_fb - a_fb);

                        if k_fb == n_fb-2
                            break
                        end

                        x_mu(ii,seq(w)) = mu_fb;

                        % Convert x_mu into Xmu
                        if seq(w) == 1      % Pdg
                            % Sequence of the cheapest generators
                            [~,cdg] = sort(DGc(ii,:));                
                            % Available active power to be distributed to the cheapest generators
                            pav = x_mu(ii,seq(w)) - Xrmin(ii,seq(w));

                            % Fix all variables in the lower limit
                            for j = Pdgind(1):Pdgind(nDG)
                                Xmu(ii,j) = Xmin(ii,j);
                            end
                            
                            for j = Xdgind(1):Xdgind(nDG)
                                Xmu(ii,j) = Xmin(ii,j);
                            end

                            for j = 1:nDG
                                k = cdg(j);
                                Xmu(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                                if (Xmax(ii,k) - Xmin(ii,k) > 0) && (pav >= Xmax(ii,k) - Xmin(ii,k))
                                    Xmu(ii,k) = Xmax(ii,k);
                                    Xmu(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                                    
                                    pav = pav - (Xmax(ii,k) - Xmin(ii,k));
                                elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (pav > 0) && (pav < Xmax(ii,k) - Xmin(ii,k))
                                    Xmu(ii,k) = Xmin(ii,k) + pav;
                                    Xmu(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                                    
                                    pav = 0;                        
                                elseif pav == 0
                                    break;                        
                                end
                            end 
%                         elseif seq(w) == 2  % Qdg
%                             % Sequence of the cheapest generators
%                             [~,cdg] = sort(DGc(ii,:));
%                             cdg = cdg + nDG;
%                             % Available reactive power to be distributed to the cheapest generators
%                             qav = x_mu(ii,seq(w)) - Xrmin(ii,seq(w));
% 
%                             % Fix all variables in the lower limit
%                             for j = Qdgind(1):Qdgind(nDG)
%                                 Xmu(ii,j) = Xmin(ii,j);
%                             end
% 
%                             for j = 1:nDG
%                                 k = cdg(j);
%                                 if (Xmax(ii,k) - Xmin(ii,k) > 0) && (qav >= Xmax(ii,k) - Xmin(ii,k))
%                                     Xmu(ii,k) = Xmax(ii,k);
%                                     qav = qav - (Xmax(ii,k) - Xmin(ii,k));
%                                 elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (qav > 0) && (qav < Xmax(ii,k) - Xmin(ii,k))
%                                     Xmu(ii,k) = Xmin(ii,k) + qav;
%                                     qav = 0;                        
%                                 elseif qav == 0
%                                     break;                        
%                                 end
%                             end
                        elseif seq(w) == 2  % Xdg
                            % Sequence of the cheapest generators
                            [~,cdg] = sort(DGc(ii,:));
                            cdg = cdg + 2*nDG;

                            % Number of operating generators to be distributed to the cheapest generators
                            xav = x_mu(ii,seq(w)) - Xrmin(ii,seq(w));

                            % Fix all variables in the lower limit
                            for j = Xdgind(1):Xdgind(nDG)
                                Xmu(ii,j) = Xmin(ii,j);
                            end

                            for j = 1:nDG
                                k = cdg(j);
                                if (Xmax(ii,k) - Xmin(ii,k) > 0) && (xav >= Xmax(ii,k) - Xmin(ii,k))
                                    Xmu(ii,k) = Xmax(ii,k);
                                    xav = xav - (Xmax(ii,k) - Xmin(ii,k));
                                elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (xav > 0) && (xav < Xmax(ii,k) - Xmin(ii,k))
                                    Xmu(ii,k) = Xmax(ii,k);
                                    xav = 0;                        
                                elseif xav == 0
                                    break;                        
                                end
                            end
                        elseif seq(w) == 3  % V2G
                            pav2g = (x_mu(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));
                            if pav2g > 1
                                pav2g = 1;
                            elseif pav2g < 0
                                pav2g = 0;
                            end                  
                            for j = V2Gind(1):V2Gind(nV2G)
                                Xmu(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pav2g;                    
                            end
                        elseif seq(w) == 4  % DR
                            pavdr = (x_mu(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));
                            if pavdr > 1
                                pavdr = 1;
                            elseif pavdr < 0
                                pavdr = 0;
                            end                  
                            for j = DRind(1):DRind(nDR)
                                Xmu(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pavdr;                    
                            end
                        elseif seq(w) == 5  % ESS
                            pavESS = (x_mu(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));                
                            if pavESS > 1
                                pavESS = 1;
                            elseif pavESS < 0
                                pavESS = 0;
                            end               

                            for j = ESSind(1):ESSind(nESS)
                                Xmu(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pavESS;                    
                            end
                        elseif seq(w) == 6  % Market
                            Xmu(ii,MKind(1)) = x_mu(ii,seq(w));
                        end

                        % Convert Xmu in xmu
                        xmu = zeros(1,D);
                        k = 1;
                        for i = 1:nper
                            for j = 1:nctrl
                                xmu(k) = Xmu(i,j);
                                k = k + 1;
                            end
                        end

                        for i = 1:D
                            if xmu(i) < xmin(i)
                                %%fprintf('LI %4d',i);
                                xmu(i) = xmin(i);
                                %pause
                            elseif xmu(i) > xmax(i)
                                %%fprintf('LS %4d',i);
                                xmu(i) = xmax(i);  
                                %pause
                            end
                        end                    

                        % Mu evaluation
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if iter < Niter
                            
                               [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xmu,caseStudyData, otherParameters,deParameters.Scenarios);
        
                                nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

                                for i=1:deParameters.I_NP   % Eval Initial population
                                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                                    omu(i)  = FIT(i); 
                                    fmu(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
                                end
       
                                xmu=xlb;
                            
                            %[FIT,omu,xrep,otherParameters] = fitnessFun_DER(xmu,caseStudyData,otherParameters);
                            %[fmu,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(omu,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
                            iter = iter + 1;
                            fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];

                            if FIT < FITBEST
                                FITBEST = FIT;
                                OBJBEST = omu;
                                fitMaxVector(iter) = FITBEST;
                                objMaxVector(:,iter)= OBJBEST(1,:);
                            else
                                fitMaxVector(iter) = FITBEST;
                                objMaxVector(:,iter)= OBJBEST(1,:);
                            end
                        else
                            break;
                            1004
                        end
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                        if fmu < fbest
                             %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,fmu,omu(2)-omu(1),fmu+omu(1)-omu(2),omu(2),omu(1));

                            fbest = fmu;
                            nEvals
                            104
                            objbest = omu;
                            xbest = xmu;

                            Xbest = Xmu;
                            Xrbest = x_mu;                        
                        end
                        %%
                    else
                        %%
                        b_fb = mu_fb;

                        mu_fb = lambda_fb;                    
                        lambda_fb = a_fb + (F_fb(n_fb-1-k_fb)/F_fb(n_fb+1-k_fb))*(b_fb - a_fb);

                        if k_fb == n_fb-2
                            break
                        end

                        x_lb(ii,seq(w)) = lambda_fb;

                        % Convert x_lb into Xlb
                        if seq(w) == 1      % Pdg
                            % Sequence of the cheapest generators
                            [~,cdg] = sort(DGc(ii,:));                
                            % Available active power to be distributed to the cheapest generators
                            pav = x_lb(ii,seq(w)) - Xrmin(ii,seq(w));

                            % Fix all variables in the lower limit
                            for j = Pdgind(1):Pdgind(nDG)
                                Xlb(ii,j) = Xmin(ii,j);
                            end
                            
                            for j = Xdgind(1):Xdgind(nDG)
                                Xlb(ii,j) = Xmin(ii,j);
                            end                          

                            for j = 1:nDG
                                k = cdg(j);
                                Xlb(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                                if (Xmax(ii,k) - Xmin(ii,k) > 0) && (pav >= Xmax(ii,k) - Xmin(ii,k))
                                    Xlb(ii,k) = Xmax(ii,k);
                                    Xlb(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                                    
                                    pav = pav - (Xmax(ii,k) - Xmin(ii,k));
                                elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (pav > 0) && (pav < Xmax(ii,k) - Xmin(ii,k))
                                    Xlb(ii,k) = Xmin(ii,k) + pav;
                                    Xlb(ii,k+2*nDG) = Xmax(ii,k+2*nDG);
                                    
                                    pav = 0;                        
                                elseif pav == 0
                                    break;                        
                                end
                            end 
%                         elseif seq(w) == 2  % Qdg
%                             % Sequence of the cheapest generators
%                             [~,cdg] = sort(DGc(ii,:));
%                             cdg = cdg + nDG;
%                             % Available reactive power to be distributed to the cheapest generators
%                             qav = x_lb(ii,seq(w)) - Xrmin(ii,seq(w));
% 
%                             % Fix all variables in the lower limit
%                             for j = Qdgind(1):Qdgind(nDG)
%                                 Xlb(ii,j) = Xmin(ii,j);
%                             end
% 
%                             for j = 1:nDG
%                                 k = cdg(j);
%                                 if (Xmax(ii,k) - Xmin(ii,k) > 0) && (qav >= Xmax(ii,k) - Xmin(ii,k))
%                                     Xlb(ii,k) = Xmax(ii,k);
%                                     qav = qav - (Xmax(ii,k) - Xmin(ii,k));
%                                 elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (qav > 0) && (qav < Xmax(ii,k) - Xmin(ii,k))
%                                     Xlb(ii,k) = Xmin(ii,k) + qav;
%                                     qav = 0;                        
%                                 elseif qav == 0
%                                     break;                        
%                                 end
%                             end
                        elseif seq(w) == 2  % Xdg
                            % Sequence of the cheapest generators
                            [~,cdg] = sort(DGc(ii,:));
                            cdg = cdg + 2*nDG;

                            % Number of operating generators to be distributed to the cheapest generators
                            xav = x_lb(ii,seq(w)) - Xrmin(ii,seq(w));

                            % Fix all variables in the lower limit
                            for j = Xdgind(1):Xdgind(nDG)
                                Xlb(ii,j) = Xmin(ii,j);
                            end

                            for j = 1:nDG
                                k = cdg(j);
                                if (Xmax(ii,k) - Xmin(ii,k) > 0) && (xav >= Xmax(ii,k) - Xmin(ii,k))
                                    Xlb(ii,k) = Xmax(ii,k);
                                    xav = xav - (Xmax(ii,k) - Xmin(ii,k));
                                elseif (Xmax(ii,k) - Xmin(ii,k) > 0) && (xav > 0) && (xav < Xmax(ii,k) - Xmin(ii,k))
                                    Xlb(ii,k) = Xmax(ii,k);
                                    xav = 0;                        
                                elseif xav == 0
                                    break;                        
                                end
                            end
                        elseif seq(w) == 3  % V2G
                            pav2g = (x_lb(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));
                            if pav2g > 1
                                pav2g = 1;
                            elseif pav2g < 0
                                pav2g = 0;
                            end                  
                            for j = V2Gind(1):V2Gind(nV2G)
                                Xlb(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pav2g;                    
                            end
                        elseif seq(w) == 4  % DR
                            pavdr = (x_lb(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));
                            if pavdr > 1
                                pavdr = 1;
                            elseif pavdr < 0
                                pavdr = 0;
                            end                  
                            for j = DRind(1):DRind(nDR)
                                Xlb(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pavdr;                    
                            end              
                        elseif seq(w) == 5  % ESS
                            pavESS = (x_lb(ii,seq(w)) - Xrmin(ii,seq(w)))/(Xrmax(ii,seq(w)) - Xrmin(ii,seq(w)));                
                            if pavESS > 1
                                pavESS = 1;
                            elseif pavESS < 0
                                pavESS = 0;
                            end               

                            for j = ESSind(1):ESSind(nESS)
                                Xlb(ii,j) = Xmin(ii,j) + (Xmax(ii,j) - Xmin(ii,j))*pavESS;                    
                            end
                        elseif seq(w) == 6  % Market
                            Xlb(ii,MKind(1)) = x_lb(ii,seq(w));                 
                        end

                        % Convert Xlb in xlb
                        xlb = zeros(1,D);
                        k = 1;
                        for i = 1:nper
                            for j = 1:nctrl
                                xlb(k) = Xlb(i,j);
                                k = k + 1;
                            end
                        end

                        for i = 1:D
                            if xlb(i) < xmin(i)
                                %%fprintf('LI %4d',i);
                                xlb(i) = xmin(i);
                                %pause
                            elseif xlb(i) > xmax(i)
                                %%fprintf('LS %4d',i);
                                xlb(i) = xmax(i);  
                                %pause
                            end
                        end 

                        % Lambda evaluation
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if iter < Niter
                                [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xlb,caseStudyData, otherParameters,deParameters.Scenarios);

                                nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);

                                for i=1:deParameters.I_NP   % Eval Initial population
                                    FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
                                    olb(i)  = FIT(i); 
                                    flb(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
                                end

                                xrep=xlb;
                            
                            %[FIT,olb,xrep,otherParameters] = fitnessFun_DER(xlb,caseStudyData,otherParameters);
                            %[flb,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(olb,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
                            iter = iter + 1;
                            fitMaxVector1(:,iter)=[mean(solFitness_M(1,:));mean(solPenalties_M(1,:))];

                            if FIT < FITBEST
                                FITBEST = FIT;
                                OBJBEST = olb;
                                fitMaxVector(iter) = FITBEST;
                                objMaxVector(:,iter)= OBJBEST(1,:);
                            else
                                fitMaxVector(iter) = FITBEST;
                                objMaxVector(:,iter)= OBJBEST(1,:);
                            end
                        else
                            break;
                            1005
                        end
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                        if flb < fbest
                            %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,flb,olb(2)-olb(1),flb+olb(1)-olb(2),olb(2),olb(1));

                            fbest = flb
                            nEvals
                            105
                            objbest = olb;
                            xbest = xlb;

                            Xbest = Xlb;
                            Xrbest = x_lb;                        
                        end
                        %%
                    end
                end
            end
        end
    end
end

% %% Local search in each variable
% 
% display('Local search in each variable')
% Niter=360
% nEvals
% 
% % Convert xbest into Xbest
% X = vec2mat(xbest,nctrl);
% 
% % Sequence for the CCM
% %seq = [Qdgind(1):Qdgind(nDG), DRind(1):MKind(nMK)];
% seq = [DRind(1):MKind(nMK)];
% nvXr = length(seq);
% 
% Xbest = X;
% 
% % Number of repetitions
% WW = 1;
% 
% for ww = 1:WW        
%     for ii = 1:nper
%         for w = 1:nvXr
% %             %% Turn the generator on
% %             if xtype(seq(w)) == 1
% %                 X(ii,seq(w)+2*nDG) = Xmax(ii,seq(w)+2*nDG);
% %             elseif xtype(seq(w)) == 2
% %                 X(ii,seq(w)+2*nDG) = Xmax(ii,seq(w)+nDG);
% %             end
%                 
%             %% Repair the best solution
% 
%             x = xbest;                
%             for i = 1:D
%                 if x(i) < xmin(i)
%                     x(i) = xmin(i);
%                 elseif x(i) > xmax(i)
%                     x(i) = xmax(i);  
%                 end
%                 
%                 if abs(x(i) - xmin(i)) < 10^-3
%                     x(i) = xmin(i);
%                 elseif abs(xmax(i) - x(i)) < 10^-3
%                     x(i) = xmax(i);
%                 end
%             end            
%             
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             if iter < Niter
%                     [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
% 
%                                 nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                                 for i=1:deParameters.I_NP   % Eval Initial population
%                                     FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                                     obj(i)  = FIT(i); 
%                                     frep(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                                 end
% 
%                                 xrep=x;
%                 
%                 %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
%                 %[frep,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                 iter = iter + 1;
% 
%                 if FIT < FITBEST
%                     FITBEST = FIT;
%                     OBJBEST = obj;
%                     fitMaxVector(iter) = FITBEST;
%                     objMaxVector(:,iter)= OBJBEST(1,:);
%                 else
%                     fitMaxVector(iter) = FITBEST;
%                     objMaxVector(:,iter)= OBJBEST(1,:);
%                 end
%             else
%                 break;
%             end
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             
%             if frep < fbest
%                 fbest = frep
%                 nEvals
%                 106
%                 objbest = obj;
%                 xbest = xrep;
%                 Xbest = vec2mat(xbest,nctrl);
%             end
%             
%             X = Xbest;            
%             %% Lambda and mu calculation
%             a_fb = Xmin(ii,seq(w));
%             b_fb = Xmax(ii,seq(w));
%             
%             flag = 0;
%             if a_fb == b_fb
%                 flag = 1;
%             end
%             
%             if flag == 0
%             
% %                 if (abs(b_fb - a_fb)/1000) < (10^-3)
%                     l_fb = abs(b_fb - a_fb)/10;
% %                 else
% %                     l_fb = 10^-3;
% %                 end
% 
%                 n_fb = 1; 
%                 while F_fb(n_fb) <= (b_fb - a_fb)/l_fb
%                     n_fb = n_fb + 1;                
%                 end
% 
%                 lambda_fb = a_fb + (F_fb(n_fb-1)/F_fb(n_fb+1))*(b_fb - a_fb);
%                 mu_fb = a_fb + (F_fb(n_fb)/F_fb(n_fb+1))*(b_fb - a_fb);
% 
%                 %%
%                 x_lb = X;
%                 x_lb(ii,seq(w)) = lambda_fb;            
% 
%                 % Convert x_lb in xlb
%                 xlb = zeros(1,D);
%                 k = 1;
%                 for i = 1:nper
%                     for j = 1:nctrl
%                         xlb(k) = x_lb(i,j);
%                         k = k + 1;
%                     end
%                 end
% 
%                 for i = 1:D
%                     if xlb(i) < xmin(i)
%                         %%fprintf('LI %4d',i);
%                         xlb(i) = xmin(i);
%                         %pause
%                     elseif xlb(i) > xmax(i)
%                         %%fprintf('LS %4d',i);
%                         xlb(i) = xmax(i);  
%                         %pause
%                     end
%                 end            
% 
%                 % Lambda evaluation
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 if iter < Niter
%                         [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xlb,caseStudyData, otherParameters,deParameters.Scenarios);
% 
%                                 nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                                 for i=1:deParameters.I_NP   % Eval Initial population
%                                     FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                                     olb(i)  = FIT(i); 
%                                     flb(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                                 end
% 
%                                 xrep=xlb;
%                            
%                     %[FIT,olb,xrep,otherParameters] = fitnessFun_DER(xlb,caseStudyData,otherParameters);
%                     %[flb,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(olb,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                     
%                     iter = iter + 1;
% 
%                     if FIT < FITBEST
%                         FITBEST = FIT;
%                         OBJBEST = olb;
%                         fitMaxVector(iter) = FITBEST;
%                         objMaxVector(:,iter)= OBJBEST(1,:);
%                     else
%                         fitMaxVector(iter) = FITBEST;
%                         objMaxVector(:,iter)= OBJBEST(1,:);
%                     end
%                 else
%                     break;
%                 end
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                 if flb < fbest
%                     %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,flb,olb(2)-olb(1),flb+olb(1)-olb(2),olb(2),olb(1));
% 
%                     fbest = flb
%                     nEvals
%                     107
%                     objbest = olb;
%                     xbest = xlb;
% 
%                     Xbest = x_lb;
%                 end
% 
%                 %%
%                 x_mu = X;
%                 x_mu(ii,seq(w)) = mu_fb;
% 
%                 % Convert x_mu in xmu
%                 xmu = zeros(1,D);
%                 k = 1;
%                 for i = 1:nper
%                     for j = 1:nctrl
%                         xmu(k) = x_mu(i,j);
%                         k = k + 1;
%                     end
%                 end
% 
%                 for i = 1:D
%                     if xmu(i) < xmin(i)
%                         %%fprintf('LI %4d',i);
%                         xmu(i) = xmin(i);
%                         %pause
%                     elseif xmu(i) > xmax(i)
%                         %%fprintf('LS %4d',i);
%                         xmu(i) = xmax(i);  
%                         %pause
%                     end
%                 end            
% 
%                 % Mu evaluation
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 if iter < Niter
%                         [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xmu,caseStudyData, otherParameters,deParameters.Scenarios);
% 
%                                 nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                                 for i=1:deParameters.I_NP   % Eval Initial population
%                                     FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                                     omu(i)  = FIT(i); 
%                                     fmu(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                                 end
% 
%                                 xrep=xmu;
%                            
%                     %[FIT,omu,xrep,otherParameters] = fitnessFun_DER(xmu,caseStudyData,otherParameters);
%                     %[fmu,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(omu,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                     iter = iter + 1;
% 
%                     if FIT < FITBEST
%                         FITBEST = FIT;
%                         OBJBEST = omu;
%                         fitMaxVector(iter) = FITBEST;
%                         objMaxVector(:,iter)= OBJBEST(1,:);
%                     else
%                         fitMaxVector(iter) = FITBEST;
%                         objMaxVector(:,iter)= OBJBEST(1,:);
%                     end
%                 else
%                     break;
%                 end
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                 if fmu < fbest
%                     %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,fmu,omu(2)-omu(1),fmu+omu(1)-omu(2),omu(2),omu(1));
% 
%                     fbest = fmu
%                     nEvals
%                     108
%                     objbest = omu;
%                     xbest = xmu;
% 
%                     Xbest = x_mu;             
%                 end
% 
%                 %%
%                 for k_fb = 1:n_fb-2
%                     %%
%                     if flb > fmu
%                         a_fb = lambda_fb;
% 
%                         lambda_fb = mu_fb;
%                         mu_fb = a_fb + (F_fb(n_fb-k_fb)/F_fb(n_fb+1-k_fb))*(b_fb - a_fb);
% 
%                         if k_fb == n_fb-2
%                             break
%                         end
% 
%                         x_mu(ii,seq(w)) = mu_fb;
% 
%                         % Convert x_mu in xmu
%                         xmu = zeros(1,D);
%                         k = 1;
%                         for i = 1:nper
%                             for j = 1:nctrl
%                                 xmu(k) = x_mu(i,j);
%                                 k = k + 1;
%                             end
%                         end
% 
%                         for i = 1:D
%                             if xmu(i) < xmin(i)
%                                 %%fprintf('LI %4d',i);
%                                 xmu(i) = xmin(i);
%                                 %pause
%                             elseif xmu(i) > xmax(i)
%                                 %%fprintf('LS %4d',i);
%                                 xmu(i) = xmax(i);  
%                                 %pause
%                             end
%                         end                    
% 
%                         % Mu evaluation
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         if iter < Niter
%                                 [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xmu,caseStudyData, otherParameters,deParameters.Scenarios);
% 
%                                 nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                                 for i=1:deParameters.I_NP   % Eval Initial population
%                                     FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                                     omu(i)  = FIT(i); 
%                                     fmu(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                                 end
% 
%                                 xrep=xmu;
%                            
%                             %[FIT,omu,xrep,otherParameters] = fitnessFun_DER(xmu,caseStudyData,otherParameters);
%                             %[fmu,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(omu,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                             iter = iter + 1;
% 
%                             if FIT < FITBEST
%                                 FITBEST = FIT;
%                                 OBJBEST = omu;
%                                 fitMaxVector(iter) = FITBEST;
%                                 objMaxVector(:,iter)= OBJBEST(1,:);
%                             else
%                                 fitMaxVector(iter) = FITBEST;
%                                 objMaxVector(:,iter)= OBJBEST(1,:);
%                             end
%                         else
%                             break;
%                         end
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                         if fmu < fbest
%                              %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,fmu,omu(2)-omu(1),fmu+omu(1)-omu(2),omu(2),omu(1));
% 
%                             fbest = fmu
%                             nEvals
%                             109
%                             objbest = omu;
%                             xbest = xmu;
% 
%                             Xbest = x_mu;                    
%                         end
%                         %%
%                     else
%                         %%
%                         b_fb = mu_fb;
% 
%                         mu_fb = lambda_fb;                    
%                         lambda_fb = a_fb + (F_fb(n_fb-1-k_fb)/F_fb(n_fb+1-k_fb))*(b_fb - a_fb);
% 
%                         if k_fb == n_fb-2
%                             break
%                         end
% 
%                         x_lb(ii,seq(w)) = lambda_fb;
% 
%                         % Convert x_lb in xlb
%                         xlb = zeros(1,D);
%                         k = 1;
%                         for i = 1:nper
%                             for j = 1:nctrl
%                                 xlb(k) = x_lb(i,j);
%                                 k = k + 1;
%                             end
%                         end
% 
%                         for i = 1:D
%                             if xlb(i) < xmin(i)
%                                 %%fprintf('LI %4d',i);
%                                 xlb(i) = xmin(i);
%                                 %pause
%                             elseif xlb(i) > xmax(i)
%                                 %%fprintf('LS %4d',i);
%                                 xlb(i) = xmax(i);  
%                                 %pause
%                             end
%                         end 
% 
%                         % Lambda evaluation
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         if iter < Niter
%                                 [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xlb,caseStudyData, otherParameters,deParameters.Scenarios);
% 
%                                 nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                                 for i=1:deParameters.I_NP   % Eval Initial population
%                                     FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                                     olb(i)  = FIT(i); 
%                                     flb(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                                 end
% 
%                                 xrep=xlb;
%                            
%                             %[FIT,olb,xrep,otherParameters] = fitnessFun_DER(xlb,caseStudyData,otherParameters);
%                             %[flb,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(olb,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                             iter = iter + 1;
% 
%                             if FIT < FITBEST
%                                 FITBEST = FIT;
%                                 OBJBEST = olb;
%                                 fitMaxVector(iter) = FITBEST;
%                                 objMaxVector(:,iter)= OBJBEST(1,:);
%                             else
%                                 fitMaxVector(iter) = FITBEST;
%                                 objMaxVector(:,iter)= OBJBEST(1,:);
%                             end
%                         else
%                             break;
%                         end
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                         if flb < fbest
%                             %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,flb,olb(2)-olb(1),flb+olb(1)-olb(2),olb(2),olb(1));
% 
%                             fbest = flb
%                             nEvals
%                             110
%                             objbest = olb;
%                             xbest = xlb;
% 
%                             Xbest = x_lb;                       
%                         end
%                         %%
%                     end
%                 end
%             end
%         end
%     end
% end
% 
%     nEvals=iter*deParameters.Scenarios
%     fbest
%     
%     x=xbest; 

% %% Local search V2G
% display('V2G')
% Niter=361
% nEvals
% 
% nb = caseStudyData(1).parameterData.numBus;
% v2gloc = caseStudyData(1).v2gData.v2gBusNo;
% 
% xv2g = zeros(nper,nb);  % Percentage of v2g connected to a bus
% xTv2g = zeros(nper,nb); % Total v2g connected to a bus
% xmv2g = zeros(nper,nb); % Min v2g connected to a bus
% xMv2g = zeros(nper,nb); % Max v2g connected to a bus
% 
% for t = 1:nper
%     for i = 1:nb
%         k = 1;
%         for j = V2Gind(1):V2Gind(nV2G)
%             if i == v2gloc(k)
%                 xTv2g(t,i) = xTv2g(t,i) + Xbest(t,j);
%                 xmv2g(t,i) = xmv2g(t,i) + Xmin(t,j);
%                 xMv2g(t,i) = xMv2g(t,i) + Xmax(t,j);
%             end
%             k = k + 1;
%         end
%     end
% end
% 
% for i = 1:nper
%     for j = 1:nb
%         if (xMv2g(i,j) - xmv2g(i,j)) > 0
%             xv2g(i,j) = (xTv2g(i,j) - xmv2g(i,j))/(xMv2g(i,j) - xmv2g(i,j));
%         else
%             xv2g(i,j) = 0;
%         end
%     end
% end
% 
% seq = randperm(nb);
% nvXr = length(seq);
% 
% xv2gbest = xv2g;
% 
% % Number of repetitions
% WW = 1;
% 
% for ww = 1:WW
%     for ii = 1:nper
%         for w = 1:nvXr         
%             %% Repair the best solution
% 
%             x = xbest;
%             for i = 1:D
%                 if x(i) < xmin(i)
%                     x(i) = xmin(i);
%                 elseif x(i) > xmax(i)
%                     x(i) = xmax(i);  
%                 end
%                 
%                 if abs(x(i) - xmin(i)) < 10^-3
%                     x(i) = xmin(i);
%                 elseif abs(xmax(i) - x(i)) < 10^-3
%                     x(i) = xmax(i);
%                 end
%             end
%             
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             if iter < Niter
%                 [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,x,caseStudyData, otherParameters,deParameters.Scenarios);
%         
%                     nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                     for i=1:deParameters.I_NP   % Eval Initial population
%                         FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                         obj(i)  = FIT(i); 
%                         frep(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                     end
%        
%                     xrep=x;
%                 %[FIT,obj,xrep,otherParameters] = fitnessFun_DER(x,caseStudyData,otherParameters);
%                 %[frep,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(obj,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                 iter = iter + 1;
% 
%                 if FIT < FITBEST
%                     FITBEST = FIT;
%                     OBJBEST = obj;
%                     fitMaxVector(iter) = FITBEST;
%                     objMaxVector(:,iter)= OBJBEST(1,:);
%                 else
%                     fitMaxVector(iter) = FITBEST;
%                     objMaxVector(:,iter)= OBJBEST(1,:);
%                 end
%             else
%                 break;
%             end
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             
%             if frep < fbest
%                 fbest = frep
%                 nEvals
%                 106
%                 objbest = obj;
%                 
%                 xbest = xrep;
%                 Xbest = vec2mat(xbest,nctrl);
%                 
%                 xTv2g = zeros(nper,nb);
%                 for nn = 1:nper
%                     for qq = 1:nb
%                         kk = 1;
%                         for rr = V2Gind(1):V2Gind(nV2G)
%                             if qq == v2gloc(kk)
%                                 xTv2g(nn,qq) = xTv2g(nn,qq) + Xbest(nn,rr);
%                             end
%                             kk = kk + 1;
%                         end
%                     end
%                 end
% 
%                 for qq = 1:nper
%                     for rr = 1:nb
%                         if (xMv2g(qq,rr) - xmv2g(qq,rr)) > 0
%                             xv2g(qq,rr) = (xTv2g(qq,rr) - xmv2g(qq,rr))/(xMv2g(qq,rr) - xmv2g(qq,rr));
%                         else
%                             xv2g(qq,rr) = 0;
%                         end
%                     end
%                 end
%                 xv2gbest = xv2g;                
%             end
%             
%             X = Xbest;
%             xv2g = xv2gbest;
%             %% Lambda and mu calculation
%             a_fb = xmv2g(ii,seq(w));
%             b_fb = xMv2g(ii,seq(w));
%             
%             flag = 0;
%             if a_fb == b_fb
%                 flag = 1;
%             end
%             
%             if flag == 0
%             
% %                 if (abs(b_fb - a_fb)/1000) < (10^-3)
%                     l_fb = abs(b_fb - a_fb)/10;
% %                 else
% %                     l_fb = 10^-3;
% %                 end
% 
%                 n_fb = 1; 
%                 while F_fb(n_fb) <= (b_fb - a_fb)/l_fb
%                     n_fb = n_fb + 1;                
%                 end
% 
%                 lambda_fb = a_fb + (F_fb(n_fb-1)/F_fb(n_fb+1))*(b_fb - a_fb);
%                 mu_fb = a_fb + (F_fb(n_fb)/F_fb(n_fb+1))*(b_fb - a_fb);
% 
%                 %%
%                 x_lb = xv2g;
%                 x_lb(ii,seq(w)) = lambda_fb;            
% 
%                 % Convert x_lb in xlb
%                 alpha = (lambda_fb - xmv2g(ii,seq(w)))/(xMv2g(ii,seq(w)) - xmv2g(ii,seq(w)));
%                 kk = 1;
%                 for rr = V2Gind(1):V2Gind(nV2G)
%                     if seq(w) == v2gloc(kk)
%                         X(ii,rr) = Xmin(ii,rr) + alpha*(Xmax(ii,rr) - Xmin(ii,rr));
%                     end
%                     kk = kk + 1;
%                 end
%                 
%                 xlb = zeros(1,D);
%                 k = 1;
%                 for i = 1:nper
%                     for j = 1:nctrl
%                         xlb(k) = X(i,j);
%                         k = k + 1;
%                     end
%                 end
% 
%                 for i = 1:D
%                     if xlb(i) < xmin(i)
%                         %%fprintf('LI %4d',i);
%                         xlb(i) = xmin(i);
%                         %pause
%                     elseif xlb(i) > xmax(i)
%                         %%fprintf('LS %4d',i);
%                         xlb(i) = xmax(i);  
%                         %pause
%                     end
%                 end            
% 
%                 % Lambda evaluation
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 if iter < Niter
%                     [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xlb,caseStudyData, otherParameters,deParameters.Scenarios);
%         
%                     nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                     for i=1:deParameters.I_NP   % Eval Initial population
%                         FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                         olb(i)  = FIT(i); 
%                         flb(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                     end
%        
%                     xrep=xlb;
%                     %[FIT,olb,xrep,otherParameters] = fitnessFun_DER(xlb,caseStudyData,otherParameters);
%                     %[flb,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(olb,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                     iter = iter + 1;
% 
%                     if FIT < FITBEST
%                         FITBEST = FIT;
%                         OBJBEST = olb;
%                         fitMaxVector(iter) = FITBEST;
%                         objMaxVector(:,iter)= OBJBEST(1,:);
%                     else
%                         fitMaxVector(iter) = FITBEST;
%                         objMaxVector(:,iter)= OBJBEST(1,:);
%                     end
%                 else
%                     break;
%                 end
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                 if flb < fbest
%                     %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,flb,olb(2)-olb(1),flb+olb(1)-olb(2),olb(2),olb(1));
% 
%                     fbest = flb
%                     nEvals
%                     107
%                     objbest = olb;
%                     xbest = xlb;
% 
%                     Xbest = vec2mat(xbest,nctrl);
%                     xv2gbest = x_lb;
%                 end
% 
%                 %%
%                 x_mu = xv2g;
%                 x_mu(ii,seq(w)) = mu_fb;
% 
%                 % Convert x_mu in xmu
%                 alpha = (mu_fb - xmv2g(ii,seq(w)))/(xMv2g(ii,seq(w)) - xmv2g(ii,seq(w)));
%                 kk = 1;
%                 for rr = V2Gind(1):V2Gind(nV2G)
%                     if seq(w) == v2gloc(kk)
%                         X(ii,rr) = Xmin(ii,rr) + alpha*(Xmax(ii,rr) - Xmin(ii,rr));
%                     end
%                     kk = kk + 1;
%                 end
%                 
%                 xmu = zeros(1,D);
%                 k = 1;
%                 for i = 1:nper
%                     for j = 1:nctrl
%                         xmu(k) = X(i,j);
%                         k = k + 1;
%                     end
%                 end
% 
%                 for i = 1:D
%                     if xmu(i) < xmin(i)
%                         %%fprintf('LI %4d',i);
%                         xmu(i) = xmin(i);
%                         %pause
%                     elseif xmu(i) > xmax(i)
%                         %%fprintf('LS %4d',i);
%                         xmu(i) = xmax(i);  
%                         %pause
%                     end
%                 end            
% 
%                 % Mu evaluation
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 if iter < Niter
%                     [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xmu,caseStudyData, otherParameters,deParameters.Scenarios);
%         
%                     nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                     for i=1:deParameters.I_NP   % Eval Initial population
%                         FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                         omu(i)  = FIT(i); 
%                         fmu(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                     end
%        
%                     xrep=xmu;
%                     %[FIT,omu,xrep,otherParameters] = fitnessFun_DER(xmu,caseStudyData,otherParameters);
%                     %[fmu,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(omu,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                     iter = iter + 1;
% 
%                     if FIT < FITBEST
%                         FITBEST = FIT;
%                         OBJBEST = omu;
%                         fitMaxVector(iter) = FITBEST;
%                         objMaxVector(:,iter)= OBJBEST(1,:);
%                     else
%                         fitMaxVector(iter) = FITBEST;
%                         objMaxVector(:,iter)= OBJBEST(1,:);
%                     end
%                 else
%                     break;
%                 end
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                 if fmu < fbest
%                     %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,fmu,omu(2)-omu(1),fmu+omu(1)-omu(2),omu(2),omu(1));
% 
%                     fbest = fmu
%                     nEvals
%                     108
%                     objbest = omu;
%                     xbest = xmu;
% 
%                     Xbest = vec2mat(xbest,nctrl);
%                     xv2gbest = x_mu;             
%                 end
% 
%                 %%
%                 for k_fb = 1:n_fb-2
%                     %%
%                     if flb > fmu
%                         a_fb = lambda_fb;
% 
%                         lambda_fb = mu_fb;
%                         mu_fb = a_fb + (F_fb(n_fb-k_fb)/F_fb(n_fb+1-k_fb))*(b_fb - a_fb);
% 
%                         if k_fb == n_fb-2
%                             break
%                         end
% 
%                         x_mu(ii,seq(w)) = mu_fb;
% 
%                         % Convert x_mu in xmu
%                         alpha = (mu_fb - xmv2g(ii,seq(w)))/(xMv2g(ii,seq(w)) - xmv2g(ii,seq(w)));
%                         kk = 1;
%                         for rr = V2Gind(1):V2Gind(nV2G)
%                             if seq(w) == v2gloc(kk)
%                                 X(ii,rr) = Xmin(ii,rr) + alpha*(Xmax(ii,rr) - Xmin(ii,rr));
%                             end
%                             kk = kk + 1;
%                         end
% 
%                         xmu = zeros(1,D);
%                         k = 1;
%                         for i = 1:nper
%                             for j = 1:nctrl
%                                 xmu(k) = X(i,j);
%                                 k = k + 1;
%                             end
%                         end
% 
%                         for i = 1:D
%                             if xmu(i) < xmin(i)
%                                 %%fprintf('LI %4d',i);
%                                 xmu(i) = xmin(i);
%                                 %pause
%                             elseif xmu(i) > xmax(i)
%                                 %%fprintf('LS %4d',i);
%                                 xmu(i) = xmax(i);  
%                                 %pause
%                             end
%                         end
%                 
%                         % Mu evaluation
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         if iter < Niter
%                             [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xmu,caseStudyData, otherParameters,deParameters.Scenarios);
% 
%                             nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                             for i=1:deParameters.I_NP   % Eval Initial population
%                                 FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                                 omu(i)  = FIT(i); 
%                                 fmu(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                             end
% 
%                             xrep=xmu;
%                             %[FIT,omu,xrep,otherParameters] = fitnessFun_DER(xmu,caseStudyData,otherParameters);
%                             %[fmu,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(omu,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                             iter = iter + 1;
% 
%                             if FIT < FITBEST
%                                 FITBEST = FIT;
%                                 OBJBEST = omu;
%                                 fitMaxVector(iter) = FITBEST;
%                                 objMaxVector(:,iter)= OBJBEST(1,:);
%                             else
%                                 fitMaxVector(iter) = FITBEST;
%                                 objMaxVector(:,iter)= OBJBEST(1,:);
%                             end
%                         else
%                             break;
%                         end
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                         if fmu < fbest
%                              %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,fmu,omu(2)-omu(1),fmu+omu(1)-omu(2),omu(2),omu(1));
% 
%                             fbest = fmu
%                             nEvals
%                             109
%                             objbest = omu;
%                             xbest = xmu;
% 
%                             Xbest = vec2mat(xbest,nctrl);
%                             xv2gbest = x_mu;                     
%                         end
%                         %%
%                     else
%                         %%
%                         b_fb = mu_fb;
% 
%                         mu_fb = lambda_fb;                    
%                         lambda_fb = a_fb + (F_fb(n_fb-1-k_fb)/F_fb(n_fb+1-k_fb))*(b_fb - a_fb);
% 
%                         if k_fb == n_fb-2
%                             break
%                         end
% 
%                         x_lb(ii,seq(w)) = lambda_fb;
% 
%                         % Convert x_lb in xlb
%                         alpha = (lambda_fb - xmv2g(ii,seq(w)))/(xMv2g(ii,seq(w)) - xmv2g(ii,seq(w)));
%                         kk = 1;
%                         for rr = V2Gind(1):V2Gind(nV2G)
%                             if seq(w) == v2gloc(kk)
%                                 X(ii,rr) = Xmin(ii,rr) + alpha*(Xmax(ii,rr) - Xmin(ii,rr));
%                             end
%                             kk = kk + 1;
%                         end
% 
%                         xlb = zeros(1,D);
%                         k = 1;
%                         for i = 1:nper
%                             for j = 1:nctrl
%                                 xlb(k) = X(i,j);
%                                 k = k + 1;
%                             end
%                         end
% 
%                         for i = 1:D
%                             if xlb(i) < xmin(i)
%                                 %%fprintf('LI %4d',i);
%                                 xlb(i) = xmin(i);
%                                 %pause
%                             elseif xlb(i) > xmax(i)
%                                 %%fprintf('LS %4d',i);
%                                 xlb(i) = xmax(i);  
%                                 %pause
%                             end
%                         end            
% 
%                         % Lambda evaluation
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         if iter < Niter
%                             [solFitness_M, solPenalties_M,Struct_Eval]=feval(fnc,xlb,caseStudyData, otherParameters,deParameters.Scenarios);
% 
%                                 nEvals  = nEvals+(deParameters.I_NP*deParameters.Scenarios);
% 
%                                 for i=1:deParameters.I_NP   % Eval Initial population
%                                     FIT(i)  = mean(solFitness_M(i,:))+mean(solPenalties_M(i,:));  
%                                     olb(i)  = FIT(i); 
%                                     flb(i)  =  mean(solFitness_M(i,:))+std(solFitness_M(i,:))+mean(solPenalties_M(i,:))+std(solPenalties_M(i,:));
%                                 end
% 
%                                 xrep=xmu;
%                             %[FIT,olb,xrep,otherParameters] = fitnessFun_DER(xlb,caseStudyData,otherParameters);
%                             %[flb,~,penSlack,penSlines,penVmin,penVmax] = constraint_handling(olb,xrep,Xmax,Xmin,penalty,caseStudyData,otherParameters);
%                             iter = iter + 1;
% 
%                             if FIT < FITBEST
%                                 FITBEST = FIT;
%                                 OBJBEST = olb;
%                                 fitMaxVector(iter) = FITBEST;
%                                 objMaxVector(:,iter)= OBJBEST(1,:);
%                             else
%                                 fitMaxVector(iter) = FITBEST;
%                                 objMaxVector(:,iter)= OBJBEST(1,:);
%                             end
%                         else
%                             break;
%                         end
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%                         if flb < fbest
%                             %fprintf('%2d, %2d, ITER: %5d, FITNESS: %10.4f, OBJECTIVE: %10.4f, INF: %10.4f, OC: %10.4f, IN: %10.4f; \n',ii,seq(w),iter,flb,olb(2)-olb(1),flb+olb(1)-olb(2),olb(2),olb(1));
% 
%                             fbest = flb
%                             nEvals
%                             110
%                             objbest = olb;
%                             xbest = xlb;
% 
%                             Xbest = vec2mat(xbest,nctrl);
%                             xv2gbest = x_lb;                        
%                         end
%                         %%
%                     end
%                 end
%             end
%         end
%     end
% end

    nEvals=iter*deParameters.Scenarios
    fbest
    
    x=xbest; 


%%
    display('DEEPSO')
    DEparametersDEEPSO %Function defined by the participant
    No_solutions=deParameters.I_NP;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Set other parameters
     otherParameters =setOtherParameters(caseStudyData,No_solutions);

    [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    DEEPSO_RE(deParameters,caseStudyData,otherParameters,xmin,xmax,nEvals,xbest);

end