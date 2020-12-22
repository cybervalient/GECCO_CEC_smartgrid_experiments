%% Proposed ALGORITHM: Gauss Mapped Variable Neighbourhood Particle Swarm Optimization (GM_VNPSO)
%% Developers: 
% Kartik S. Pandya, Professor, Dept. of Electrical Engineering, CSPIT,
% CHARUSAT, Gujarat, INDIA
% Jigar Sarda, Asst. Professor, Dept. of Electrical Engineering, CSPIT,
% CHARUSAT, Gujarat, INDIA
% Margi Shah, Independent Researcher, Gujarat, INDIA


%% References:
%The codes of VNS is available on http://sites.ieee.org/psace-mho/2017-smart-grid-operation-problems-competition-panel/.
%The codes of EPSO is available on http://www.gecad.isep.ipp.pt/WCCI2018-SG-COMPETITION/
%The Codes of VNS were modified by Sergio Rivera, srriverar@unal.edu.co,professor at UN.
% The Codes of EPSO were developed by Phillipe Vilaça Gomes
% The codes have been modified by the developers to propose GM_VNPSO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Fit_and_p,FVr_bestmemit, fitMaxVector, Best_otherInfo] = ...
    EPSOu(epsouParameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit,No_eval_Scenarios,xbest,nEvals)



I_NP         = epsouParameters.I_NP;
I_D          = numel(up_habitat_limit); %Number of variables or dimension
I_itermax    = epsouParameters.stop;
replic = epsouParameters.r;

wors = round(epsouParameters.selecWorse*I_NP/100);

%-----Initialize population and some arrays-------------------------------

minPositionsMatrix=repmat(low_habitat_limit,I_NP,1);
maxPositionsMatrix=repmat(up_habitat_limit,I_NP,1);

populacao = unifrnd(minPositionsMatrix,maxPositionsMatrix,I_NP,I_D);
populacao(1,:)=xbest;
populacao(2,:)=low_habitat_limit;



%-----Evaluation of the first population and parameters--------------------

[solFitness_M,solPenalties_M, Struct_Eval] = ...
    fitnessFun_DER_WCCI(populacao,caseStudyData,otherParameters,No_eval_Scenarios);

nEvals  = nEvals+(epsouParameters.I_NP*No_eval_Scenarios);

xx=mean(solFitness_M');
yy=std(solFitness_M');
MedDesv = xx+yy;
MedDesv=MedDesv';

PenDesv = mean(solPenalties_M')+std(solPenalties_M');
PenDesv = PenDesv';

[a1,b1] = sort (MedDesv);
I_best_index = b1(1);
pbest = populacao;
gbest = populacao(b1(1),:);
worstbest = populacao(b1(2),:);

Vpbest = MedDesv;
Vgbest = MedDesv(b1(1));
Vgbestpen = PenDesv(b1(1));

iteration = 1;
GVEST(iteration,1) = Vgbest;

fitMaxVector(:,1)=[mean(solFitness_M(I_best_index,:));mean(solPenalties_M(I_best_index,:))]; %

%----------------------EPSO Iterative Process------------------------------

wi1 = rand(I_NP,replic);
wi2 = rand(I_NP,replic);
wi3 = rand(I_NP,replic);
wi4 = rand();
wi5 = rand(I_NP,replic);

Vel11 = rand(I_D,I_NP);

while iteration<I_itermax;  %nEvals <40000; %I_itermax          %%%%%%
    
   iteration = iteration + 1;
    
    %% Replication Block
    for rep=1:replic
        POP{rep}=populacao;
    end
    
   Kbest1 = gbest + round(0.2*wi4-1);
        
    %% Recombination Block
    for rep=1:replic
        for i=1:I_NP
            
   %%%%%%% Apply normal distribution for each particle to enhance global search.
             
   beta=2; %scalar
   sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);%scalar
   u=randn(1,I_D)*sigma;
   v=randn(1,I_D);
   step=u./abs(v).^(1/beta); 
   Normal_constant=(mean(up_habitat_limit-low_habitat_limit))/100;
   stepsize= Normal_constant*step.*(populacao(i,:)-gbest);
   normalstep=stepsize;
   
   
   
         %% worst step levy    
         
be=3/2; %scalar
sima=(gamma(1+be)*sin(pi*be/2)/(gamma((1+be)/2)*be*2^((be-1)/2)))^(1/be);%scalar
p=randn(1,I_D)*sima;
q=randn(1,I_D);
ste=p./abs(q).^(1/be); 
Levy_constant=(mean(up_habitat_limit-low_habitat_limit))/100;
stesize= Levy_constant*ste.*(populacao(i,:)-worstbest);
worststep=stesize;
Vel11(:,i)=Vel11(:,i).*wi1(i,1)+wi2(i,1).*(pbest(i,:)-POP{1,rep}(i,:))'+wi3(i,1).*(Kbest1* (1+wi5(i,1 ))-POP{1,rep}(i,:))'+ normalstep'-worststep';


POP{1,rep}(i,:) = (POP{1,rep}(i,:)+2.6*Vel11(:,i)');
          
% Verificação do limite superior
            vetAn = POP{1,rep}(i,:);
            vetAn(vetAn>up_habitat_limit)=up_habitat_limit(vetAn>up_habitat_limit);
            POP{1,rep}(i,:)=vetAn;
            POP{1,rep}(i,8:14)=round(POP{1,rep}(i,8:14));
 % Verificação do limite inferior
            vetAn = POP{1,rep}(i,:);
            vetAn(vetAn<low_habitat_limit)=low_habitat_limit(vetAn<low_habitat_limit);
            POP{1,rep}(i,:)=vetAn;
            POP{1,rep}(i,8:14)=round(POP{1,rep}(i,8:14));
        end
    end
    
    %% Evaluation block
    for rep=1:replic
        [solFitness_MPOP{rep},solPenalties_MPOP{rep}, Struct_Eval_MPOP{rep}] = ...
        fitnessFun_DER_WCCI(POP{1,rep},caseStudyData,otherParameters,No_eval_Scenarios);
    end
        
    nEvals  = nEvals+(epsouParameters.I_NP*No_eval_Scenarios*replic);
    %% Selection Block
    
    for rep=1:replic
        MedDesvPOP{rep} = mean(solFitness_MPOP{1,rep}')+std(solFitness_MPOP{1,rep}');
        MedDesvPOP{rep}=MedDesvPOP{rep}';
        PenDesv = mean(solPenalties_MPOP{1,rep}')+std(solPenalties_MPOP{1,rep}');
        pbest(MedDesvPOP{rep}<Vpbest,:)= POP{1,rep}(MedDesvPOP{rep}<Vpbest,:);
        Vpbest(MedDesvPOP{rep}<Vpbest)=MedDesvPOP{rep}(MedDesvPOP{rep}<Vpbest);
        [a1,b1] = sort (MedDesvPOP{rep});
        VbestP=a1(1);
           [S_val, worstS]=max(solFitness_MPOP{rep},[],2);
            Best_otherInfo.idBestParticle = I_best_index;
            Best_otherInfo.genCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.genCosts(I_best_index,:);
            Best_otherInfo.loadDRcostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.loadDRcosts(I_best_index,:);
            Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gChargeCosts(I_best_index,:);
            Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(worstS(I_best_index)).otherParameters.v2gDischargeCosts(I_best_index,:);
            Best_otherInfo.storageChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageChargeCosts(I_best_index,:);
            Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageDischargeCosts(I_best_index,:);
            Best_otherInfo.stBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.stBalance(I_best_index,:,:);
            Best_otherInfo.v2gBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gBalance(I_best_index,:,:);
            Best_otherInfo.penSlackBusFinal = Struct_Eval(worstS(I_best_index)).otherParameters.penSlackBus(I_best_index,:);
            
        if VbestP<Vgbest
            Vgbest=VbestP;
            gbest=POP{1,rep}(b1(1),:);
            Vgbestpen=PenDesv(b1(1));
         
            
            [S_val, worstS]=max(solFitness_MPOP{rep},[],2);
            solFitness_M = solFitness_MPOP{rep};
            solPenalties_M = solPenalties_MPOP{rep};
            Struct_Eval = Struct_Eval_MPOP{rep};
            [~,I_best_index] = min(S_val);
            Best_otherInfo.idBestParticle = I_best_index;
            Best_otherInfo.genCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.genCosts(I_best_index,:);
            Best_otherInfo.loadDRcostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.loadDRcosts(I_best_index,:);
            Best_otherInfo.v2gChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gChargeCosts(I_best_index,:);
            Best_otherInfo.v2gDischargeCostsFinal =Struct_Eval(worstS(I_best_index)).otherParameters.v2gDischargeCosts(I_best_index,:);
            Best_otherInfo.storageChargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageChargeCosts(I_best_index,:);
            Best_otherInfo.storageDischargeCostsFinal = Struct_Eval(worstS(I_best_index)).otherParameters.storageDischargeCosts(I_best_index,:);
            Best_otherInfo.stBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.stBalance(I_best_index,:,:);
            Best_otherInfo.v2gBalanceFinal = Struct_Eval(worstS(I_best_index)).otherParameters.v2gBalance(I_best_index,:,:);
            Best_otherInfo.penSlackBusFinal = Struct_Eval(worstS(I_best_index)).otherParameters.penSlackBus(I_best_index,:);
        end
    end
    
    
    
    MatrizMedDesvPop = [MedDesvPOP{1} MedDesvPOP{2} MedDesvPOP{3}];
    
    [valoresM,indiceM] = sort(MatrizMedDesvPop');
    
    Melhores = [valoresM(1,:);indiceM(1,:)]';
    Piores = [valoresM(replic,:);indiceM(replic,:)]';
    Setwors = randi([1,I_NP],1,wors);
    
    Melhores(Setwors,:)=Piores(Setwors,:);
    newpop = populacao;
    newfitpop = MedDesv;
    
    for indix=1:I_NP
        newpop(indix,:)=POP{1,Melhores(indix,2)}(indix,:);
        newfitpop(indix)=MedDesvPOP{1,Melhores(indix,2)}(indix);
    end
    
    %% Important Results
    
    
    GVEST(iteration,1) = Vgbest;
    PENBES(iteration,1) = Vgbestpen;
    
    %% Store variables
    sol = gbest;
    
    %% New population
    populacao = newpop;
    MedDesv = newfitpop;
    
    
end

          
%% Final Results
fitMaxVector=[GVEST PENBES]';
Fit_and_p = fitMaxVector(:,iteration)';
sol = gbest;
FVr_bestmemit=sol;
p1=sum(Best_otherInfo.penSlackBusFinal);
end




