function [fitness,nEvals]=fobj(X1,NuRun,fnc,caseStudyData,otherParameters,deParameters,nEvals)
            fit_superorganism=0;
            for gh=1:NuRun
                [fit_superorganism1, ~]=feval(fnc,X1,caseStudyData, otherParameters);
                fit_superorganism=fit_superorganism+fit_superorganism1;
            end
            fitness=fit_superorganism/NuRun;

            nEvals=nEvals+(NuRun*deParameters.I_NP*deParameters.Scenarios);
end      