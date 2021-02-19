#############################################################
#         This code was created by: Yoan Martínez López     #
#         email: yoan.martinez@reduc.edu.cu                 #
#                cybervalient@gmail.com                     #  
#                         2021                              #
#############################################################
source(file.choose()) #Load to tests.R
source(file.choose()) #Load to post_hoc.R
library(FSA)
Wilcoxontest <-function(x,y){
  wilcoxonSignedTest(x,y)
}

archive<-read.table(file.choose(), header = TRUE, sep =",")
#CUMDANCauchy vs chaotic_deepso
Wilcoxontest(archive$CUMDANCauchy,archive$RDG3_DEEPSO)
#CUMDANCauchy vs DESS 
Wilcoxontest(archive$CUMDANCauchy, archive$CE_CMAES)
#CUMDANCauchy vs EPSO
Wilcoxontest( archive$CUMDANCauchy, archive$HFEABC)
#CUMDANCauchy vs evdeepso
Wilcoxontest( archive$CUMDANCauchy, archive$DE_TLBO)
#CUMDANCauchy vs Firefly
Wilcoxontest( archive$CUMDANCauchy, archive$PSO_GBP)
#CUMDANCauchy vs Guide_DE
Wilcoxontest( archive$CUMDANCauchy, archive$GASAPSO)
#CUMDANCauchy vs UPSO
Wilcoxontest( archive$CUMDANCauchy, archive$AJSO)
#CUMDANCauchy vs GMVNPSO
Wilcoxontest( archive$CUMDANCauchy, archive$CUMDANCauchy)
#CUMDANCauchy vs VNSDEEPSO
Wilcoxontest( archive$CUMDANCauchy, archive$EHL_PS_VNSO)
#CUMDANCauchy vs PSO_GBP
Wilcoxontest( archive$CUMDANCauchy, archive$Ensembled_method_of_CBBO_Cauchy_and_DEEPSO)
Wilcoxontest( archive$CUMDANCauchy, archive$Ensembled_method_of_CBBO_Cauchy.and.DEEPSO)
#CUMDANCauchy vs HL_PS_VNSO
Wilcoxontest( archive$CUMDANCauchy, archive$DEEDA)

