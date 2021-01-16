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
Wilcoxontest(archive$CUMDANCauchy,archive$chaotic_deepso)
#CUMDANCauchy vs DESS 
Wilcoxontest(archive$CUMDANCauchy, archive$DESS)
#CUMDANCauchy vs EPSO
Wilcoxontest( archive$CUMDANCauchy, archive$EPSO)
#CUMDANCauchy vs evdeepso
Wilcoxontest( archive$CUMDANCauchy, archive$evdeepso)
#CUMDANCauchy vs Firefly
Wilcoxontest( archive$CUMDANCauchy, archive$Firefly)
#CUMDANCauchy vs Guide_DE
Wilcoxontest( archive$CUMDANCauchy, archive$Guide_DE)
#CUMDANCauchy vs UPSO
Wilcoxontest( archive$CUMDANCauchy, archive$UPSO)
#CUMDANCauchy vs GMVNPSO
Wilcoxontest( archive$CUMDANCauchy, archive$GMVNPSO)
#CUMDANCauchy vs VNSDEEPSO
Wilcoxontest( archive$CUMDANCauchy, archive$VNSDEEPSO)
#CUMDANCauchy vs PSO_GBP
Wilcoxontest( archive$CUMDANCauchy, archive$PSO_GBP)
#CUMDANCauchy vs HL_PS_VNSO
Wilcoxontest( archive$CUMDANCauchy, archive$HL_PS_VNSO)
#CUMDANCauchy vs ABC_DE
Wilcoxontest( archive$CUMDANCauchy, archive$ABC_DE)
#CUMDANCauchy vs AJSO
Wilcoxontest( archive$CUMDANCauchy, archive$AJSO)
#CUMDANCauchy vs CE_CMAES
Wilcoxontest( archive$CUMDANCauchy, archive$CE_CMAES)
#CUMDANCauchy vs GASAPSO
Wilcoxontest( archive$CUMDANCauchy, archive$GASAPSO)
#CUMDANCauchy vs HFEABC
Wilcoxontest( archive$CUMDANCauchy, archive$HFEABC)
Wilcoxontest( archive$CUMDANCauchy, archive$CUMDANCauchy)
