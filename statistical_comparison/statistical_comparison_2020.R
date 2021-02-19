#############################################################
#         This code was created by: Yoan Martínez López     #
#         email: yoan.martinez@reduc.edu.cu                 #
#                cybervalient@gmail.com                     #  
#                         2021                              #
#############################################################
#install.packages(readr)
# Cargar el paquete
library(readr)
#archive = read_csv(file.choose(),header=T)
archive = read_csv(file.choose())
#archive=read_tsv("https://github.com/cybervalient/GECCO_CEC_smartgrid_experiments/tree/master/statistical_comparison/avg_conv_rate.csv")
str(archive)

#normality test < 100 cases (Shapiro-Wilks Test)

shapiro.test(archive$RDG3_DEEPSO)
shapiro.test(archive$CE_CMAES)
shapiro.test(archive$HFEABC)
shapiro.test(archive$DE_TLBO)
shapiro.test(archive$PSO_GBP)
shapiro.test(archive$GASAPSO)
shapiro.test(archive$AJSO)
shapiro.test(archive$CUMDANCauchy)
shapiro.test(archive$EHL_PS_VNSO)
shapiro.test(archive$Ensembled_method_of_CBBO_Cauchy_and_DEEPSO)
shapiro.test(archive$`Ensembled_method_of_CBBO_Cauchy and DEEPSO`)
shapiro.test(archive$DEEDA)
shapiro.test(archive$VNS_DEEPSO)

