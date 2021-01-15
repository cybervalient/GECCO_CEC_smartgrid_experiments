#############################################################
#         This code was created by: Yoan Martínez López     #
#         email: yoan.martinez@reduc.edu.cu                 #
#                cybervalient@gmail.com                     #  
#                         2021                              #
#############################################################

archive=read.csv("C://Users//Yoan M//Documents//miproyecto//GECCO_CEC_smartgrid_experiments//avg_conv_rate.csv",header=T);
str(archive)

#normality test < 100 cases (Shapiro-Wilks Test)
#chaotic_deepso
shapiro.test(archive$chaotic_deepso)
#DESS
shapiro.test(archive$DESS)
#EPSO
shapiro.test(archive$EPSO)
#evdeepso
shapiro.test(archive$evdeepso)
#Firefly
shapiro.test(archive$Firefly)
#Guide_DE
shapiro.test(archive$Guide_DE)
#UPSO
shapiro.test(archive$UPSO)
#GMVNPSO
shapiro.test(archive$GMVNPSO)
#VNSDEEPSO
shapiro.test(archive$VNSDEEPSO)
#PSO_GBP
shapiro.test(archive$PSO_GBP)
#CUMDANCauchy
shapiro.test(archive$CUMDANCauchy)
#HL_PS_VNSO
shapiro.test(archive$HL_PS_VNSO)
#ABC_DE
shapiro.test(archive$ABC_DE)
#AJSO
shapiro.test(archive$AJSO)
#CE_CMAES
shapiro.test(archive$CE_CMAES)
#GASAPSO
shapiro.test(archive$GASAPSO)
#HFEABC
shapiro.test(archive$HFEABC)
