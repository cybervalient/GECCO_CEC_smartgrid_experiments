#############################################################
#         This code was created by: Yoan Martínez López     #
#         email: yoan.martinez@reduc.edu.cu                 #
#                cybervalient@gmail.com                     #  
#                         2021                              #
#############################################################
source(file.choose()) #Load to tests.R
source(file.choose()) #Load to post_hoc.R
library(FSA)

FriendmanTest <-function(dataFile, sep =","){
  x<-read.table(dataFile, header = TRUE, sep = sep)
# 
  x <- abs(x) #if negative value
  x <- x*(-1)  #Minimize
  
  print(friedmanTest(x))
  
  print(imanDavenportTest(x))
  #calculate Mean of rank
  mean.rank<-colMeans(rankMatrix(x))
  #Sorted Vector 
  pos<-order(mean.rank)[1]
 
  mean.rank = mean.rank[order(mean.rank)]
  
  FriendRank<-data.frame(Ranking=mean.rank)
  
  print(FriendRank)
  
  postHoc<-friedmanPost(x, control = NULL)
  hoc_vector<-postHoc[pos,]
  bad<-is.na(hoc_vector)
  hoc_v<-hoc_vector[!bad]
  Data <- data.frame(P_value=hoc_v)
  
  ### Check if data is ordered the way we intended
  headtail(Data)
  
  ### Perform p-value adjustments and add to data frame
  ##Controlling the familywise error rate: Bonferroni correction
  ####The methods Holm, Hochberg, Hommel, and Bonferroni control the family-wise error rate.  These methods attempt to limit the probability of even one false discovery (a type I error, incorrectly rejecting the null hypothesis when there is no real effect), and so are all relatively strong (conservative).  
  Data$Bonferroni = 
    p.adjust(Data$P_value, 
             method = "bonferroni")
  
  Data$BH = 
    p.adjust(Data$P_value, 
             method = "BH")
  
  Data$Holm = 
    p.adjust(Data$P_value, 
             method = "holm")
  
  Data$Hochberg = 
    p.adjust(Data$P_value, 
             method = "hochberg")
  
  Data$Hommel = 
    p.adjust(Data$P_value, 
             method = "hommel")
  
  Data$BY = 
    p.adjust(Data$P_value, 
             method = "BY")
  
  print(Data)
  X = Data$P_value
  Y = cbind(Data$Bonferroni,
            Data$BH,
            Data$Holm,
            Data$Hochberg,
            Data$Hommel,
            Data$BY)
  
  matplot(X, Y,
          xlab="p-value",
          ylab="Adjusted p-value",
          type="l",
          asp=1,
          col=1:6,
          lty=1,
          lwd=2)
  
  legend('bottomright', 
         legend = c("Bonferroni", "BH", "Holm", "Hochberg", "Hommel", "BY"), 
         col = 1:6, 
         cex = 1,    
         pch = 16)
  
  abline(0, 1,
         col=1,
         lty=2,
         lwd=1) 
}


FriendmanTest(file.choose()) #Load to file .csv

