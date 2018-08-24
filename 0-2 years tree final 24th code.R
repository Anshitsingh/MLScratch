
library(data.table)
library(dplyr)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(caret)
library(ggplot2)

#set working directory
setwd("E:/Petplan")

claim_level_data<-read.csv("ProfitabilityData_V1.csv",sep="|",quote = "",  row.names = NULL,stringsAsFactors = FALSE)
claim_level_data$PetId<-toupper(claim_level_data$PetId)

claim_level_data$Severity<-ifelse(claim_level_data$Severity=="","Curable",claim_level_data$Severity)
claim_level_data$Claimcodecategory<-ifelse(claim_level_data$Claimcodecategory=="","Unknown",claim_level_data$Claimcodecategory)
claim_level_data$Claimcodecategory<-ifelse(claim_level_data$Claimcodecategory=="Others","Illness",claim_level_data$Claimcodecategory)


claim_level_data$claimdurationInception<-as.integer(claim_level_data$claimdurationInception)

claim_level_data<-claim_level_data[!is.na(claim_level_data$claimdurationInception),]

# Fix negative claim inception duration
claim_level_data$claimdurationInception<-ifelse(claim_level_data$claimdurationInception< 5 & claim_level_data$Claimcodecategory=="Accident",5,claim_level_data$claimdurationInception)
claim_level_data$claimdurationInception<-ifelse(claim_level_data$claimdurationInception< 15 & claim_level_data$Claimcodecategory=="Illness",15,claim_level_data$claimdurationInception)


claim_level_data<-claim_level_data[order(claim_level_data$claimdurationInception),]

claim3<-subset(claim_level_data,claim_level_data$claimdurationInception<=30)

claim6<-subset(claim_level_data,claim_level_data$claimdurationInception<=60)

claim9<-subset(claim_level_data,claim_level_data$claimdurationInception<=90)

claim36<-subset(claim_level_data,claim_level_data$claimdurationInception<=365)

claim_level_data<-subset(claim_level_data,claim_level_data$claimdurationInception<=730)


claim_level_data$Claimin30<-"No"
claim_level_data$Claimin30[claim_level_data$PetId %in% claim3$PetId]<-"Yes"
claim_level_data$Claimin30<-as.factor(claim_level_data$Claimin30)

claim_level_data$Claimin60<-"No"
claim_level_data$Claimin60[claim_level_data$PetId %in% claim6$PetId]<-"Yes"
claim_level_data$Claimin60<-as.factor(claim_level_data$Claimin60)

claim_level_data$Claimin90<-"No"
claim_level_data$Claimin90[claim_level_data$PetId %in% claim9$PetId]<-"Yes"
claim_level_data$Claimin90<-as.factor(claim_level_data$Claimin90)

claim_level_data$Claimin365<-"No"
claim_level_data$Claimin365[claim_level_data$PetId %in% claim36$PetId]<-"Yes"
claim_level_data$Claimin365<-as.factor(claim_level_data$Claimin365)


#retain only one instance of each petid
claim_level_data<-claim_level_data[order(claim_level_data$claimdurationInception,decreasing=F),]
claim_level_data<-claim_level_data[!duplicated(claim_level_data$PetId),]

#retain required variables
claim_level_data<-claim_level_data[,c("PetId","Claimcodecategory","Severity","claimdurationInception","ClaimDetails","Claimin30","Claimin60","Claimin90","Claimin365")]
claim_level_data$ClaimDetailsCode<-substr(claim_level_data$ClaimDetails,1,3)
claim_level_data$ClaimDetailsCode<-ifelse(claim_level_data$ClaimDetailsCode=="","Unknown",claim_level_data$ClaimDetailsCode)


#Read claim details code file
claimdetailscode<-read.csv("claimdetailscodecount.csv",stringsAsFactors = F)
claimdetailscode<-claimdetailscode[order(claimdetailscode$Freq,decreasing = T),]

#Retain top 30 codes, these codes cover 60% of the claims
claimdetailscode<-claimdetailscode[1:30,]


#collapse claim details code column
claim_level_data$ClaimDetailsCode[!claim_level_data$ClaimDetailsCode %in% claimdetailscode$Var1]<-"Others"


# Read policy level data
input<-read.csv("ProfitabilityData_V2.csv",sep="|",quote = "",  row.names = NULL,stringsAsFactors = FALSE)
input$PetId<-toupper(input$PetId)

input$CouponUsed<-ifelse(input$CampaignCd==0,0,1)

input$Duration<-ifelse(input$Duration<0,0,input$Duration)

input$TransactionDt<-as.Date(input$TransactionDt,format="%Y-%m-%d %H:%M:%S")
input<-input[order(input$TransactionDt),]

input$ClaimAmount[is.na(input$ClaimAmount)]<-0
input$EarnedPremiumAmt[is.na(input$EarnedPremiumAmt)]<-0

input$CumDuration <- ave(input$Duration, input$PetId, FUN=cumsum)
input$CumClaimAmount <- ave(input$ClaimAmount, input$PetId, FUN=cumsum)
input$CumEarnedPremiumAmt <- ave(input$EarnedPremiumAmt, input$PetId, FUN=cumsum)

inputdata<-input


##### For 2 to 5 yrs uncomment

#tocallr1<-subset(inputdata,inputdata$CumDuration<=730)
#tocallr1<-tocallr1[order(tocallr1$TransactionDt),]
#tocallr1$CumClaimAmount <- ave(tocallr1$ClaimAmount, tocallr1$PetId, FUN=cumsum)
#tocallr1$CumEarnedPremiumAmt <- ave(tocallr1$EarnedPremiumAmt, tocallr1$PetId, FUN=cumsum)
#tocallr1$LR1<-tocallr1$CumClaimAmount/tocallr1$CumEarnedPremiumAmt
#tocallr1<-tocallr1[order(tocallr1$CumDuration,decreasing=T),]
#tocallr1<-tocallr1[!duplicated(tocallr1$PetId),]
#tocallr1<-tocallr1[,c("PetId","LR1")]
#tocallr1$LR1[is.nan(tocallr1$LR1)]<-0
#tocallr1$LR1[is.infinite(tocallr1$LR1)]<-0


input<-subset(inputdata,inputdata$CumDuration<=730)
input<-input[order(input$TransactionDt),]
input$CumClaimAmount <- ave(input$ClaimAmount, input$PetId, FUN=cumsum)
input$CumEarnedPremiumAmt <- ave(input$EarnedPremiumAmt, input$PetId, FUN=cumsum)



###########################################

#input<-subset(input,input$CumDuration>1825)

input$LossRatio<-input$CumClaimAmount/input$CumEarnedPremiumAmt

input<-merge(input,claim_level_data,by="PetId",all.x = T)

#input$CCount2[is.na(input$CCount2)]=0
#input$CCount1[is.na(input$CCount1)]=0
#input$AverageClaimAmt1[is.na(input$AverageClaimAmt1)]=0
#input$AverageClaimAmt2[is.na(input$AverageClaimAmt2)]=0



#input$claimdurationInception[is.na(input$claimdurationInception)]<-input$CumDuration[is.na(input$claimdurationInception)]

input$claimdurationInception[is.na(input$claimdurationInception)]<-0

input$Severity[is.na(input$Severity)]<-"None"
input$Severity<-as.factor(input$Severity)

input$Claimcodecategory[is.na(input$Claimcodecategory)]<-"None"
input$Claimcodecategory<-as.factor(input$Claimcodecategory)


input$ClaimDetailsCode[is.na(input$ClaimDetailsCode)]<-"None"
input$ClaimDetailsCode<-as.factor(input$ClaimDetailsCode)

#rm(claim_level_data)



# To flag policies which are renewed
trans<-input[,c("PetId","TransactionCd")]
dmy <- dummyVars(" ~ TransactionCd", data = trans)
trsf <- data.frame(predict(dmy, newdata = trans))
trans<-cbind(trans,trsf)
trans$TransactionCd<-NULL
trans$TransactionCdRenewal<-ifelse(trans$TransactionCdRewrite.Renewal>0,trans$TransactionCdRewrite.Renewal,trans$TransactionCdRenewal)

res=aggregate(.~PetId, trans, sum)
res$TransactionCdNew.Business<-NULL
res$TransactionCdPolicy<-NULL
res$TransactionCdRewrite.Renewal<-NULL

input$PolicyForm<-ifelse(input$PolicyForm=="Intro","Introductory",input$PolicyForm)

notintropol<-subset(input,input$PolicyForm!="Introductory")
intropol<-subset(input,input$PolicyForm=="Introductory")
intropol$Introductory_Upgrade<-1
intropol<-intropol[,c("PetId","Introductory_Upgrade")]

intropol<-intropol[!duplicated(intropol$PetId),]

input<-merge(notintropol,intropol,by.x = "PetId",by.y = "PetId",all.x = T)

input<-merge(input,res,by.x = "PetId",by.y = "PetId",all.x = T)


input$Introductory_Upgrade<-ifelse(is.na(input$Introductory_Upgrade),0,input$Introductory_Upgrade)
input$Introductory_Upgrade<-ifelse(input$Introductory_Upgrade==0,"No","Yes")
input$Introductory_Upgrade<-as.factor(input$Introductory_Upgrade)

rm(list=setdiff(ls(), "input"))


input<-input[order(input$CumDuration,decreasing=T),]
input<-input[!duplicated(input$PetId),]


input$LossRatio[is.na(input$LossRatio)]<-0

input <- input[!is.infinite(input$LossRatio),]

input$Age[is.na(input$Age)]<-median(input$Age,na.rm=T)
input$Duration[is.na(input$Duration)]<-median(input$Duration,na.rm=T)

#make carrier code as categorical
input$CarrierCd<-as.factor(input$CarrierCd)

#make annual deduction indicator as categorical
input$AnnualDedInd <-as.factor(input$AnnualDedInd)

#Adjust copaypct
input$CopayPct<-ifelse(input$CopayPct==10,0.9,input$CopayPct)
input$CopayPct<-ifelse(input$CopayPct==20,0.8,input$CopayPct)
input$CopayPct<-ifelse(input$CopayPct==30,0.7,input$CopayPct)
input$CopayPct<-ifelse(input$CopayPct==0,1,input$CopayPct)
input$CopayPct<-as.factor(input$CopayPct)


#Indicator to whether the pet is of mixed type
input$BreedName<-as.character(input$BreedName)
input$Mixed_Breed=ifelse(input$BreedName %like% "Mix","Yes","No")
input$Mixed_Breed<-as.factor(input$Mixed_Breed)
input$BreedName<-NULL

# Fix pet types
input$PetType<-as.character(input$PetType)
input$PetType<-ifelse(input$PetType=="PPCAT001","Cat",input$PetType)
input$PetType<-ifelse(input$PetType=="PPDOG001","Dog",input$PetType)
input$PetType<-as.factor(input$PetType)


#fix states
input$ControllingStateCd<-toupper(input$ControllingStateCd)
reg<-read.csv("region_codes.csv",stringsAsFactors = F)
input<-merge(input,reg,by.x="ControllingStateCd",by.y="Code",all.x = T)
input$Region[is.na(input$Region)]<-"West"
input$Region<-as.factor(input$Region)


# make policy types categorical
input$PolicyForm<-as.factor(input$PolicyForm)

# make rpt sales executve categorical
input$RptSalesExecutive<-as.factor(input$RptSalesExecutive)

input$Country<-as.factor(input$Country)

input$CarrierCd<-toupper(input$CarrierCd)
input<-subset(input,input$CarrierCd!="000" & input$CarrierCd!="137")



df<-input %>% select(#PetId,
                     Age,
                     AnnualDedInd,
                     #CampaignCd,
                     #CancelDt,
                     CarrierCd,
                     #ControllingStateCd,
                     CopayPct,
                     #CrtdDateTime,
                     #CrtdUser,
                     Country,
                     #CustomerNumber,
                     Deductible,
                     #Duration,
                     #EarnedPremiumAmt,
                     #EarnedPremiumAmtLocal,
                     #Effectivedt,
                     #EmailAddr,
                     #ExchangeRate,
                     #ExpirationDt,
                     #InitialWrittenPremiumAmt,
                     #InitialWrittenPremiumAmtLocal,
                     #LastAnnualPremiumAmt,
                     #LastAnnualPremiumAmtLocal,
                     #PayplanCd,
                     #PetName,
                     PetType,
                     #PolicyDisplayNumber,
                     #PolicyEndDate,
                     PolicyForm,
                     #PolicyRef,
                     #PolicyVersion,
                     #Processed,
                     #RenewedFromPolicyRef,
                     #RptSalesExecutive,
                     #SourceSystem,
                     #StatusCd,
                     #TransactionCd,
                     #TransactionDt,
                     #TransactionNumber,
                     #UpdateTimestamp,
                     #ClaimCount,
                     #ClaimAmount,
                     #CumClaimAmount,
                     #CumEarnedPremiumAmt,
                     Introductory_Upgrade,
                     TransactionCdRenewal,
                     LossRatio,
                     Mixed_Breed,
                     Region,
                     Severity,
                     #claimdurationInception,
                     Claimcodecategory,
                     #LR1,
                     Claimin30,
                     Claimin60,
                     Claimin90,
                     Claimin365,
                     #ClaimCur,
                     #ClaimNonCur,
                     CouponUsed )
                     #ClaimDetailsCode,
                     #CCount2,
                     #CCount1,
                     #AverageClaimAmt2,
                     #AverageClaimAmt1



## 80% of the sample size
smp_size <- floor(0.8 * nrow(df))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

train <- df[train_ind,]
test <- df[-train_ind,]



fit<-rpart(LossRatio~.,data=train,method = "anova")

fit<-rpart(LossRatio~.,data=train,method = "anova",
           control=rpart.control(minsplit=5000, minbucket=5000, cp=0.0001))

cptable<-as.data.frame(fit$cptable)
#cptable<-cptable[order(cptable$CP,decreasing = F)]

pfit<-prune(fit,cp=0.0005492773)

plotcp(pfit)

rpart.plot(fit,digits=6,cex=0.5)

rpart.plot(pfit,digits=6,cex=0.5)

fancyRpartPlot(pfit)#



train$Severity<-as.character(train$Severity)
data2vaibhav<-subset(train,train$Severity!="None")
data2vaibhav$ClaimNonCur<-as.character(data2vaibhav$ClaimNonCur)
data2vaibhav<- subset(data2vaibhav,data2vaibhav$ClaimNonCur_1>="Yes")
write.csv(data2vaibhav,file="highloss.csv",row.names = F)
test$Prediction<-predict(pfit,test)
error<-mean(abs(test$Prediction-test$LossRatio))
#rmse<-sqrt(error)
rmse
error
