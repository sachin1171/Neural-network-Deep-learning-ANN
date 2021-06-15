##################### problem 1 ############################
library(neuralnet)  # regression
library(nnet) # classification 
#install.packages("NeuralNetTools")
library(NeuralNetTools)
library(plyr)
# Read the data
Startups <- read.csv(file.choose())

View(Startups)
class(Startups)

Startups$State <- as.numeric(revalue(Startups$State,
                                     c("New York"="0", "California"="1",
                                       "Florida"="2")))
str(Startups)


Startups <- as.data.frame(Startups)
attach(Startups)


# Exploratory data Analysis :

plot(R.D.Spend, Profit)
plot(Administration, Profit)
plot(Marketing.Spend, Profit)
plot(State, Profit)

windows()
# Find the correlation between Output (Profit) & inputs (R.D Spend, Administration, Marketing, State) - SCATTER DIAGRAM
pairs(Startups)

# Correlation coefficient - Strength & Direction of correlation
cor(Startups)

####### Scatter plot matrix with Correlations inserted in graph
panel.cor <- function(x, y, digits=2, prefix="", cex.cor)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r = (cor(x, y))
  txt <- format(c(r, 0.123456789), digits=digits)[1]
  txt <- paste(prefix, txt, sep="")
  if(missing(cex.cor)) cex <- 0.4/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex)
}
pairs(Startups, upper.panel=panel.cor,main="Scatter Plot Matrix with Correlation Coefficients")

summary(Startups) # Confirms on the different scale and demands normalizing the data.


# Apply Normalization technique to the whole dataset :

normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}

Startups_norm <- as.data.frame(lapply(Startups,FUN=normalize))

summary(Startups_norm$Profit) # Normalized form of profit

summary(Startups$profit) # Orginal profit value

# Data Partition 
set.seed(123)
ind <- sample(2, nrow(Startups_norm), replace = TRUE, prob = c(0.7,0.3))
Startups_train <- Startups_norm[ind==1,]
startups_test  <- Startups_norm[ind==2,]


# Creating a neural network model on training data

startups_model <- neuralnet(Profit~R.D.Spend+Administration
                            +Marketing.Spend+State,data = Startups_train)
str(startups_model)


plot(startups_model)
summary(startups_model)

par(mar = numeric(4), family = 'serif')
plotnet(startups_model, alpha = 0.6)

# Evaluating model performance

set.seed(12323)
model_results <- compute(startups_model,startups_test[1:4])
predicted_profit <- model_results$net.result

# Predicted profit Vs Actual profit of test data.
cor(predicted_profit,startups_test$Profit)
#0.9652506


# Improve the model performance :
set.seed(12345)
Startups_model2 <- neuralnet(Profit~R.D.Spend+Administration
                             +Marketing.Spend+State,data = Startups_train,
                             hidden = 2)
plot(Startups_model2 ,rep = "best")
summary(Startups_model2)

model_results2<-compute(Startups_model2,startups_test[1:4])

predicted_Profit2<-model_results2$net.result

cor(predicted_Profit2,startups_test$Profit)
#0.9648132

plot(predicted_Profit2,startups_test$Profit)

par(mar = numeric(4), family = 'serif')
windows()
plotnet(Startups_model2, alpha = 0.6)

# SSE(Error) has reduced and training steps had been increased as the number of neurons  under hidden layer are increased
##################### problem 2 ############################
library(neuralnet)  # regression
library(nnet) # classification 
library(NeuralNetTools)
library(plyr)

Forest <- read.csv(file.choose())
View(Forest)
class(Forest)

str(Forest)

FF<- Forest[,1:11]
View(FF)

# Convert month and day string variables into numeric values
FF$month <- as.numeric(as.factor(FF$month))
FF$day <- as.numeric(as.factor(FF$day))

# The area value has lots of zeros
# Transform the Area value to Y 

FF1 <- mutate(FF, y = log(area + 1))  # default is to the base e, y is lower case
hist(FF1$y)

summary(FF1) # Confirms on the different scale and demands normalizing the data.
# Prediction of Forest fires requires only prediction form 

# Apply Normalization technique to the whole dataset :

normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}

FF_norm<-as.data.frame(lapply(FF1,FUN=normalize))
#FF_norm = cbind(FF[,c(1,2)], FF_norm)
summary(FF1$area) # Normalized form of area

summary(FF_norm) # Orginal  value

windows()
pairs(FF1)

cor(FF1)

####### Scatter plot matrix with Correlations inserted in graph
panel.cor <- function(x, y, digits=2, prefix="", cex.cor)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r = (cor(x, y))
  txt <- format(c(r, 0.123456789), digits=digits)[1]
  txt <- paste(prefix, txt, sep="")
  if(missing(cex.cor)) cex <- 0.4/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex)
}
pairs(FF1, upper.panel=panel.cor,main="Scatter Plot Matrix with Correlation Coefficients")


# Data Partition 
set.seed(123)
ind <- sample(2, nrow(FF_norm), replace = TRUE, prob = c(0.7,0.3))
FF_train <- FF_norm[ind==1,]
FF_test  <- FF_norm[ind==2,]

# to train model
# Creating a neural network model on training data
FF_model <- neuralnet(area~.,data = FF_train)
str(FF_model)

plot(FF_model, rep = "best")

summary(FF_model)
#Error is 0.0161. SSE sum of squared errors . least SSE best model

# Evaluating model performance
# compute function to generate ouput for the model prepared
set.seed(12323)
model_results <- compute(FF_model,FF_test)
str(model_results)
predicted_strength <- model_results$net.result

cor(predicted_strength,FF_test$area) #0.908
plot(predicted_strength,FF_test$area)


mean(predicted_strength==FF_test$area)


#Building Model 2
model_5<-neuralnet(area~.,data= FF_train,hidden = 5,linear.output = T)

plot(model_5)
model_5
#Evaluating model performance
model_5_res<-compute(model_5,FF_test)
model_5_res$net.result
str(model_5_res)

pred_strn_5<-model_5_res$net.result
cor(pred_strn_5,FF_test$area) #0.5521

plot(pred_strn_5,FF_test$area)

length(pred_strn_5)
length(FF_test$area)
mean(pred_strn_5)
mean(FF_test$area)

mean(pred_strn_5==FF_test$area)

#Building Model 3
model_3<-neuralnet(area~.,data= FF_train,hidden = 2,linear.output = T)

plot(model_3)

#Evaluating model performance
model_3_res<-compute(model_3,FF_test)
model_3_res$net.result
str(model_3_res)

pred_strn_3<-model_3_res$net.result
cor(pred_strn_3,FF_test$area) #0.0924

plot(pred_strn_3,FF_test$area)

length(pred_strn_3)
length(FF_test$area)
mean(pred_strn_3)
mean(FF_test$area)

mean(pred_strn_3==FF_test$area)
##################### problem 3  ############################
# Load the Concrete data as concrete
concrete <-read.csv(file.choose(), stringsAsFactors = TRUE)
# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
concrete_norm <- as.data.frame(lapply(concrete, normalize))

# create training and test data
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# simple ANN with only a single hidden neuron
concrete_model <- neuralnet(formula = strength ~ cement + slag +
                              ash + water + superplastic + 
                              coarseagg + fineagg + age,
                            data = concrete_train)


# visualize the network topology
plot(concrete_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(concrete_model, concrete_test[1:8])
# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_strength, concrete_test$strength)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden = 5)


# plot the network
plot(concrete_model2)

# evaluate the results as we did before
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)
plot(predicted_strength,concrete_test$strength)
par(mar = numeric(4), family = 'serif')
plotnet(concrete_model2, alpha = 0.6)
#SSE(Error) has reduced and training steps had been increased as the number of neurons
# under hidden layer are increased
######################## problem 4 #############################
# Loading the dataset
library(readr)
RPL_data <- read.csv(file.choose(), header = TRUE)
RPL_data <- RPL_data[,4:14]
summary(RPL_data)
str(RPL_data)

#label encoding the data
#install.packages('superml')
library(superml)
label <- LabelEncoder$new()
RPL_data$Geography <- label$fit_transform(RPL_data$Geography)
RPL_data$Gender <- label$fit_transform(RPL_data$Gender)
str(RPL_data)

#attaching dataframe
attach(RPL_data)

# Exploratory data Analysis :
plot(Geography, Exited)
plot(Gender, Exited)

windows()
# Find the correlation between Output (Exited) & inputs (R.D Spend, Administration, Marketing, State) - SCATTER DIAGRAM
pairs(RPL_data)

# Correlation coefficient - Strength & Direction of correlation
cor(RPL_data)

# Apply Normalization technique to the whole dataset
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
RPL_norm<-as.data.frame(lapply(RPL_data,FUN=normalize))
summary(RPL_norm$Exited) # Normalized form of profit

# Data Partition 
set.seed(123)
ind <- sample(2, nrow(RPL_norm), replace = TRUE, prob = c(0.7,0.3))
RPL_train <- RPL_norm[ind==1,]
RPL_test  <- RPL_norm[ind==2,]

# Creating a neural network model on training data
#since output is in binary its classification 
#nnet for classification
#install.packages("nnet")
library(nnet)
RPL_model <- nnet(Exited~.,data = RPL_train , size = 5,rang = 0.1 , decay = 5e-2,maxit = 5000)
str(RPL_model)

# visualize the network topology
plotnet(RPL_model)
garson(RPL_model)

# Evaluating model performance
set.seed(12323)
predicted_exited <- predict(RPL_model,RPL_test[-c(11)])

# Predicted profit Vs Actual profit of test data.
cor(predicted_exited,RPL_test$Exited)

# since the prediction is in Normalized form, we need to de-normalize it 
# to get the actual prediction on profit
str_max <- max(RPL_data$Exited)
str_min <- min(RPL_data$Exited)

unnormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}

Actualexited_pred <- unnormalize(predicted_exited,str_min,str_max)
head(Actualexited_pred)

# Improve the model performance :
set.seed(12345)
RPL_model2 <- nnet(Exited~.,data = RPL_train,
                   hidden = 2 , size = 5)
plotnet(RPL_model2 ,rep = "best")

summary(RPL_model2)

# evaluate the results as we did before
predicted_exited2 <- predict(RPL_model2,RPL_test[-c(11)])

#plotting
plot(predicted_exited2,RPL_test$Exited)

#install.packages("NeuralNetTools")
library(NeuralNetTools)
par(mar = numeric(4), family = 'serif')
plotnet(RPL_model2, alpha = 0.6)

