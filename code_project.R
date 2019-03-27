#Libraries used
library(dplyr)
library(ggplot2)
library(ggthemes)
library(MASS)
library(FNN)
library(corrplot)
library(reshape2)
library(pROC)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)

#Reading the dataset into R
dataset <- read.csv ("~/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

#Summary stats
str(dataset)
summary(dataset)
head(dataset)
table(dataset$quality)

#Creating a Correlation heatmap of variables
corrplot(cor(dataset))

plot(density((dataset$quality)))

library(moments)
agostino.test(dataset$quality)

library(car)
scatterplotMatrix(dataset)



#Distribution of red wine quality ratings
ggplot (data = dataset, aes(x=quality, fill = factor(quality))) +
  geom_bar (stat = "count") +
  scale_x_continuous (breaks = seq(3,8,1)) +
  ggtitle ("Distribution of quality ratings of red wine") +
  xlab (label = "Quality rating") +
  ylab (label = "Count") 
  

#Most of the ratings are concentrated around 5 and 6

#Start by viewing as a regression problem

#Linear regression model

#Start by building a linear regression model with all variables
model_LR_1 = lm(quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=dataset)
summary(model_LR_1)

#Too many insignificant variables - very low adjusted R-squared - 35.61%

#Use stepwise procedure to automatically eliminate variables
model_LR_2 = lm(quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=dataset)
model_LR_stepwise <- stepAIC(model_LR_2, direction = "both")
model_LR_stepwise$anova
summary(model_LR_stepwise)

#Eliminated 4 variables - better model - Adjusted R squared value - 35.67%
#can we remove free.sulfur.dioxide (has high correl with total.sulfur.dioxide)? - p-value > 0.01
model_LR_2 = lm(quality ~ volatile.acidity+chlorides+total.sulfur.dioxide+pH+sulphates+alcohol, data=dataset)
summary(model_LR_2)

#The adjusted R squared value has dropped to 35.48% - Clearly stepwise regression model was better

#Visualizing the results
ggplot() +
  geom_point(aes(x = dataset$quality, y = predict(model_LR_stepwise, newdata = dataset)), colour = "red", size = 4) +
  geom_line(aes(x = dataset$quality, y = dataset$quality), color = "black") +
  ggtitle ("Actual v/s Predicted values") +
  xlab (label = "Actual rating") +
  ylab (label = "Predicted rating") +
  theme_base()

#We can see how bad the model is
#Besides adjusted regression value of 35.67% indicates that we need to do something else

#Generalized Linear Model

#Let's just use the variables that the stepwise
model_GLM_1 = glm(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=dataset, family=gaussian(link="log"))
model_GLM_1_stepwise <- stepAIC(model_GLM_1, direction = "both")
model_GLM_1_stepwise$anova
summary(model_GLM_1_stepwise)

#Let's compare this to previous stepwise linear regression model
#Visualizing the results
ggplot() +
  geom_point(aes(x = dataset$quality, y = predict(model_LR_stepwise, newdata = dataset)), colour = "red", size = 4) +
  geom_point(aes(x = dataset$quality, y = exp(predict(model_GLM_1_stepwise, newdata = dataset))), shape = 2 , size = 4) +
  geom_line(aes(x = dataset$quality, y = dataset$quality), color = "black") +
  ggtitle ("Actual v/s Predicted values") +
  xlab (label = "Actual rating") +
  ylab (label = "Predicted rating") +
  theme_base()

#No real improvement - let's try another GLM model and compare against current one
model_GLM_2 = glm(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=dataset, family=poisson(link="identity"))
model_GLM_2_stepwise <- stepAIC(model_GLM_2, direction = "both")
model_GLM_2_stepwise$anova
summary(model_GLM_2_stepwise)

ggplot() +
  geom_point(aes(x = dataset$quality, y = exp(predict(model_GLM_1_stepwise, newdata = dataset))), colour = "blue", size = 4) +
  geom_point(aes(x = dataset$quality, y = predict(model_GLM_2_stepwise, newdata = dataset)), shape = 2, size = 4) +
  geom_line(aes(x = dataset$quality, y = dataset$quality), color = "black") +
  ggtitle ("Actual v/s Predicted values") +
  xlab (label = "Actual rating") +
  ylab (label = "Predicted rating") +
  theme_base()

#No real improvement here as well - let's move on to another type of model

#k-nearest neighbors regression

#Let's try with 5 neighbors
model_knn5 = knn.reg (train = dataset[,1:11], test = dataset[,1:11], y=dataset$quality, k=5)

ggplot() +
  geom_point(aes(x = dataset$quality, y = model_knn5$pred), colour = "green", size = 4) +
  geom_line(aes(x = dataset$quality, y = dataset$quality), color = "black") +
  ggtitle ("Actual v/s Predicted values") +
  xlab (label = "Actual rating") +
  ylab (label = "Predicted rating") +
  theme_base()

#Is there improvement against previous model?
ggplot() +
  geom_point(aes(x = dataset$quality, y = model_knn5$pred), colour = "green", size = 4) +
  geom_point(aes(x = dataset$quality, y = predict(model_GLM_2_stepwise, newdata = dataset)), shape = 2, size = 4) +
  geom_line(aes(x = dataset$quality, y = dataset$quality), color = "black") +
  ggtitle ("Actual v/s Predicted values") +
  xlab (label = "Actual rating") +
  ylab (label = "Predicted rating") +
  theme_base()

#Let's make a KNNR model with 15 neighbors and compare it to the model with 5 neighbors
model_knn15 = knn.reg (train = dataset[,1:11], test = dataset[,1:11], y=dataset$quality, k=15)

ggplot() +
  geom_point(aes(x = dataset$quality, y = model_knn5$pred), colour = "green", size = 4) +
  geom_point(aes(x = dataset$quality, y = model_knn15$pred), shape = 2, size = 4) +
  geom_line(aes(x = dataset$quality, y = dataset$quality), color = "black") +
  ggtitle ("Actual v/s Predicted values") +
  xlab (label = "Actual rating") +
  ylab (label = "Predicted rating") +
  theme_base()

#Not a good model


#Now viewing as a classification problem

#Creating variable for classification of wine into good/bad
dataset$good <- ifelse (dataset$quality > 6, 1, 0)

#Scatterplot of each variable against every other variable
plot(dataset)

#Creating the Correlation heatmap of variable
corrplot(cor(dataset))

#Highest correlation with good_tag: alcohol, volatile.acidity

#Distribution of good/bad red wines
ggplot (data = dataset, aes(x=good_tag, fill = factor(good_tag))) +
  geom_bar (stat = "count") +
  scale_x_continuous (breaks = seq(0,1,1)) +
  ggtitle ("Distribution of bood/bad red wines") +
  xlab (label = "Good wine?") +
  ylab (label = "Count") +
  theme_base()

#Checking how each of the independent variables affects the dependent variable

#Fixed Acidity vs Wine Quality
ggplot (data = dataset, aes(x=fixed.acidity, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(fixed.acidity[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(fixed.acidity[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(4,16,1)) +
  ggtitle ("Distribution of Fixed Acidity levels") +
  xlab (label = "Fixed Acidity Level") +
  theme_base()

#Volatile Acidity vs Wine Quality
ggplot (data = dataset, aes(x=volatile.acidity, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(volatile.acidity[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(volatile.acidity[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(0.1,1.6,0.1)) +
  ggtitle ("Distribution of Volatile Acidity levels") +
  xlab (label = "Volatile Acidity Level") +
  theme_base()

#Citric Acid vs Wine Quality
ggplot (data = dataset, aes(x=citric.acid, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(citric.acid[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(citric.acid[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(0,1,0.1)) +
  ggtitle ("Distribution of Citric Acid levels") +
  xlab (label = "Citric Acid Level") +
  theme_base()

#Residual Sugar vs Wine Quality
ggplot (data = dataset, aes(x=residual.sugar, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(residual.sugar[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(residual.sugar[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(0,16,1)) +
  ggtitle ("Distribution of Residual Sugar levels") +
  xlab (label = "Residual Sugar Level") +
  theme_base()

#Chlorides vs Wine Quality
ggplot (data = dataset, aes(x=chlorides, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(chlorides[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(chlorides[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(0,0.65,0.05)) +
  ggtitle ("Distribution of Chloride levels") +
  xlab (label = "Residual Chloride Level") +
  theme_base()

#Free Sulfur dioxide vs Wine Quality
ggplot (data = dataset, aes(x=free.sulfur.dioxide, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(free.sulfur.dioxide[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(free.sulfur.dioxide[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(0,73,5)) +
  ggtitle ("Distribution of Free Sulphur Dioxide levels") +
  xlab (label = "Free Sulfur Dioxide Level") +
  theme_base()

#Total Sulfur dioxide vs Wine Quality
ggplot (data = dataset, aes(x=total.sulfur.dioxide, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(total.sulfur.dioxide[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(total.sulfur.dioxide[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(0,290,50)) +
  ggtitle ("Distribution of Total Sulphur Dioxide levels") +
  xlab (label = "Total Sulfur Dioxide Level") +
  theme_base()

#Density vs Wine Quality
ggplot (data = dataset, aes(x=density, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(density[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(density[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(0.9,1.1,0.005)) +
  ggtitle ("Distribution of Density levels") +
  xlab (label = "Density Level") +
  theme_base()

#pH vs Wine Quality
ggplot (data = dataset, aes(x=pH, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(pH[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(pH[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(2.5,4.5,0.5)) +
  ggtitle ("Distribution of pH levels") +
  xlab (label = "pH Level") +
  theme_base()

#Sulphates vs Wine Quality
ggplot (data = dataset, aes(x=sulphates, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(sulphates[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(sulphates[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(0.25,2,0.25)) +
  ggtitle ("Distribution of sulphate levels") +
  xlab (label = "Sulphate Level") +
  theme_base()

#Alcohol vs Wine Quality
ggplot (data = dataset, aes(x=alcohol, fill = factor(good_tag))) + 
  geom_density (alpha = 0.5) +
  geom_vline (aes(xintercept = mean(alcohol[good_tag == 0], na.rm = T)), color = "red", lwd = 1, linetype = "dashed") +
  geom_vline (aes(xintercept = mean(alcohol[good_tag == 1], na.rm = T)), color = "blue", lwd = 1, linetype = "dashed") +
  scale_x_continuous (breaks = seq(8,16,2)) +
  ggtitle ("Distribution of alcohol levels") +
  xlab (label = "Alcohol Level") +
  theme_base()

#Factors which discriminate between good and bad wine the most: alcohol, sulphates, volatile acidity

#Logistic model

#Let's first make a logistic regression model with just variable - alcohol

model_log_1 = glm(good~alcohol, data=dataset, family=binomial(link="logit"))
summary(model_log_1)

#Values predicted by the model
pred_model_1 <- pnorm(predict(model_log_1))

#Plotting its ROC curve
roc1 <- plot.roc(dataset$good_tag,pred_model_1,main="",percent=TRUE, ci=TRUE, print.auc=TRUE)
roc1.se <- ci.se(roc1,specificities=seq(0,100,5))
plot(roc1.se,type="shape", col="orange")

#Excellent results - AUC: 82.2%


#Let's fit a logistic regression model based on stepwise procedure

#model_log_2 = glm(good_tag~alcohol + volatile.acidity + citric.acid + sulphates,data=dataset,family=binomial(link="logit"))

model_log_2 = glm(good~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=dataset,family=binomial(link="logit"))
model_log_2_stepwise <- stepAIC(model_log_2, direction = "both")
model_log_2_stepwise$anova
summary(model_log_2_stepwise)

#Values predicted by the model
pred_model_2 <- pnorm(predict(model_log_2_stepwise))

#Plotting its ROC curve
roc2 <-plot.roc(dataset$good,pred_model_2,main="",percent=TRUE, ci=TRUE, print.auc=TRUE)
roc2.se <- ci.se(roc2,specificities=seq(0,100,5))
plot(roc2.se,type="shape", col="blue")

#Even better results - AUC: 88.2%

#Let's see if any higher order models can do better


#KNN Classifier

#Let's try out KNN models with 20 and 10 neighbors respectively 
#Note that we are not spliting the data into test and training

model_class_knn20 = knn(train=dataset[,1:11],test=dataset[,1:11], cl= dataset$good_tag, k=20)
model_class_knn10 = knn(train=dataset[,1:11],test=dataset[,1:11], cl= dataset$good_tag, k=10) 
table(dataset$good_tag,model_class_knn20)
table(dataset$good_tag,model_class_knn10)

(204+4)/(204+4+13+1378)
(185+14)/(185+14+32+1368)

#Models are getting better:
#KNN20 has a classification error rate of 13%
#KNN10 has a classification error rate of 12.44%


# Decision trees

model_DF <- rpart(good_tag ~ . -quality, dataset, method="class")
summary(model_DF)

#Very hard to interpret - Let's plot and see
rpart.plot(model_DF)

#Let's see the results
pred_DF <- predict(model_DF,newdata=dataset,type="class")
table(dataset$good_tag,pred_DF)

(91+34)/(91+34+126+1348)

# Massive resduction in error rate to 7.817%


#Random forest

#Modelling - Basic Random Forest classifier 
set.seed(1234)
model_RF <- randomForest(factor(good_tag) ~ . -quality, dataset, ntree = 1000)
model_RF

(97+31)/(97+31+120+1351)

#Although it has a higher error classification rate than the decision tree model, a random forest does a better job of not overfitting the data
#Hence it is a better model

#Determining importance of each of the variables
imp <- importance(model_RF)
imp_table <- data.frame(Variables = row.names(imp), Importance = round(imp[,"MeanDecreaseGini"],2))

#Ranking the importance
rank <- imp_table %>%
  mutate(Rank = paste0("Rank ", dense_rank(desc(Importance))))

#Results
ggplot(rank, aes(x = reorder(Variables, Importance), y = Importance, fill = Importance)) +
  geom_bar(stat = "identity") +
  geom_text(aes(x = Variables, y = 2, label = Rank),
            hjust = 0, colour = "white") +
  labs(x = "Variables") +
  ggtitle("Importance of variables used in the model") +
  coord_flip() +
  theme_base()
