#Using Multi-class and One-vs-Rest SVMs to Classify Pokemon by their Primary Type
#by: Tyler Nelson

rm(list = ls())

library(EBImage) #used for image processing
library(OpenImageR) #used for image processing
library(reshape2) #used for data cleaning
library(ggplot2) #used for data visualization
library(dplyr) #used for data cleaning
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(grid) #used for tuning SVM algorithm
library(gridExtra) #used for tuning
library(doBy) #used for data cleaning
library(imager) #core image processing functions
library(countcolors) #color counting functions
library(magick) #image processing functions
library(e1071) #Used for Principal Component Analysis
library(caTools) #other modeling functions tested
library(caret) #svm tuning functions
library(Morpho) #other image processing functions

#####

##Load Labels
setwd("H:/apps/xp/Desktop")
labels <- read_csv("pokemon.csv")

#Extract file names of .png Pokemon images
filenames <- list.files(path = "H:/apps/xp/Desktop/images/images/", pattern="*.png")

tib <- as_tibble(filenames)

#Load Images
setwd("H:/apps/xp/Desktop/images/images")

pics16 <- tib %>% 
  filter(str_detect(path_ext(value), 
                    fixed("png", ignore_case = TRUE))) %>% 
  mutate(data = map(value, load.image)) 

#Clean up titles of names
pics16$value <- gsub("\\....","",pics16$value)#,0, str_locate("[.]"))

#Join file name with labels
pics16 <- left_join(pics16, labels, by=c("value"="Name"))

pics <- bind_rows(pics16)

#Scale images down to 60x60 images for speed purposes
pics$data<- lapply(pics$data, resize_halfXY)

#Augment data
pics_rot30 <- list()
pics_rotneg30 <- list()
pics_mirror <- list()
pics_flip <- list()
pics_shift10u <- list()
pics_shift10d <- list()
pics_shift10l <- list()
pics_shift10r <- list()

pics_rot30 <- lapply(pics$data, rotate_xy, angle= 30,cx=60,cy=60)
pics_rotneg30 <- lapply(pics$data, rotate_xy, angle= -30,cx=60,cy=60)
pics_mirror <- lapply(pics$data, imager::mirror, axis="y")
pics_flip <- lapply(pics$data, imager::mirror, axis="x")
pics_shift10u<- lapply(pics$data, imshift, delta_y=10)
pics_shift10d <- lapply(pics$data, imshift, delta_y=-10)
pics_shift10l <- lapply(pics$data, imshift, delta_x=-10)
pics_shift10r<- lapply(pics$data, imshift, delta_x=10)

#Merge augmented data with original data
data_aug <- list()
data_aug$data <- rbind(pics$data, pics_rot30, pics_rotneg30, pics_mirror, pics_flip, pics_shift10u, pics_shift10d,pics_shift10r, pics_shift10l)


#Save down image files as vectors

pics_vector <- list()
for(n in 1:length(data_aug$data)){
pics_vector[[n]]<-as.vector(data_aug$data[[n]])
}

#Turn vectors into a matrix
pics_mat <- do.call(rbind,pics_vector)

pics_mat <- rbind(pics_mat)

#Clean the names of the types to all be lowercase
pics_df <-data.frame(type = tolower(pics$Type1), pics_mat)

#Remove the "A" alpha layer of .png images, only keeping the RGB channels + type tag
pics_df <- pics_df[,1:(length(pics_df)*.75+1)]

#Run PCA on the small images, keeping the top 70 Principal Components (80% of the variance)
pca_small <-  prcomp(pics_df[,2:length(pics_df)], center = T, scale = F, rank=70)

#Save down to save ~20 minutes in runtime, given PCA has an O(mp^2n+p^3) algorithmic complexity
#save(pca_small, file = "pca_small.RData")
#load('pca_small.RData')
summary(pca_small)
screeplot(pca_small)

data_reduced <- data.frame(type=pics_df$type, pca_small$x)

set.seed(804)
sample <- sample.split(data_reduced$type, SplitRatio = .8)
train <- subset(data_reduced, sample==T)
test <- subset(data_reduced, sample == F)

##Tune multiclass model with 10-fold cross validation across numerous gamma & cost variables
tuned_parameters <- tune.svm(type~., data = train, gamma = c(.05,1), cost = c(.1,.5,1))
summary(tuned_parameters)

model_multiclass <- svm(formula = type ~., data=train, gamma=2, cost=0.01)
plot(model_multiclass, data=train, PC1 ~ PC2)

#How'd we do on multi-class SVM?
#Train - 100%
confusionMatrix(train$type, predict(model_multiclass, newdata=train))
#Test - 14.5% (just guessing Water)
confusionMatrix(test$type, predict(model_multiclass, newdata=test))

model_svm <- list()
#load("model_svm.Rdata")
train_f <- list()
test_f <- list()
res_svm <- list()
res_svm_train <-list()
res_svm_test <- list()


#Calculate all 18 One vs Rest SVMs, by one-hot encoding each type
for( t in levels(pics_df$type)){
  train_f[[t]] <- train
  train_f[[t]]$type <- as.numeric(train$type==t)
  test_f[[t]] <- test
  test_f[[t]]$type <- as.numeric(test$type==t)
  model_svm[[t]] <- best.svm(formula = factor(type) ~., data =train_f[[t]],gamma=c(0.05,0.2,0.5,1,1.5,2), cost=c(0.01,1,4,10,20,30,60), probability=T)
  res_svm_train[[t]] <- predict(model_svm[[t]], newdata = train_f[[t]], type="class", probability=T)
  res_svm_test[[t]] <- predict(model_svm[[t]], newdata = test_f[[t]], type="class", probability =T)
}
#save(model_svm, file = "model_svm.RData")

#Plot "water" one-vs-rest SVM
plot(model, data=train_f[['water']], PC11 ~ PC12)


#Training Data
tst <- as.data.frame(res_svm_train)
#tst_scale <- scale(tst)
tst2 <- colnames(tst)[apply(tst, 1,which.max)]

#How'd we do on the training data?
confusionMatrix(train$type, factor(tst2, levels=unique(train$type)))

#Test Data
tst <- as.data.frame(res_svm_test)
#tst_scale <- scale(tst)
tst2 <- colnames(tst)[apply(tst, 1,which.max)]

#How'd we do on the test data?
confusionMatrix(test$type, factor(tst2, levels=unique(train$type)))



#Visualize misclassified test images
incorrect <- which(test$type != factor(tst2))

test_im <- subset(pics_df, sample==F)[incorrect,]

dev.off()
par(mar=c(.1,.1,.1,.1))
layout(matrix(1:35,nr=5),1,1)

for(i in 1:35){
  test_im[i,2:10801]%>%
    as.numeric()%>%
    array(c(60,60,1,3))%>%
    as.cimg()%>%
    plot(axes=F, main=paste("","",test$type[incorrect[i]], factor(tst2)[incorrect[i]], sep="\n"))
}



