library(randomForest)
library(doMC)
input <- read.csv("training_set.csv", header=FALSE)

registerDoMC(4)

model <- randomForest(V1 ~ ., data=input, ntree=ntree)

save(model, "model.rda")

