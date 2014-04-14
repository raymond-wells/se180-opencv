library(Boruta)
library(randomForest)

ts.data <- read.csv("data/training_set.csv", header=FALSE)
print("Training Random Forest Model")
ts.rfmodel <- randomForest(V1 ~ ., data=ts.data)
save(ts.rfmodel, file="data/ts.rfmodel.rda")
print("Random forest stats:")
print(ts.rfmodel)

