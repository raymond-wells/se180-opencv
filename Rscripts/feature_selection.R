library(Boruta)
ts <- na.omit(read.csv("training_set.csv",header=FALSE))
print("Feature selection begins...")
result <- Boruta(V1 ~ ., data=ts, doTrace=2)
save(result, file="features.rda")


  
