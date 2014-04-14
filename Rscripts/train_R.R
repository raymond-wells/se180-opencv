library(randomForest)
algo <- commandArgs()[length(commandArgs())]

ts.data <- read.csv(paste("data/training_set_",algo,".csv",sep=""), header=FALSE)
print("Training Random Forest Model")
ts.rfmodel <- randomForest(V1 ~ ., data=ts.data)
save(ts.rfmodel, file=paste("data/ts.rfmodel.",algo,".rda",sep=""))
print("Random forest stats:")
print(ts.rfmodel)

