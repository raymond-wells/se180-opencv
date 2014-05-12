library(randomForest)
algo <- commandArgs()[length(commandArgs())]
print(paste("Training algorithm:",algo))
## Let's forget about V1-Like for now and focus on this.

ts.data <- read.csv("data/training_set.csv", header=FALSE)
    print("Training Random Forest Model")
    ts.rfmodel <- randomForest(V1 ~ ., data=ts.data, ntrees=1500)
    save(ts.rfmodel, file=paste("data/ts.rfmodel.",algo,".rda",sep=""))
    print("Random forest stats:")
    print(ts.rfmodel)

