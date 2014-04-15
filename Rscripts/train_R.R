library(randomForest)
algo <- commandArgs()[length(commandArgs())]
print(paste("Training algorithm:",algo))
ts.data <- read.csv(paste("data/training_set_",algo,".csv",sep=""), header=FALSE)
if (algo=="v1l") {  # When BOF cannot be used, reduce the feature set with Boruta.
    print("Feature-happy algorithm detected...")
    print("Reducing dimensionality...")
    resp = as.factor(ts.data$V1)
    set = ts.data[2,ncol(ts.data)]
    ## Use Ferns, or else this will take a century... (and probably consume all the RAM in the universe)
    ts.boruta <- Boruta(x=set, y=resp, doTrace=2, getImp=getImpFerns)
    ts.rfmodel <- randomForest(ts.data[,getSelectedAttributes(ts.boruta)], ts.data)
    save(ts.boruta, file=paste("data/ts.boruta.",algo,".rda",sep=""))
    save(ts.rfmodel, file=paste("data/ts.rfmodel",algo,".rda",sep=""))
} else {           # Regular algorithms used with BOF
    print("Training Random Forest Model")
    ts.rfmodel <- randomForest(V1 ~ ., data=ts.data)
    save(ts.rfmodel, file=paste("data/ts.rfmodel.",algo,".rda",sep=""))
    print("Random forest stats:")
    print(ts.rfmodel)
}

