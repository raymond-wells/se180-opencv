library(randomForest)
input_file = commandArgs()[length(commandArgs())-1]
algo = commandArgs()[length(commandArgs())]
inp <- read.csv(input_file, header=FALSE)
load(paste("data/ts.rfmodel",algo,".rda"))
predict(ts.rfmodel, inp)
