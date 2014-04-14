library(randomForest)
input_file = commandArgs()[length(commandArgs())]
inp <- read.csv(input_file, header=FALSE)
load("data/ts.rfmodel.rda")
predict(ts.rfmodel, inp)
