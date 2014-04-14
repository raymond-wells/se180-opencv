library(e1071)
input_file = commandArgs()[length(commandArgs())]
inp <- read.csv(input_file, header=FALSE)
load("data/ts.svmmodel.rda")
print(predict(ts.svmmodel, inp))
