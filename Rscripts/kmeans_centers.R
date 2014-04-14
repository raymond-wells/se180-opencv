algo <- commandArgs()[length(commandArgs())]
comb <- read.csv(paste("data/training_set/combined_",algo,".csv"), header=FALSE)
km <- kmeans(comb, 100)
write.csv(km$centers, file=paste("data/centers",algo,".csv"))
