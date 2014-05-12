algo <- commandArgs()[length(commandArgs())]
comb <- read.csv("data/training_set/combined.csv", header=FALSE)
km <- kmeans(comb, 100)
write.csv(km$centers, file=paste("data/centers.csv"))
