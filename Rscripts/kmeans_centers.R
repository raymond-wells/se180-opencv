comb <- read.csv("data/training_set/combined.csv")
km <- kmeans(comb, 200, iter.max=25)
write.csv(km$centers, file="data/centers.csv")
