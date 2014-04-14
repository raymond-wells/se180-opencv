library(rminer)
data <- read.csv("training_set.csv")
print("Performing Fit w/ feature selection")
model = fit(x=data, model = "svm", task = "class", search="heuristic",
    mpar=NULL, feature = c("sabsg", -1, 5,"all", 0))
print("Machine learning done!  Let's save that model.")
savemodel(model, "images.model")
