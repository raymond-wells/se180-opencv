from orbalgorithm import ORBAlgorithm;
from training_set_builder import TrainingSetBuilder;
from vectorizer import Vectorizer;
import sys
from os import path
import csv

RECON_HELP = """
usage: recon <operation> [...]

Runs the recon program on a set of images.

Operations:
    buildset <training_definitions.csv>
    train_svm <trainingset>                        Train SVM
    train_R <trainingset>                          Train Random Forest
    test_opencv <trainingset> <testset.csv>        Test SVM
    test_R <trainingset> <testset.csv>             Test Random Forest

    Misc:
    vectorize  <image.png> <algo>                  Feature extract image

training_definitions:  A csv file containing the following:
objectname, /path/to/image.png

trainingset.csv: The output from buildset

The files for each model (.svm and .forest) are required BEFORE classifying.

        """

def build_training_set(defname):
    builder = TrainingSetBuilder(defname, "training_set.csv",
        Vectorizer(ORBAlgorithm()))
    builder.build_training_set()

def vectorize(image, algo):
    vec = Vectorizer(ORBAlgorithm())
    with open(path.join(path.dirname(image), path.splitext(
        path.basename(image))[0]+".csv"), "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(vec.vectorize(image))

def main(argv = None):
    if len(argv) < 2:
        print RECON_HELP
        return 1

    if argv[1] == "buildset":
        build_training_set(argv[2])
    elif argv[1] == "train":
        print ":( no training yet."
    elif argv[1]=="vectorize":
        vectorize(argv[2], argv[3])



if __name__ == "__main__":
    sys.exit(main(sys.argv))
