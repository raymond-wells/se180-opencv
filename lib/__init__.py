from training_set_builder import TrainingSetBuilder;
from algorithmfactory import AlgorithmFactory;
import sys
from os import path
import os
import csv

RECON_HELP = """
usage: recon <operation> [...]

Runs the recon program on a set of images.  A random forest model is used to provide final classification.

Operations:
    buildset <training_definitions.csv> <algo>
    train_R   <algo>                               Train Random Forest
    test_R  <algo>                                 Test Random Forest
    classify_R <image.orb>                         Classify with Random Forest

    Misc:
    vectorize  <image.png> <algo>                  Feature extract image

training_definitions:  A csv file containing the following:
objectname, /path/to/image.png

trainingset.csv: The output from buildset

The files for each model (.svm and .forest) are required BEFORE classifying.
Algorithms supported:  orb and brisk
Use orb if you want speed and (a bit) less accuracy
Use brisk if you want more accuracy, but less speed.
        """

def build_training_set(defname, algo):
    fac = AlgorithmFactory()
    builder = TrainingSetBuilder(defname, algo,
                                 fac.get(algo)())
    builder.build_training_set()

def classify_R(orb_file,algo):
    os.system("Rscript Rscripts/classify_R.R '"+orb_file+"' " + algo)

def train_R(algo):
    os.system("Rscript Rscripts/train_R.R " + algo)

def vectorize(image, algo):
    fac = AlgorithmFactory()
    vec = fac.get(algo)()
    orbfile = path.join(path.dirname(image), path.splitext(
        path.basename(image))[0]+'.orb')
    with open(orbfile, "w") as output_file:
            writer = csv.writer(output_file)
            desc = vec.preprocess(image)
            if (vec.needs_bof):
                for feat in desc:
                    writer.writerow(feat)
                print("Now processing with ruby...")
                os.system("ruby lib/histogram.rb '" + orbfile + "' orb")
            else:
                writer.writerow(desc)
def main(argv = None):
    if len(argv) < 2:
        print RECON_HELP
        return 1

    if argv[1] == "buildset":
        build_training_set(argv[2], argv[3])
    elif argv[1] == "train":
        print ":( no training yet."
    elif argv[1]=="vectorize":
        vectorize(argv[2], argv[3])
    elif argv[1]=="train_R":
        train_R(argv[2])
    elif argv[1]=="classify_R":
        classify_R(argv[2],argv[3])
    elif argv[1]=='test_R':
        print("This has not yet been implemented.  Stay tuned folks!")
#    elif argv[1]=="train_svm":
#        os.system("Rscript Rscripts/train_SVM.R "+argv[2])
#    elif argv[1]=="classify_svm":
#        os.system("Rscript Rscripts/classify_SVM.R '"+argv[2]+"' " + argv[3])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
