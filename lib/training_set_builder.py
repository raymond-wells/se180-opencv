# -*- coding: utf-8 -*-
import csv
from os import path
import os
flatten = lambda arr: reduce(lambda x, y: ((isinstance(y, (list, tuple)) or
    x.append(y)) and x.extend(flatten(y))) or x, arr, [])

# Extracts features from an images in the training set and writes them to a
# comma-delimeted format.


class TrainingSetBuilder:
    def __init__(self, training_desc, output_file, vec):
        self.training_desc = training_desc
        self.vectorizer = vec
        self.output_file = output_file

    def build_training_set(self):
        self.build_csv()

    def build_csv(self):
            os.system("mkdir -p data/training_set")
            with open(self.training_desc, 'rb') as set_descriptor_file:
                set_descriptor_reader = csv.reader(set_descriptor_file,
                  delimiter=',', quotechar='|')

                for row in set_descriptor_reader:
                    image_file = path.join(
                      path.dirname(path.abspath(self.training_desc)),
                      row[1])
                    desc = self.vectorizer.preprocess(image_file)
                    orbfile = "data/training_set/" + row[1]. split('.')[0] + ".orb"
                    if desc==None:
                            print("Skipping "+orbfile+"because we had a hard" +
                              "time getting any features. :(")
                    else:
                        with open(orbfile, "w") as output_file:
                            output = csv.writer(output_file, delimiter=',',
                            quotechar='|')
                            for feat in desc:
                                output.writerow(feat)
            os.system("cat data/training_set/*.orb" +
                    " > data/training_set/combined.csv")
            print ("Calculating Centers.... (R)")
            os.system("Rscript Rscripts/kmeans_centers.R")
            print ("Now converting the training set (Ruby)...")
            os.system("ruby lib/histogram_trainingset.rb")
            print ("done!")

