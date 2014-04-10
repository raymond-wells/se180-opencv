# -*- coding: utf-8 -*-
import csv
from os import path
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
        with open(self.output_file, 'w') as output_file:
            output = csv.writer(output_file)
            with open(self.training_desc, 'rb') as set_descriptor_file:
                set_descriptor_reader = csv.reader(set_descriptor_file,
                    delimiter=',', quotechar='|')

                for row in set_descriptor_reader:
                    image_file = path.join(
                        path.dirname(path.abspath(self.training_desc)),
                        row[1])
                    output.writerow([row[0]] +
                        self.vectorizer.vectorize(image_file))


