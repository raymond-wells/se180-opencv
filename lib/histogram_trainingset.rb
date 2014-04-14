require './lib/histogram_lib.rb'

## Compute the histogram of the entire training set...
chistogram = ComputeHistogram.new("data/centers.csv")
output_file = CSV.open("data/training_set.csv", "w")
`find data/training_set/ -iname '*.orb'`.split("\n").each do |orb_file|
  puts "Processing #{orb_file}..."
  hist = chistogram.compute(orb_file)
  output_file << [File.basename(orb_file).split(/[_\.]/)[0],hist].flatten
end
