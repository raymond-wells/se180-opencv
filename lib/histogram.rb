require './lib/histogram_lib.rb'

input = ARGV.shift
output_histogram = input.split(".")[0]+".hst"
hist = ComputeHistogram.new("data/centers.csv").compute(input)
CSV.open(output_histogram, "w") {|csv| csv<<["NA",hist].flatten}

