require './lib/histogram_lib.rb'

input = ARGV.shift
algorithm = ARGV.shift
output_histogram = input.split(".")[0]+"_#{algorithm}.hst"
hist = ComputeHistogram.new("data/centers_#{algorithm}.csv").compute(input)
CSV.open(output_histogram, "w") {|csv| csv<<["NA",hist].flatten}

