require 'csv'


class ComputeHistogram
  def initialize centersfile
    centers_init = CSV.read(centersfile)
    @centers = centers_init.slice(1,centers_init.length).map{|c| c.slice(1,c.length).map{|e| e.to_f}}
  end

  def compute input
    features = CSV.read(input).map{|x| x.map{|e| e.to_f}}
    hist = @centers.map{0}
    features.each do |feat|
      dists = @centers.map{|c| compute_dist(c,feat)}
      hist[dists.index(dists.min)] += 1
    end
    hist
  end
  private
  def compute_dist p1, p2
      Math.sqrt(p1.zip(p2).map{|e| (e[0]-e[1])**2}.reduce(:+))
  end
end


