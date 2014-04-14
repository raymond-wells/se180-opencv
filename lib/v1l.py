
import scipy as N
from copy import deepcopy
from v1like import v1s
class V1LikeAlgorithm:
    def __init__(self):
        ## Credit to Nicolas Pinto
        protocol = {
            # number of training examples
            'ntrain':15,
            # number of testing examples
            'ntest':30,
            # number of trials
            'ntrials':10,
            # random seed
            'seed':1,
        }
        
        # -- representation 
        # some filter parameters
        norients = 16
        orients = [ o*N.pi/norients for o in xrange(norients) ]
        divfreqs = [2, 3, 4, 6, 11, 18]
        freqs = [ 1./n for n in divfreqs ]
        phases = [0]

        # dict with all representation parameters
        representation = {
    
        # - preprocessing
            # prepare images before processing
            'preproc': {
                # resize input images by keeping aspect ratio and fix the biggest edge
                'max_edge': 150,
                # kernel size of the box low pass filter
                'lsum_ksize': 3,
            },

            # - input local normalization
            # local zero-mean, unit-magnitude
            'normin': {
                # kernel shape of the local normalization
                'kshape': (3,3),
                # magnitude threshold
                # if the vector's length is below, it doesn't get resized
                'threshold': 1.0,
            },

            # - linear filtering
            'filter': {
                # kernel shape of the gabors
                'kshape': (43,43),
                # list of orientations
                'orients': orients,
                # list of frequencies
                'freqs': freqs,
                # list of phases
                'phases': phases,
                # threshold (variance explained) for the separable convolution
                'sep_threshold': .9,
            },    

            # - simple non-linear activation
            'activ': {
                # minimum output (clamp)
                'minout': 0,
                # maximum output (clamp)
                'maxout': 1,
            },
            
            # - output local normalization
            'normout': {
                # kernel shape of the local normalization
                'kshape': (3,3),
                # magnitude threshold
                # if the vector's length is below, it doesn't get resized
                'threshold': 1.0,
            },
            
            # - dimension reduction
            'dimr': {
                # kernel size of the local sum (2d slice)
                'lsum_ksize': 17,
                # fixed output shape (only the first 2 dimensions, y and x)
                'outshape': (30,30),
            },
        }
        
        # -- featsel details what features you want to be included in the vector
        featsel = {
            # Include representation output ? True or False
            'output': True,

            # Include grayscale values ? None or (height, width)    
            'input_gray': None,
            # Include color histograms ? None or nbins per color
            'input_colorhists': None,
            # Include input norm histograms ? None or (division, nfeatures)    
            'normin_hists': None,
            # Include filter output histograms ? None or (division, nfeatures)
            'filter_hists': None,
            # Include activation output histograms ? None or (division, nfeatures)    
            'activ_hists': None,
            # Include output norm histograms ? None or (division, nfeatures)
            'normout_hists': None,
            # Include representation output histograms ? None or (division, nfeatures)
            'dimr_hists': None,
        }

        # -- model is a list of (representation, featureselection)
        # that will be combine resulting in the final feature vector
        self.model = [(representation, featsel)]
    def preprocess(self, image_file):
        v1 = v1s.V1S()
        features = v1._generate_repr(image_file, self.model)
        return features
        
        
