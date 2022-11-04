import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

        
class rrlyrae_classifier:
    def __init__(self, dist_rrlyrae, dist_stars):
        self.dist_rrlyrae = dist_rrlyrae
        self.dist_stars = dist_stars
        return None
    
    def get_test_data(self, test_data):
        """
        load the testdata array and the corresponding y (target, 0 or 1)
        """
        self.test_data = test_data[:, :4]
        self.y_true = test_data[:, 4].numpy()
        return None
    
    def predict(self, activation="tanh"):
        """
        evaluate the prediction
        
        :activation : {tanh, atan, logistic, erfc}:
        
        returns: y_score in range [0,1]
        """
        from scipy.special import erfc
        if activation=="tanh":
            afunc = lambda x: (np.tanh(2*x)+1)/2
        if activation=="atan":
            afunc = lambda x: 0.5+np.arctan(x)/np.pi
        if activation=="logistic":
            afunc = lambda x: 1/(1+np.exp(-x))
        if activation=="erfc":
            afunc = lambda x: erfc(-x)/2.

        # evaluate at given PDF, gives prob per input
        prob_stars = self.dist_stars.prob(self.test_data).numpy()
        prob_rrlyrae = self.dist_rrlyrae.prob(self.test_data).numpy()
        
        # log, to map them equally to (-inf, inf)
        log_prob = np.log10(prob_rrlyrae / prob_stars)
        
        # activation
        self.y_score = afunc(log_prob)
        
        return self.y_score
        
    def roc_curve(self, name=False):
        """
        plot a ROC curve of the classified data
        """
        fpr0, tpr0, thresholds = roc_curve(self.y_true, self.y_score) 
        plt.plot(fpr0, tpr0, color="navy")

        plt.title("ROC curve")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        props = dict(boxstyle='round', facecolor='orange', alpha=0.3)
        plt.text(0.5,0.1, f" Correct:{self.correct:2.2%}\n False pos.:{self.fpr:2.2%}\n True pos.:{self.tpr:2.2%}\n False neg.:{self.fnr:2.2%}\n True neg.:{self.tnr:2.2%}",size=15,bbox=props)

        plt.grid(alpha=.7, ls=":")
        if name:
            plt.savefig(name)
            
        plt.show()
        
        return None
    
    def evaluate(self, thresh=0.5):
        N_test_data = self.y_true.size
        
        self.correct = ( (self.y_score > thresh) == (self.y_true > thresh) ).sum() / N_test_data
        self.fpr = ( (self.y_score > thresh) * (self.y_true < thresh) ).sum() / N_test_data
        self.fnr = ( (self.y_score < thresh) * (self.y_true > thresh) ).sum() / N_test_data
        self.tpr = ( (self.y_score > thresh) * (self.y_true > thresh) ).sum() / N_test_data
        self.tnr = ( (self.y_score < thresh) * (self.y_true < thresh) ).sum() / N_test_data

        pos_fraction = self.y_true.sum() / self.y_true.size
        neg_fraction = (1-self.y_true).sum() / self.y_true.size

        print("The Dataset:")
        print(f"Total percentage of positive/real RR Lyrae: \t {pos_fraction:.2%}")
        print(f"Total percentage of negative/not RR Lyrae: \t{neg_fraction:.2%} \n")

        print(f"Rate of predictions\n\t- correct: \t{self.correct:2.2%}")
        print(f"\t- false pos.: \t {self.fpr:2.2%}")
        print(f"\t- true pos.: \t {self.tpr:2.2%}")
        print(f"\t- false neg.: \t {self.fnr:2.2%}")
        print(f"\t- true neg.: \t{self.tnr:2.2%}\n")

        print(f"Percentage of correctly identified \n\t- RR Lyrae: \t{self.tpr/pos_fraction:.2%}")
        print(f"\t- non-RR Lyrae:\t{self.tnr/neg_fraction:.2%}")
        
        return None