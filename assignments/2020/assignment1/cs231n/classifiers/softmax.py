from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # loss_i = -(f_yi + stability_constant) + log(sum_j(exp(f_j + stability_constant))) 
    
    num_samples = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_samples):
        # 1 x D * D x K -> 1 x K
        # logits
        Xi = X[i]
        
        # Scores 1 x K
        scores = Xi.dot(W)
        
        # numerical stability
        stability_constant = -np.max(scores)
        
        # exponentiate the scores for the sum
        scores_exp = np.exp(scores + stability_constant)
        scores_exp_sum = scores_exp.sum()
        
        # take the log of the sum
        scores_sum = np.log(scores_exp_sum)
        
        loss += -scores[y[i]] - stability_constant + scores_sum
        
        ##### Gradient #####
        
        
        for j in range(num_classes):
            # calculate the fraction for the gradient
            scores_exp_frac = scores_exp[j] / scores_exp_sum

            if j != y[i]:
                dW[:, j] += Xi * scores_exp_frac
            else:
                dW[:, j] += Xi * (scores_exp_frac - 1)
        
        
    loss /= num_samples
    loss += reg * np.sum(W * W)
    
    dW /= num_samples
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]
    num_classes = W.shape[1]
    
    # N x K
    scores = X.dot(W)
    stability_constants = np.max(scores, axis=1)[:,None]
    scores_exp = np.exp(scores + stability_constants)
    scores_exp_sum = scores_exp.sum(axis=1)[:, None]    
    scores_sum = np.log(scores_exp_sum)

    fs = scores[range(len(y)), y, None]
    
    scores_sum -= fs
    scores_sum -= stability_constants
    
    loss = scores_sum.sum()
    loss /= num_samples
    loss += reg * np.sum(W * W)    
    
    
    
    # Gradient
    # N x K
    scores_exp_frac = scores_exp / scores_exp_sum

    # DxN @ NxK = DxK
    dW = X.T @ scores_exp_frac
    
    true_lab_mask = np.zeros((num_samples, num_classes))
    true_lab_mask[range(len(y)), y] = 1
    dW -= X.T @ true_lab_mask
    
    dW /= num_samples
    dW += reg * W

    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
