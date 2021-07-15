
class monotoneLoss():

  ## Loss function to enforce monotonicity of the neural network

  def __init__(self, feature_inds, weight_lambda):

    ## Args:
      ## feature_inds: list with the indices of the features that should be monotone
      ## weight_lambda: The scaling factor for the loss

    self.monotone_feat = feature_inds 
    self.weight_lambda = weight_lambda

  def calc_loss(self, my_net, x):

    ## Args:
      ## my_net: The neural network
      ## x: The input points where the monotonicity should be enforced
    ## Assume that input x is of the shape (batch_size, num_features)

    ## Create boolean matrix to zero out derivatives for non-monotone features
    self.zero_t = torch.zeros([x.shape[0], x.shape[1]])
    self.zero_t[:, self.monotone_feat] = 1.
    
    x.requires_grad=True
    o = my_net(x)
    o = o.view(-1)

    dx = torch.autograd.grad(o, x, create_graph=True, grad_outputs = torch.ones(o.shape[0]))[0]
    
    dx = torch.clip(-dx*self.zero_t, min = 0)
    dxloss = torch.mean(dx)

    ## Calculate partial derivative for all parameters
    for idx,w in enumerate(my_net.parameters()):      
      dx_c = torch.clone(dxloss).view(-1)
      
      dxdw = torch.autograd.grad(dx_c, w, retain_graph = True,allow_unused=True, grad_outputs = torch.ones(dx_c.shape[0]))[0]
      
      
      if not dxdw is None and not w.grad is None:
        ## If we have already added something to the gradient
        w.grad = w.grad + self.weight_lambda*dxdw
      elif not dxdw is None:
        w.grad = self.weight_lambda*dxdw
      elif not w.grad is None:
        #w.grad = w.grad + torch.zeros_like(w)
        pass
      else:
        w.grad = torch.zeros_like(w)
      
    x.requires_grad = False
