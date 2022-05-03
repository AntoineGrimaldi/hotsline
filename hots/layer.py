import torch

class hotslayer(torch.nn.Module):
    def __init__(self, ts_size, n_neurons, homeostasis = True, device="cpu", bias=False):
        super(hotslayer, self).__init__()
        self.synapses = torch.nn.Linear(ts_size, n_neurons, bias=bias, device=device)
        torch.nn.init.uniform_(self.synapses.weight, a=0, b=1)
        self.cumhisto = torch.ones([n_neurons], device=device)
        self.homeo_flag = homeostasis
        
    def homeo_gain(self):
        lambda_homeo = .25
        gain = torch.exp(lambda_homeo*(1-self.cumhisto.size(dim=0)*self.cumhisto/self.cumhisto.sum()))
        return gain

    def forward(self, all_ts, clustering_flag):
        if clustering_flag:
            n_star = torch.zeros(all_ts.shape[0])
            for iev in range(len(all_ts)):
                ts = all_ts[iev].ravel()
                ts = ts/torch.linalg.norm(ts)
                beta = self.synapses(ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1))
                if self.homeo_flag:
                    beta_homeo = self.homeo_gain()*(self.synapses(ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1)))
                    n_star_ev = torch.argmax(beta_homeo)
                else:
                    n_star_ev = torch.argmax(beta)

                Ck = self.synapses.weight.data[n_star_ev,:]
                alpha = 0.01/(1+self.cumhisto[n_star_ev]/20000)
                self.synapses.weight.data[n_star_ev,:] = Ck + alpha*beta[n_star_ev]*(ts - Ck)
                # learning rule from Lagorce 2017
                #self.synapses[:,n_star] = Ck + alpha*(TS - simil[closest_proto_idx]*Ck)
                self.cumhisto[n_star_ev] += 1
                n_star[iev] = n_star_ev
        else:
            all_ts = torch.flatten(all_ts, start_dim=1, end_dim=- 1)
            all_ts = all_ts/torch.linalg.norm(all_ts, dim=0)
            beta = self.synapses(all_ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1))
            n_star = torch.argmax(beta, dim=1)
        return n_star
    
    
class mlrlayer(torch.nn.Module):
    
    def __init__(self, ts_size, n_classes, device='cpu', bias=True):
        super(mlrlayer, self).__init__()
        self.linear = torch.nn.Linear(ts_size, n_classes, bias=bias, device=device)
        self.nl = torch.nn.Softmax(dim=1)

    def forward(self, factors):
        return self.nl(self.linear(factors))
    
    
class snnlayer(torch.nn.Module):
    def __init__(self, ts_size, n_neurons, homeostasis = True, device="cpu", bias=False):
        super(snnlayer, self).__init__()
        self.synapses = torch.nn.Linear(ts_size, n_neurons, bias=bias, device=device)
        torch.nn.init.uniform_(self.synapses.weight, a=0, b=1)
        self.cumhisto = torch.ones([n_neurons], device=device)
        self.homeo_flag = homeostasis
        self.nl = torch.nn.Softmax(dim=0)
        self.threshold = 5/n_neurons
        
    def homeo_gain(self):
        lambda_homeo = .25
        gain = torch.exp(lambda_homeo*(1-self.cumhisto.size(dim=0)*self.cumhisto/self.cumhisto.sum()))
        return gain

    def forward(self, all_ts, clustering_flag):
        n_star = torch.zeros(all_ts.shape[0])-1
        if clustering_flag:
            for iev in range(len(all_ts)):
                ts = all_ts[iev].ravel()
                ts = ts/torch.linalg.norm(ts)
                beta = self.synapses(ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1))
                if self.homeo_flag:
                    beta_homeo = self.homeo_gain()*(self.synapses(ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1)))
                    softmax_ev = self.nl(beta_homeo)
                else:
                    softmax_ev = self.nl(beta)
                n_star_ev = torch.where(softmax_ev>self.threshold)[0]
                if len(n_star_ev)>0:
                    print(n_star_ev, iev)
                    Ck = self.synapses.weight.data[n_star_ev,:]
                    alpha = 0.01/(1+self.cumhisto[n_star_ev]/20000)
                    self.synapses.weight.data[n_star_ev,:] = Ck + alpha*beta[n_star_ev]*(ts - Ck)
                    self.cumhisto[n_star_ev] += 1
                    n_star[iev] = n_star_ev
            n_star = n_star[n_star>-1]
        else:
            all_ts = torch.flatten(all_ts, start_dim=1, end_dim=- 1)
            all_ts = all_ts/torch.linalg.norm(all_ts, dim=0)
            beta = self.synapses(all_ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1))
            softmax = self.nl(beta, dim=1)
            
        return n_star