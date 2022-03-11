import torch



class hotslayer(torch.nn.Module):
    def __init__(self, ts_size, n_neurons, bias=False, homeostasis = True, device="cpu"):
        super(hotslayer, self).__init__()
        self.synapses = torch.nn.Linear(ts_size, n_neurons, bias=bias)
        torch.nn.init.uniform_(self.synapses.weight, a=0, b=1)
        self.cumhisto = torch.ones([n_neurons]).to(device)
        self.learning_flag = True
        self.homeo_flag = homeostasis
        
    def homeo_gain(self):
        lambda_homeo = .25
        gain = torch.exp(lambda_homeo*(1-self.cumhisto.size(dim=0)*self.cumhisto/self.cumhisto.sum()))
        return gain

    def forward(self, all_ts):
        if self.learning_flag:
            for iev in range(len(all_ts)):
                ts = all_ts[iev].ravel()
                ts = ts/torch.linalg.norm(ts)
                beta = self.synapses(ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1))
                if self.homeo_flag:
                    beta_homeo = self.homeo_gain()*(self.synapses(ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1)))
                    n_star = torch.argmax(beta_homeo)
                else:
                    n_star = torch.argmax(beta)

                Ck = self.synapses.weight.data[n_star,:]
                alpha = 0.01/(1+self.cumhisto[n_star]/20000)
                self.synapses.weight.data[n_star,:] = Ck + alpha*beta[n_star]*(ts - Ck)
                # learning rule from Lagorce 2017
                #self.synapses[:,n_star] = Ck + alpha*(TS - simil[closest_proto_idx]*Ck)
                self.cumhisto[n_star] += 1
        else:
            all_ts = torch.flatten(all_ts, start_dim=1, end_dim=- 1)
            all_ts = all_ts/torch.linalg.norm(all_ts, dim=0)
            beta = self.synapses(all_ts)/(torch.linalg.norm(self.synapses.weight.data, dim=1))
            n_star = torch.argmax(beta, dim=1)
        return n_star