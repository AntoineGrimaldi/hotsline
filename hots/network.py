from hots.layer import hotslayer
from tqdm import tqdm
from hots.timesurface import timesurface
from hots.utils import entropy
import numpy as np
import matplotlib.pyplot as plt
import torch, os
import pickle

class network(object):

    def __init__(self,  name,
                        dataset_name,
                        timestr, # date of creation of the network 
                        sensor_size, # 
                        nb_neurons = (4,8,16), # architecture of the network (default=Lagorce2017)
                        # parameters of time-surfaces and datasets
                        tau = (1e1,1e2,1e3), #time constant for exponential decay in millisec
                        R = (2,4,8), # parameter defining the spatial size of the time surface
                        homeo = True, # parameters for homeostasis (None is no homeo rule)
                        to_record = False,
                        record_path = '../Records/',
                        device = 'cuda',
                ):
        assert len(nb_neurons) == len(R) & len(nb_neurons) == len(tau)
        
        self.name = f'{timestr}_{dataset_name}_{name}_{homeo}_{nb_neurons}_{tau}_{R}'
        nb_layers = len(nb_neurons)
        self.n_pola = [nb_neurons[L] for L in range(nb_layers-1)]
        self.n_pola.insert(0,2)
        self.sensor_size = (sensor_size[0], sensor_size[1])
        self.tau = tau
        self.R = R
        self.record_path = record_path
        
        path = self.record_path+'networks/'+self.name+'.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                my_network = pickle.load(file)
            self.layers = my_network.layers
            for L in range(len(self.layers)):
                self.layers[L] = self.layers[L].to(device)
        else:
            self.layers = [hotslayer((2*R[L]+1)**2*self.n_pola[L], nb_neurons[L], homeostasis=homeo, device=device) for L in range(nb_layers)]
            
    def clustering(self, loader, ordering, filtering_threshold = None, ts_batch_size = None, device = 'cuda', record = False):
        path = self.record_path+'networks/'+self.name+'.pkl'
        if not os.path.exists(path):
            p_index = ordering.index('p')
            
            for L in range(len(self.tau)):
                self.layers[L] = self.layers[L].to(device)
                self.layers[L].cumhisto = self.layers[L].cumhisto.to(device)
            
            if record:
                entropy = []
                loss = []
                delta_w = []
                homeostasis = []

            with torch.no_grad():
                
                if not filtering_threshold: filtering_threshold = [None for L in range(len(self.tau))]
                
                for events, target in tqdm(loader):
                    events = events.squeeze(0)
                    if record:
                        previous_dic = [self.layers[L].synapses.weight.data.T.detach().clone() for L in range(len(self.tau))]
                    for L in range(len(self.tau)):
                        ts_batch_size_per_layer = int(ts_batch_size*(self.n_pola[0]*(2*self.R[0]+1)**2)/(self.n_pola[L]*(2*self.R[L]+1)**2))
                        nb_batch = len(events)//ts_batch_size_per_layer
                        if nb_batch:
                            previous_timestamp = []
                            outputs = torch.Tensor([])
                            ind_outputs = torch.Tensor([])
                            for load_nb in range(nb_batch+1):
                                all_ts, ind_filtered_timesurface, previous_timestamp = timesurface(events, (self.sensor_size[L][0], self.sensor_size[L][1], self.n_pola[L]), ordering, tau = self.tau[L], surface_dimensions=[2*self.R[L]+1,2*self.R[L]+1], filtering_threshold = filtering_threshold[L], ts_batch_size = ts_batch_size_per_layer, load_number = load_nb, previous_timestamp = previous_timestamp, device = device)
                                n_star, _, beta = self.layers[L](all_ts, True)
                                outputs = torch.hstack([outputs,n_star]) if outputs.shape[0]>0 else n_star
                                ind_outputs = torch.hstack([ind_outputs,ind_filtered_timesurface+load_nb*ts_batch_size_per_layer]) if ind_outputs.shape[0]>0 else ind_filtered_timesurface
                                if record:
                                    proto_ts = all_ts.detach().clone()
                                    kernels = self.layers[L].synapses.weight.data.T
                                    DIFF = 0
                                    for ev in range(len(n_star)):
                                        proto_ts[ev,:,:,:] = torch.reshape(kernels[:,int(n_star[ev].cpu())], (self.n_pola[L], 2*self.R[L]+1, 2*self.R[L]+1))
                                        diff = torch.linalg.norm(all_ts[ev,:,:,:]-proto_ts[ev,:,:,:])
                                        DIFF += diff.mean()
                                    DIFF/=len(n_star)
                                    loss.append(DIFF.cpu())
                                    ent.append(entropy(kernels))
                                    delta_w.append((kernels-previous_dic[L]).abs().mean().cpu())
                                    homeostasis.append((self.layers[L].cumhisto/self.layers[L].cumhisto.sum()-1/kernels.shape[1]).abs().mean().cpu())
                                del all_ts
                                torch.cuda.empty_cache()
                            events = events[ind_outputs,:]
                            events[:,p_index] = outputs.cpu()
                        else:
                            all_ts, ind_filtered_timesurface = timesurface(events, (self.sensor_size[L][0], self.sensor_size[L][1], self.n_pola[L]), ordering, tau = self.tau[L], surface_dimensions=[2*self.R[L]+1,2*self.R[L]+1], filtering_threshold = filtering_threshold[L], device=device)
                            n_star, _, beta  = self.layers[L](all_ts, True)
                            if record:
                                proto_ts = all_ts.detach().clone()
                                kernels = self.layers[L].synapses.weight.data.T
                                DIFF = 0
                                for ev in range(len(n_star)):
                                    proto_ts[ev,:,:,:] = torch.reshape(kernels[:,int(n_star[ev].cpu())], (self.n_pola[L], 2*self.R[L]+1, 2*self.R[L]+1))
                                    diff = torch.linalg.norm(all_ts[ev,:,:,:]-proto_ts[ev,:,:,:])
                                    DIFF += diff.mean()
                                DIFF/=len(n_star)
                                loss.append(DIFF.cpu())
                                ent.append(entropy(kernels))
                                delta_w.append((kernels-previous_dic[L]).abs().mean().cpu())
                                homeostasis.append((self.layers[L].cumhisto/self.layers[L].cumhisto.sum()-1/kernels.shape[1]).abs().mean().cpu())
                            del all_ts
                            torch.cuda.empty_cache()
                            events = events[ind_filtered_timesurface,:]
                            events[:,p_index] = n_star.cpu()
                        if events.shape[0]==0: break

            with open(path, 'wb') as file:
                pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
            if record:
                path = self.record_path+'networks/'+self.name+'_recorded_parameters.pkl'
                with open(path, 'wb') as file:
                    pickle.dump([loss, entropy, delta_w, homeostasis], file, pickle.HIGHEST_PROTOCOL)
            
            
    def coding(self, loader, ordering, classes, training, ts_batch_size = None, filtering_threshold = None, jitter=(None,None), layer_threshold = None, device = 'cuda', verbose=True):
        
        #homeostatic gain control is used only for the clustering phase
        for L in range(len(self.tau)):
            self.layers[L].homeo_flag = False
            self.layers[L] = self.layers[L].to(device)
            self.layers[L].cumhisto = self.layers[L].cumhisto.to(device)
        
        if not filtering_threshold: filtering_threshold = [None for L in range(len(self.tau))]
        if not layer_threshold: layer_threshold = [None for L in range(len(self.tau))]
        
        p_index = ordering.index('p')
        
        if training:
            output_path = self.record_path+f'output/train/{self.name}_{len(loader)}_{jitter}/'
        else: output_path = self.record_path+f'output/test/{self.name}_{len(loader)}_{jitter}/'
        
        if os.path.exists(output_path):
            if verbose:
                print(f'this dataset have already been processed, check at: \n {output_path}')
        else:
            for classe in classes:
                os.makedirs(output_path+f'{classe}')
            
            with torch.no_grad():
                nb = 0
                for events, target in tqdm(loader):
                    events = events.squeeze(0)
                    if events.shape[0]==0: 
                        print('no event in this sample')
                    else:
                        for L in range(len(self.tau)):
                            ts_batch_size_per_layer = int(ts_batch_size*(self.n_pola[0]*(2*self.R[0]+1)**2)/(self.n_pola[L]*(2*self.R[L]+1)**2))
                            nb_batch = len(events)//ts_batch_size_per_layer
                            if nb_batch:
                                previous_timestamp = []
                                outputs = torch.Tensor([])
                                ind_outputs = torch.Tensor([])
                                for load_nb in range(nb_batch+1):
                                    all_ts, ind_filtered_timesurface, previous_timestamp = timesurface(events, (self.sensor_size[L][0], self.sensor_size[L][1], self.n_pola[L]), ordering, tau = self.tau[L], surface_dimensions=[2*self.R[L]+1,2*self.R[L]+1], filtering_threshold = filtering_threshold[L], ts_batch_size = ts_batch_size_per_layer, load_number = load_nb, previous_timestamp = previous_timestamp, device = device)
                                    n_star, ind_filtered_layer, beta = self.layers[L](all_ts, False)
                                    ind_to_keep = ind_filtered_timesurface[ind_filtered_layer]
                                    outputs = torch.hstack([outputs,n_star]) if outputs.shape[0]>0 else n_star
                                    ind_outputs = torch.hstack([ind_outputs,ind_to_keep+load_nb*ts_batch_size_per_layer]) if ind_outputs.shape[0]>0 else ind_to_keep
                                    del all_ts
                                    torch.cuda.empty_cache()
                                events = events[ind_outputs,:]
                                events[:,p_index] = outputs.cpu()
                            else:
                                all_ts, ind_filtered_timesurface = timesurface(events, (self.sensor_size[L][0], self.sensor_size[L][1], self.n_pola[L]), ordering, tau = self.tau[L], surface_dimensions=[2*self.R[L]+1,2*self.R[L]+1], filtering_threshold = filtering_threshold[L], device=device)
                                n_star, ind_filtered_layer, beta = self.layers[L](all_ts, False)
                                events = events[ind_filtered_timesurface,:]
                                events[:,p_index] = n_star.cpu()
                                events = events[ind_filtered_layer,:]
                                del all_ts
                                torch.cuda.empty_cache()
                            if events.shape[0]==0: 
                                complete_flag = False
                                break
                            else:
                                complete_flag = True

                        if complete_flag:
                            np.save(output_path+f'{classes[target]}/{nb}', events)
                        nb+=1
                    
                    
    def plotlayers(self, maxpol=None, hisiz=2, yhis=0.3):
        '''
        '''
        N = []
        P = [2]
        R2 = []
        kernels = []
        for L in range(len(self.tau)):
            kernels.append(self.layers[L].synapses.weight.data.T.cpu().numpy())
            N.append(int(kernels[L].shape[1]))
            if L>0:
                P.append(int(kernels[L-1].shape[1]))
            R2.append(int(kernels[L].shape[0]/P[L]))
        if maxpol is None:
            maxpol=P[-1]

        fig = plt.figure(figsize=(16,9))
        gs = fig.add_gridspec(np.sum(P)+hisiz, np.sum(N)+len(self.tau)-1, wspace=0.05, hspace=0.05)
        if self.layers[-1].homeo_flag:
            fig.suptitle('Unsupervised clustering with homeostasis', size=20, y=0.95)
        else:
            fig.suptitle('Unsupervised clustering for original HOTS', size=20, y=0.95)

        for L in range(len(self.tau)):
            ax = fig.add_subplot(gs[:hisiz, int(np.sum(N[:L]))+1*L:int(np.sum(N[:L+1]))+L*1])
            plt.bar(np.arange(N[L]), (self.layers[L].cumhisto/torch.sum(self.layers[L].cumhisto)).cpu(), width=1, align='edge', ec="k")
            ax.set_xticks(())
            ax.set_title('Layer '+str(L+1), fontsize=16)
            plt.xlim([0,N[L]])
            yhis = 1.1*max(self.layers[L].cumhisto/torch.sum(self.layers[L].cumhisto)).cpu()
            plt.ylim([0,yhis])

            for k in range(N[L]):
                vmaxi = max(kernels[L][:,k])
                for j in range(P[L]):
                    if j>maxpol-1:
                        pass
                    else:
                        axi = fig.add_subplot(gs[j+hisiz,k+1*L+int(np.sum(N[:L]))])
                        krnl = kernels[L][j*R2[L]:(j+1)*R2[L],k].reshape((int(np.sqrt(R2[L])), int(np.sqrt(R2[L]))))

                        axi.imshow(krnl, vmin=0, vmax=vmaxi, cmap=plt.cm.plasma, interpolation='nearest')
                        axi.set_xticks(())
                        axi.set_yticks(())
        plt.show()
        return fig
    
    def plotlearning(self, width_fig = 30):
        path = self.record_path+'networks/'+self.name+'_recorded_parameters.pkl'
        with open(path, 'rb') as file:
            loss, entropy, delta_w, homeostasis = pickle.load(file)
            
        n_layers = len(self.tau)
        fig, axs = plt.subplots(n_layers,4, figsize=(width_fig,n_layers*width_fig//4))
        for L in range(n_layers):
            loss_layer = loss[L::n_layers]
            entropy_layer = entropy[L::n_layers]
            delta_w_layer = delta_w[L::n_layers]
            homeostasis_layer = homeostasis[L::n_layers]
            axs[L,0].plot(loss_layer)
            axs[L,1].plot(entropy_layer)
            axs[L,2].plot(delta_w_layer)
            axs[L,3].plot(homeostasis_layer)
            if L == 0:
                axs[L,0].set_title('average loss')
                axs[L,1].set_title('average entropy values for the time surfaces')
                axs[L,2].set_title('average gradient of the weights')
                axs[L,3].set_title('average homeostasic gain')
        plt.show()