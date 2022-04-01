from layer import hotslayer
from tqdm import tqdm
from timesurface import timesurface
import numpy as np
import torch, os
import pickle

class network(object):

    def __init__(self,  name,
                        dataset_name,
                        timestr, # date of creation of the network 
                        sensor_size,
                        nb_neurons = (4,8,16), # architecture of the network (default=Lagorce2017)
                        # parameters of time-surfaces and datasets
                        tau = (1e1,1e2,1e3), #time constant for exponential decay in millisec
                        R = (2,4,8), # parameter defining the spatial size of the time surface
                        homeo = True, # parameters for homeostasis (None is no homeo rule)
                        to_record = False,
                ):
        assert len(nb_neurons) == len(R) & len(nb_neurons) == len(tau)
        
        self.name = f'{timestr}_{dataset_name}_{name}_{homeo}_{nb_neurons}_{tau}_{R}'
        nb_layers = len(nb_neurons)
        #tau = np.array(tau)*1e3 # to enter tau in ms
        self.n_pola = [nb_neurons[L] for L in range(nb_layers-1)]
        self.n_pola.insert(0,2)
        self.sensor_size = (sensor_size[0], sensor_size[1])
        self.tau = tau
        self.R = R
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        path = '../Records/networks/'+self.name+'.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                network = pickle.load(file)
            self.layers = network.layers
        else:
            self.layers = [hotslayer((2*R[L]+1)**2*self.n_pola[L], nb_neurons[L], homeostasis=homeo, device=device) for L in range(nb_layers)]
            
    def clustering(self, loader, ordering, filtering_threshold):
        p_index = ordering.index('p')
        #torch.set_default_tensor_type("torch.DoubleTensor")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            for events, target in tqdm(loader):
                for L in range(len(self.tau)):
                    all_ts, ind_filtered = timesurface(events.squeeze(0), (self.sensor_size[0], self.sensor_size[1], self.n_pola[L]), ordering, tau = self.tau[L], surface_dimensions=[2*self.R[L]+1,2*self.R[L]+1], filtering_threshold = filtering_threshold[L], device=device)
                    #network.layers[L].to(device)
                    n_star = self.layers[L](all_ts, True)
                    if ind_filtered is not None:
                        events = events[:,ind_filtered,:]
                    events[0,:,p_index] = n_star.cpu()
                    #network.layers[L].to('cpu')
                    del all_ts
                    torch.cuda.empty_cache()
        path = '../Records/networks/'+self.name+'.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
            
    def coding(self, loader, ordering, classes, filtering_threshold, training, jitter=(None,None)):
        for L in range(len(self.tau)):
            self.layers[L].homeo_flag = False
        
        p_index = ordering.index('p')
        #torch.set_default_tensor_type("torch.DoubleTensor")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if training:
            output_path = f'../Records/output/train/{self.name}_{len(loader)}_{jitter}/'
        else: output_path = f'../Records/output/test/{self.name}_{len(loader)}_{jitter}/'
        
        if os.path.exists(output_path):
            print(f'this dataset have already been processed, check at: \n {output_path}')
        else:
            for classe in classes:
                os.makedirs(output_path+f'{classe}')
            
            with torch.no_grad():
                nb = 0
                for events, target in tqdm(loader):
                    for L in range(len(self.tau)):
                        all_ts, ind_filtered = timesurface(events.squeeze(0), (self.sensor_size[0], self.sensor_size[1], self.n_pola[L]), ordering, tau = self.tau[L], surface_dimensions=[2*self.R[L]+1,2*self.R[L]+1], filtering_threshold = filtering_threshold[L], device=device)
                        #network.layers[L].to(device)
                        n_star = self.layers[L](all_ts, False)
                        if ind_filtered is not None:
                            events = events[:,ind_filtered,:]
                        events[0,:,p_index] = n_star.cpu()
                        #network.layers[L].to('cpu')
                        del all_ts
                        torch.cuda.empty_cache()
                    np.save(output_path+f'{classes[target]}/{nb}', events)
                    nb+=1