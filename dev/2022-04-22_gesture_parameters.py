import tonic, torch, os
from utils import get_sliced_loader, get_dataset_info
from utils import make_histogram_classification, HOTS_Dataset
from network import network
from timesurface import timesurface

print(f' Tonic version installed -> {tonic.__version__}')

transform = tonic.transforms.NumpyAsType(int)
trainset = tonic.datasets.DVSGesture(save_to='../../Data/', train=True, transform=transform)
testset = tonic.datasets.DVSGesture(save_to='../../Data/', train=False, transform=transform)

name = 'homeohots'
homeo = True
timestr = '2022-04-22'
dataset_name = 'gesture'

R_first = [2]#, 8]
N_layers = [3]
n_first = [16]
tau_first = [5e3,1e4,2e4]

mean_nb_events = 4e5/6

slicing_time_window = 1e6

for lay in N_layers:
    for R in R_first:
        for tau in tau_first:
            for N_neuron in n_first:
                Rz = [R*2**Nl for Nl in range(lay)]
                N_neuronz = [N_neuron*2**Nl for Nl in range(lay)]
                N_pola = N_neuronz.copy()
                N_pola.insert(0,2)
                tauz = [tau*N_pola[Nl] for Nl in range(lay)]
                memory_network = 0
                for nl in range(lay):
                    mem_ts = (2*Rz[nl]+1)**2*N_pola[nl]*4
                    memory_time_surfaces = mean_nb_events*mem_ts
                    memory_network += mem_ts*N_neuronz[nl]
                    print(f'Estimated memory to allocate for time surfaces of layer L{nl+1} -> {memory_time_surfaces*1e-9} Gbit')
                print(f'Estimated memory to allocate for the network -> {memory_network*1e-9}')
                print(f'Total -> {(memory_network+memory_time_surfaces)*1e-9}')
                
                hots = network(name, dataset_name, timestr, trainset.sensor_size, nb_neurons = N_neuronz, tau = tauz, R = Rz, homeo = homeo)
                filtering_threshold = [2*Rz[L] for L in range(len(Rz))]
                #clustering
                print(hots.name)
                print('clustering')
                loader = get_sliced_loader(trainset, slicing_time_window, dataset_name, True, only_first=True, kfold=20)
                hots.clustering(loader, trainset.ordering, filtering_threshold)
                #training
                print('training')
                loader = get_sliced_loader(trainset, slicing_time_window, dataset_name, True, only_first=True, kfold=3)
                num_sample_train = len(loader)
                hots.coding(loader, trainset.ordering, trainset.classes, filtering_threshold, training=True)
                #testing
                print('testing')
                loader = get_sliced_loader(testset, slicing_time_window, dataset_name, False, only_first=True, kfold=2, kfold_ind=1)
                num_sample_test = len(loader)
                hots.coding(loader, trainset.ordering, trainset.classes, filtering_threshold, training=False)
                
                #jitter = (None, None)
                #train_path = f'../Records/output/train/{hots.name}_{num_sample_train}_{jitter}/'
                #test_path = f'../Records/output/test/{hots.name}_{num_sample_test}_{jitter}/'

                #testset_output = HOTS_Dataset(test_path, trainset.sensor_size, transform=tonic.transforms.NumpyAsType(int))
                #trainset_output = HOTS_Dataset(train_path, trainset.sensor_size, transform=tonic.transforms.NumpyAsType(int))
                
                #score = make_histogram_classification(trainset_output, testset_output, N_neuronz[-1]) 
                #print(f' Accuracy: {score*100}%')
