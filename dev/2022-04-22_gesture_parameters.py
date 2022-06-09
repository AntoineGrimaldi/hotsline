import tonic, torch, os
from hots.utils import get_loader, get_sliced_loader, get_dataset_info, HOTS_Dataset, make_histogram_classification
from hots.network import network
from hots.timesurface import timesurface

print(f' Tonic version installed -> {tonic.__version__}')

transform = tonic.transforms.NumpyAsType(int)
dataset = tonic.datasets.DVSGesture(save_to='../../Data/', train=True, transform=transform)
#get_dataset_info(dataset, properties = ['time', 'mean_isi', 'nb_events']);
print(f'number of samples in the dataset: {len(dataset)}')

trainset = tonic.datasets.DVSGesture(save_to='../../Data/', train=True)
testset = tonic.datasets.DVSGesture(save_to='../../Data/', train=False)

transform = tonic.transforms.NumpyAsType(int)

name = 'homeohots'
homeo = True
timestr = '2022-04-22'
dataset_name = 'gesture'

R_first = [4]#[2,4,8]
N_layers = [2]#,3,4]
n_first = [16]#[8,16]
tau_first = [3e4,4e4,5e4,1e5]#[.1e3,.2e3,.5e3,1e3,2e3,5e3,1e4]

slicing_time_window = 1e6

loader = get_sliced_loader(trainset, slicing_time_window, dataset_name, True, only_first=True, kfold=10)
trainloader = get_sliced_loader(trainset, slicing_time_window, dataset_name, True, only_first=True, kfold=2)
num_sample_train = len(trainloader)
testloader = get_sliced_loader(testset, slicing_time_window, dataset_name, False, only_first=True, kfold=2)
num_sample_test = len(testloader)

for lay in N_layers:
    for R in R_first:
        for tau in tau_first:
            for N_neuron in n_first:
                Rz = [R*2**Nl for Nl in range(lay)]
                N_neuronz = [N_neuron*2**Nl for Nl in range(lay)]
                N_pola = N_neuronz.copy()
                N_pola.insert(0,2)
                tauz = [tau*N_pola[Nl] for Nl in range(lay)]
                hots = network(name, dataset_name, timestr, trainset.sensor_size, nb_neurons = N_neuronz, tau = tauz, R = Rz, homeo = homeo)
                filtering_threshold = [2*Rz[L] for L in range(len(Rz))]
                #clustering
                print('clustering')
                loader = get_sliced_loader(trainset, slicing_time_window, dataset_name, True, only_first=True, kfold=10)
                hots.clustering(loader, trainset.ordering, filtering_threshold)
                #training
                print('training')
                hots.coding(trainloader, trainset.ordering, trainset.classes, training=True, filtering_threshold = filtering_threshold)
                #testing
                print('testing')
                #loader = get_sliced_loader(testset, slicing_time_window, dataset_name, False, only_first=True, kfold=2)
                #num_sample_test = len(loader)
                hots.coding(testloader, trainset.ordering, trainset.classes, training=False, filtering_threshold = filtering_threshold)
                
                jitter = (None, None)
                train_path = f'../Records/output/train/{hots.name}_{num_sample_train}_{jitter}/'
                test_path = f'../Records/output/test/{hots.name}_{num_sample_test}_{jitter}/'

                testset_output = HOTS_Dataset(test_path, trainset.sensor_size, trainset.classes, trainset.dtype, transform=transform)
                trainset_output = HOTS_Dataset(train_path, trainset.sensor_size, trainset.classes, trainset.dtype, transform=transform)
                
                score = make_histogram_classification(trainset_output, testset_output, N_neuronz[-1]) 
                print(f'Parameters -> tau:{tau} - n_layers:{lay} - R:{R} - n_neurons:{N_neuron}')
                print(f' Accuracy: {score*100}%')
