import tonic, torch, os, copy
from hots.network import network
from hots.utils import apply_jitter, get_loader, get_sliced_loader, make_histogram_classification, HOTS_Dataset, fit_mlr, predict_mlr, score_classif_events
import numpy as np
from tqdm import tqdm

def run_jitter(min_jitter, max_jitter, jitter_type, hots, hots_nohomeo, dataset_name, trainset_output, filtering_threshold = None, kfold = None, nb_trials = 10, nb_points = 20, fitting = True, figure_name = None, verbose = False):
    
    slicing_time_window = 1e6
    ts_batch_size = 50000
    
    initial_name = copy.copy(hots.name)
    initial_name_nohomeo = copy.copy(hots_nohomeo.name)
    
    n_classes = len(trainset_output.classes)
    n_output_neurons = len(hots.layers[-1].cumhisto)
    ts_size = [trainset_output.sensor_size[0],trainset_output.sensor_size[1],n_output_neurons]
    
    type_transform = tonic.transforms.NumpyAsType(int)
    
    if not os.path.exists('../Records/jitter_results/'):
        os.mkdir('../Records/jitter_results/')
    if jitter_type=='temporal':
        std_jit_t = np.logspace(min_jitter,max_jitter,nb_points)
        jitter_values = std_jit_t
    else:
        std_jit_s = np.linspace(min_jitter,max_jitter,nb_points)
        var_jit_s = std_jit_s**2
        jitter_values = var_jit_s

    jitter_path = f'../Records/jitter_results/{initial_name}_{nb_trials}_{min_jitter}_{max_jitter}_{kfold}_{nb_points}'

    if not os.path.exists(jitter_path+'.npz'):

        torch.set_default_tensor_type("torch.DoubleTensor")

        for trial in tqdm(range(nb_trials)):
            for ind_jit, jitter_val in enumerate(jitter_values):
                if jitter_val==0:
                    jitter = (None,None)
                else:
                    if jitter_type=='temporal':
                        jitter = (None,jitter_val)
                    else:
                        jitter = (jitter_val,None)
                        
                hots.name = initial_name+f'_{trial}'
                hots_nohomeo.name = initial_name_nohomeo+f'_{trial}'

                if jitter_type=='temporal':
                    temporal_jitter_transform = tonic.transforms.TimeJitter(std = jitter_val, clip_negative = True, sort_timestamps = True)
                    transform_full = tonic.transforms.Compose([temporal_jitter_transform, type_transform])
                else:
                    spatial_jitter_transform = tonic.transforms.SpatialJitter(sensor_size = trainset_output.sensor_size, variance_x = jitter_val, variance_y = jitter_val, clip_outliers = True)
                    transform_full = tonic.transforms.Compose([spatial_jitter_transform, type_transform])
                    
                if dataset_name=='poker':
                    testset = tonic.datasets.POKERDVS(save_to='../../Data/', train=False, transform=transform_full)
                    testloader = get_loader(testset, kfold = kfold)
                elif dataset_name=='nmnist':
                    testset = tonic.datasets.NMNIST(save_to='../../Data/', train=False, transform=transform_full)
                    testloader = get_loader(testset, kfold = kfold)
                elif dataset_name=='gesture':
                    testset = tonic.datasets.DVSGesture(save_to='../../Data/', train=False)#, transform=transform_full)
                    testloader = get_sliced_loader(testset, slicing_time_window, dataset_name, False, transform=transform_full, only_first=True, kfold=kfold)
                    
                #events, target = next(iter(testloader))
                print(f'Trial number {trial} - jitter indice {ind_jit}')
                    
                hots.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter = jitter, filtering_threshold = filtering_threshold, ts_batch_size=ts_batch_size, verbose=False)
                hots_nohomeo.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter=jitter, filtering_threshold=filtering_threshold, ts_batch_size=ts_batch_size, verbose=False)


print(f'Tonic version installed -> {tonic.__version__}')

print(f'Number of GPU devices available: {torch.cuda.device_count()}')
for N_gpu in range(torch.cuda.device_count()):
    print(f'GPU {N_gpu+1} named {torch.cuda.get_device_name(N_gpu)}')
    
device = "cuda"
    
kfold_test = None
kfold_clust = 10
ts_batch_size = 20000

dataset_name = 'gesture'
slicing_time_window = 1e6

type_transform = tonic.transforms.NumpyAsType(int)
trainset = tonic.datasets.DVSGesture(save_to='../../Data/', train=True, transform=type_transform)
testset = tonic.datasets.DVSGesture(save_to='../../Data/', train=False, transform=type_transform)
loader = get_sliced_loader(trainset, slicing_time_window, dataset_name, True, only_first=True, kfold=kfold_clust)
trainloader = get_sliced_loader(trainset, slicing_time_window, dataset_name, True, only_first=True, kfold=kfold_test)
num_sample_train = len(trainloader)
testloader = get_sliced_loader(testset, slicing_time_window, dataset_name, False, only_first=True, kfold=kfold_test)
num_sample_test = len(testloader)
n_classes = len(testset.classes)
print(f'number of samples in the training set: {len(trainloader)}')
print(f'number of samples in the testing set: {len(testloader)}')

name = 'homeohots'
homeo = True
timestr = '2022-04-22'
dataset_name = 'gesture'

Rz = [4, 8]
N_neuronz = [16, 32]
tauz = [5e4*2, 5e4*16]

hots = network(name, dataset_name, timestr, trainset.sensor_size, nb_neurons = N_neuronz, tau = tauz, R = Rz, homeo = homeo)

initial_name = hots.name

name_nohomeo = 'hots'
hots_nohomeo = network(name, dataset_name, timestr, trainset.sensor_size, nb_neurons = N_neuronz, tau = tauz, R = Rz, homeo = False)

initial_name_nohomeo = hots_nohomeo.name

filtering_threshold = [2*Rz[L] for L in range(len(Rz))]
if not os.path.exists('../Records/'):
    os.mkdir('../Records/')
    os.mkdir('../Records/networks/')
path = '../Records/networks/'+hots.name+'.pkl'
if not os.path.exists(path):
    hots.clustering(loader, trainset.ordering, filtering_threshold = filtering_threshold)
#path_nohomeo = '../Records/networks/'+hots_nohomeo.name+'.pkl'
#if not os.path.exists(path_nohomeo):
#    hots_nohomeo.clustering(loader, trainset.ordering, filtering_threshold = filtering_threshold)

jitter = (None, None)
num_workers = 0
learning_rate = 0.0001
beta1, beta2 = 0.9, 0.999
betas = (beta1, beta2)
num_epochs = 2 ** 5 + 1
N_output_neurons = N_neuronz[-1]
ts_size = (trainset.sensor_size[0],trainset.sensor_size[1],N_output_neurons)
tau_cla = 5e5
drop_proba = .9

train_path = f'../Records/output/train/{hots.name}_{num_sample_train}_{jitter}/'
test_path = f'../Records/output/test/{hots.name}_{num_sample_test}_{jitter}/'
model_path = f'../Records/networks/{hots.name}_{tau_cla}_{learning_rate}_{betas}_{num_epochs}_{jitter}.pkl'
results_path = f'../Records/LR_results/{hots.name}_{tau_cla}_{learning_rate}_{betas}_{num_epochs}_{jitter}.pkl'

kfold_jitter = 3
nb_trials = 3
nb_points = 20

trainset_output_jitter = HOTS_Dataset(train_path, trainset.sensor_size, trainset.classes, dtype=trainset.dtype, transform=type_transform)

standard_temporal_jitter_min = 3
standard_temporal_jitter_max = 7
run_jitter(standard_temporal_jitter_min, standard_temporal_jitter_max, 'temporal', hots, hots_nohomeo, dataset_name, trainset_output_jitter, kfold = kfold_jitter, nb_trials = nb_trials, nb_points = nb_points, filtering_threshold = filtering_threshold, fitting = False)