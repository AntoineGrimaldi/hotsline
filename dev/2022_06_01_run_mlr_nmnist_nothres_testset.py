import tonic, torch, os, pickle
from tqdm import tqdm
from network import network
from layer import mlrlayer
from timesurface import timesurface
from utils import get_loader, make_histogram_classification, HOTS_Dataset, fit_mlr, predict_mlr, score_classif_events, score_classif_time

print(f'Tonic version installed -> {tonic.__version__}')

print(f'Number of GPU devices available: {torch.cuda.device_count()}')
for N_gpu in range(torch.cuda.device_count()):
    print(f'GPU {N_gpu+1} named {torch.cuda.get_device_name(N_gpu)}')
    
kfold = None
num_workers = 0
type_transform = tonic.transforms.NumpyAsType(int)

trainset = tonic.datasets.NMNIST(save_to='../../Data/', train=True, transform=type_transform)
testset = tonic.datasets.NMNIST(save_to='../../Data/', train=False, transform=type_transform)
loader = get_loader(trainset, kfold=300, num_workers=num_workers)
trainloader = get_loader(trainset, kfold=kfold, num_workers=num_workers)
testloader = get_loader(testset, kfold=kfold, num_workers=num_workers)
num_sample_train = len(trainloader)
num_sample_test = len(testloader)
n_classes = len(testset.classes)
print(f'number of samples in the training set: {len(trainloader)}')
print(f'number of samples in the testing set: {len(testloader)}')


name = 'homeohots'
homeo = True
timestr = '2022-04-22'
dataset_name = 'nmnist'

Rz = [2, 4]
N_neuronz = [16, 32]
tauz = [1e4*2, 1e4*16]

hots = network(name, dataset_name, timestr, trainset.sensor_size, nb_neurons = N_neuronz, tau = tauz, R = Rz, homeo = homeo)

filtering_threshold = [5, None]

if not os.path.exists('../Records/'):
    os.mkdir('../Records/')
    os.mkdir('../Records/networks/')
path = '../Records/networks/'+hots.name+'.pkl'
if not os.path.exists(path):
    hots.clustering(loader, trainset.ordering, filtering_threshold = filtering_threshold)
    
hots.coding(trainloader, trainset.ordering, trainset.classes, filtering_threshold = filtering_threshold, training=True)  

initial_name = hots.name
modified_name = initial_name + '_nothres_testset'
hots.name = modified_name
hots.coding(testloader, trainset.ordering, trainset.classes, filtering_threshold = filtering_threshold, training=False)

hots.name = initial_name
     
jitter = (None, None)
num_workers = 0
learning_rate = 0.005
beta1, beta2 = 0.9, 0.999
betas = (beta1, beta2)
num_epochs = 2 ** 5 + 1
N_output_neurons = N_neuronz[-1]
ts_size = (trainset.sensor_size[0],trainset.sensor_size[1],N_output_neurons)
tau_cla = 1e5
drop_proba = None

if drop_proba:
    drop_transform = tonic.transforms.DropEvent(p = drop_proba)
    full_transform = tonic.transforms.Compose([drop_transform, type_transform])
    model_path = f'../Records/networks/{initial_name}_{tau_cla}_{learning_rate}_{betas}_{num_epochs}_{jitter}_{drop_proba}.pkl'
    results_path = f'../Records/LR_results/{modified_name}_{tau_cla}_{learning_rate}_{betas}_{num_epochs}_{jitter}_{drop_proba}.pkl'
else:
    full_transform = type_transform
    model_path = f'../Records/networks/{initial_name}_{tau_cla}_{learning_rate}_{betas}_{num_epochs}_{jitter}.pkl'
    results_path = f'../Records/LR_results/{modified_name}_{tau_cla}_{learning_rate}_{betas}_{num_epochs}_{jitter}.pkl'
    
train_path = f'../Records/output/train/{initial_name}_{num_sample_train}_{jitter}/'
test_path = f'../Records/output/test/{initial_name}_{num_sample_test}_{jitter}/'

trainset_output = HOTS_Dataset(train_path, trainset.sensor_size, dtype=trainset.dtype, transform=full_transform)
trainloader_output = get_loader(trainset_output, num_workers=num_workers)

classif_layer, losses = fit_mlr(trainloader_output, model_path, tau_cla, learning_rate, betas, num_epochs, ts_size, trainset.ordering, len(trainset.classes))

trainset_output_histp = HOTS_Dataset(train_path, trainset.sensor_size, dtype=trainset.dtype, transform=full_transform)
testset_output_histo = HOTS_Dataset(test_path, trainset.sensor_size, dtype=trainset.dtype, transform=full_transform)
testset_output = HOTS_Dataset(test_path, trainset.sensor_size, dtype=trainset.dtype, transform=full_transform)
testloader_output = get_loader(testset_output, num_workers=num_workers)

likelihood, true_target, timestamps = predict_mlr(classif_layer,tau_cla,testloader_output,results_path,ts_size,testset_output.ordering)
score = make_histogram_classification(trainset_output, testset_output, N_neuronz[-1])
meanac, onlinac, lastac = score_classif_events(likelihood, true_target, n_classes, original_accuracy = score, figure_name = 'nmnist_online.pdf')
