import torch, tonic, os, pickle, copy, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from IPython.display import Image
from tqdm import tqdm
from hots.layer import mlrlayer
from hots.timesurface import timesurface, timesurface_stack
from scipy.stats import beta
from scipy.optimize import curve_fit

def printfig(fig, name):
    dpi_exp = None
    bbox = 'tight'
    path = '../../manuscript/fig/'
    #path = '../../GrimaldiEtAl2020HOTS_clone_laurent/fig'
    fig.savefig(path+name, dpi = dpi_exp, bbox_inches=bbox, transparent=True)
    
def entropy(timesurface):
    hist = torch.histc(timesurface, bins = 256, min = 0, max = 1)
    return -torch.nansum(hist*torch.log2(hist))
    
def timesurfaces_entropy(all_ts):
    ent_by_ev = torch.zeros([all_ts.shape[0]])
    for event_indice in range(all_ts.shape[0]):
        ent = entropy(all_ts[event_indice,:,:,:])
        ent_by_ev[event_indice] = ent
    return ent_by_ev
    
def get_loader(dataset, kfold = None, kfold_ind = 0, num_workers = 0, shuffle=True, batch_size = 1, seed=42):
    # creates a loader for the samples of the dataset. If kfold is not None, 
    # then the dataset is splitted into different folds with equal repartition of the classes.
    classes = dataset.classes
    targets = dataset.targets
    
    if kfold:
        subset_indices = []
        subset_size = len(dataset)//kfold
        for i in range(len(classes)):
            all_ind = np.where(np.array(targets)==i)[0]
            subset_indices += all_ind[kfold_ind*subset_size//len(classes):
                            min((kfold_ind+1)*subset_size//len(classes), len(dataset)-1)].tolist()
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        subsampler = torch.utils.data.SubsetRandomSampler(subset_indices, g_cpu)
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, sampler=subsampler, num_workers = num_workers)
        #loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=subsampler, num_workers = num_workers, collate_fn=tonic.collation.PadTensors(batch_first=True))
    else:
        loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, num_workers = num_workers)
        #loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, num_workers = num_workers, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True))
    return loader

def get_sliced_loader(dataset, slicing_time_window, dataset_name, train, only_first = True, kfold = None, kfold_ind = 0, transform = tonic.transforms.NumpyAsType(int), num_workers = 0, shuffle=True, seed=43):
    
    classes = dataset.classes
    targets = dataset.targets

    metadata_path = f'{dataset.location_on_system}/metadata/{dataset_name}_{int(slicing_time_window*1e-3)}_{only_first}_{train}'

    if only_first:
        slicer = tonic.slicers.SliceAtTimePoints(start_tw = [0], end_tw = [slicing_time_window])
    else:
        slicer = tonic.slicers.SliceByTime(time_window = slicing_time_window, include_incomplete = True)

    sliced_dataset = tonic.SlicedDataset(dataset, slicer = slicer, metadata_path = metadata_path)
    classes = sliced_dataset.dataset.classes
    targets = sliced_dataset.dataset.targets
    
    if kfold:
        subset_indices = []
        subset_size = len(sliced_dataset)//kfold
        for i in range(len(classes)):
            all_ind = np.where(np.array(targets)==i)[0]
            subset_indices += all_ind[kfold_ind*subset_size//len(classes):
                            min((kfold_ind+1)*subset_size//len(classes), len(sliced_dataset)-1)].tolist()
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        subsampler = torch.utils.data.SubsetRandomSampler(subset_indices, g_cpu)
        loader = torch.utils.data.DataLoader(sliced_dataset, batch_size=1, shuffle=False, sampler=subsampler, num_workers = num_workers)
    else:
        loader = torch.utils.data.DataLoader(sliced_dataset, shuffle=shuffle, num_workers = num_workers)
    return loader

def get_properties(events, target, ind_sample, values, ordering = 'xytp', distinguish_polarities = False):
    t_index, p_index = ordering.index('t'), ordering.index('p')
    if distinguish_polarities: 
        for polarity in [0,1]:
            events_pol = events[(events[:, p_index]==polarity)]
            isi = np.diff(events_pol[:, t_index])
            if 'mean_isi' in values.keys():
                values['mean_isi'][polarity, ind_sample, target] = (isi[isi>0]).mean()
            if 'median_isi' in values.keys():
                values['median_isi'][polarity, ind_sample, target] = np.median((isi[isi>0]))
            if 'synchronous_events' in values.keys():
                values['synchronous_events'][polarity, ind_sample, target] = (isi==0).mean()
            if 'nb_events' in values.keys():
                values['nb_events'][polarity, ind_sample, target] = events_pol.shape[0]
            if 'time' in values.keys():
                values['time'][polarity, ind_sample, target] = events[-1,t_index]-events[0,t_index]
    else:
        events_pol = events
        isi = np.diff(events_pol[:, t_index])
        if 'mean_isi' in values.keys():
            values['mean_isi'][0, ind_sample, target] = (isi[isi>0]).mean()
        if 'median_isi' in values.keys():
            values['median_isi'][0, ind_sample, target] = np.median((isi[isi>0]))
        if 'synchronous_events' in values.keys():
            values['synchronous_events'][0, ind_sample, target] = (isi==0).mean()
        if 'nb_events' in values.keys():
            values['nb_events'][0, ind_sample, target] = events_pol.shape[0]
        if 'time' in values.keys():
            values['time'][0, ind_sample, target] = events[-1,t_index]-events[0,t_index]
    return values

def get_dataset_info(trainset, testset=None, properties = ['mean_isi', 'synchronous_events', 'nb_events'], distinguish_labels = False, distinguish_polarities = False):
    
    print(f'number of samples in the trainset: {len(trainset)}')
    if testset: print(f'number of samples in the testset: {len(testset)}')
    print(40*'-')
    
    #x_index, y_index, t_index, p_index = trainset.ordering.index("x"), trainset.ordering.index("y"), trainset.ordering.index("t"), trainset.ordering.index("p")
    nb_class = len(trainset.classes)
    nb_sample = len(trainset)
    if testset: nb_sample += len(testset)
    nb_pola = 2
    
    values = {}
    for name in properties:
        values.update({name:np.zeros([nb_pola, nb_sample, nb_class])})

    ind_sample = 0
    num_labels_trainset = np.zeros([nb_class])
    if testset: num_labels_testset = np.zeros([nb_class])
    
    loader = get_loader(trainset, shuffle=False)
    for events, target in loader:
        events = events.squeeze().numpy()
        values = get_properties(events, target, ind_sample, values, ordering = trainset.ordering, distinguish_polarities = distinguish_polarities)
        num_labels_trainset[target] += 1
        ind_sample += 1
       
    if testset:
        loader = get_loader(testset, shuffle=False)
        for events, target in loader:
            events = events.squeeze().numpy()
            values = get_properties(events, target, ind_sample, values, ordering = trainset.ordering, distinguish_polarities = distinguish_polarities)
            num_labels_testset[target] += 1
            ind_sample += 1
        
    print(f'number of samples in each class for the trainset: {num_labels_trainset}')
    if testset: print(f'number of samples in each class for the testset: {num_labels_testset}')
    print(40*'-')
        
    width_fig = 30
    fig, axs = plt.subplots(1,len(values.keys()), figsize=(width_fig,width_fig//len(values.keys())))
    for i, value in enumerate(values.keys()):
        if distinguish_polarities:
            x = []
            for p in range(nb_pola):
                x.append(values[value][p,:,:].sum(axis=1).ravel())
                #print(values[value][p,:,:].sum(axis=1).ravel())
            ttl = value
        elif distinguish_labels:
            x = []
            for c in range(nb_class):
                x.append(values[value][0,np.nonzero(values[value][0,:,c]),c].ravel())
                #print(values[value][0,np.nonzero(values[value][0,:,c]),c].ravel())
            ttl = value
        else:
            x = []
            x.append(values[value][0,:,:].sum(axis=1).ravel())
            ttl = value

        for k in range(len(x)):
            n, bins, patches = axs[i].hist(x=x[k], bins='auto',
                                    alpha=.5, rwidth=0.85)
            
        axs[i].grid(axis='y', alpha=0.75)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'Histogram for the {ttl}')
        maxfreq = n.max()
        axs[i].set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        if not distinguish_polarities and not distinguish_labels:
            print(f'Mean value for {ttl}: {np.array(x).mean()}')
        #axs[i].set_xscale("log")
        #axs[i].set_yscale("log")
    return values

class HOTS_Dataset(tonic.dataset.Dataset):
    """Make a dataset from the output of the HOTS network
    """

    def __init__(self, path_to, sensor_size, classes, dtype, train=True, transform=None, target_transform=None):
        super(HOTS_Dataset, self).__init__(
            path_to, transform=transform, target_transform=target_transform
        )
        
        self.dtype = dtype
        self.ordering = dtype.names
        self.classes = classes
        self.int_classes = dict(zip(self.classes, range(len(classes))))
        self.location_on_system = path_to
        self.sensor_size = sensor_size
        
        if not os.path.exists(self.location_on_system):
            print('no output, process the samples first')
            return
        
        for path, dirs, files in os.walk(self.location_on_system):
            for file in files:
                if file.endswith("npy"):
                    self.data.append(os.path.join(path, file))
                    n_target = 0
                    indice = path.find(self.classes[n_target])
                    while indice==-1:
                        n_target += 1
                        indice = path.find(self.classes[n_target])
                    self.targets.append(self.int_classes[self.classes[n_target]])
                        
        self.num_samples = len(self.targets)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events, target = np.load(self.data[index]), self.targets[index]
        while len(events.shape)>2:
            events = events.squeeze(0)
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return self.num_samples

def make_histogram_classification(trainset, testset, nb_output_pola, k = 6, device = 'cuda'):
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f'device -> {device}')
    
    p_index = trainset.ordering.index('p')
    train_histo_map = torch.zeros([len(trainset),nb_output_pola], device = device)
    train_labels = torch.zeros([len(trainset)], device = device)
    
    dist = torch.nn.PairwiseDistance(p=2)
    score = 0
    
    for sample in range(len(trainset)):
        events, label = trainset[sample]
        histo = torch.bincount(torch.tensor(events[:,p_index], device = device))
        train_histo_map[sample,:len(histo)] = histo/histo.sum()
        train_labels[sample] = label
        
    for sample in range(len(testset)):
        histo = torch.zeros([nb_output_pola], device = device)
        events, label = testset[sample]
        if events.shape[0]>0:
            histo_bin = torch.bincount(torch.tensor(events[:,p_index], device = device))
            histo[:len(histo_bin)] = histo_bin/histo_bin.sum()
            distances = dist(histo, train_histo_map)
            distances_sorted, indices = torch.sort(distances)
            label_sorted = train_labels[indices].clone().detach().int()
            inference = torch.bincount(label_sorted[:k]).argmax()
            if inference==label:
                score+=1
        else:
            inference = np.random.randint(0,len(trainset.classes))
            if inference==label:
                score+=1
            
    score/=len(testset)
    return score  

def fit_mlr(loader, 
            model_path,
            tau_cla,
            learning_rate,
            betas,
            num_epochs,
            sensor_size,
            ordering,
            n_classes,
            ts_size = None,
            ts_batch_size = None,
            drop_proba = None,
            device = 'cuda'):
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            classif_layer, mean_loss_epoch = pickle.load(file)
    else:
        criterion = torch.nn.BCELoss(reduction="mean")
        amsgrad = True #or False gives similar results
            
        print(f'device -> {device}')
        
        N = sensor_size[0]*sensor_size[1]*sensor_size[2]
        if ts_size: N = ts_size[0]*ts_size[1]*sensor_size[2]

        classif_layer = mlrlayer(N, n_classes, device=device)
        classif_layer.train()
        optimizer = torch.optim.Adam(
            classif_layer.parameters(), lr=learning_rate, betas=betas, amsgrad=amsgrad
        )
        mean_loss_epoch = []
        
        past_trainings = glob.glob(glob.escape(model_path[:-10])+'*')
        if len(past_trainings)==1:
            with open(past_trainings[0], 'rb') as file:
                classif_layer, mean_loss_epoch = pickle.load(file)
            epoch_vector = np.arange(len(mean_loss_epoch), int(num_epochs))
        else:
            epoch_vector = range(int(num_epochs))
        
        for epoch in tqdm(epoch_vector):
            if epoch > 1:
                drop_proba = None
            losses = []
            for events, label in loader:
                events = events.squeeze(0)
                if ts_batch_size and len(events)>ts_batch_size:
                    nb_batch = len(events)//ts_batch_size+1
                    previous_timestamp = []
                    for load_nb in range(nb_batch):
                        X, ind_filtered, previous_timestamp = timesurface(events, (sensor_size[0], sensor_size[1], sensor_size[2]), ordering, tau = tau_cla, surface_dimensions = ts_size, ts_batch_size = ts_batch_size, drop_proba = drop_proba, load_number = load_nb, previous_timestamp = previous_timestamp, device = device)
                        
                        n_events = X.shape[0]

                        X = X.reshape(n_events, N)
                        optimizer.zero_grad()

                        outputs = classif_layer(X)

                        labels = label.to(device)*torch.ones(n_events).to(device).to(torch.int64)
                        labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).to(device).to(torch.float32)

                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())

                        del X, outputs, labels, loss
                        torch.cuda.empty_cache()
                    
                else:
                    X, ind_filtered = timesurface(events, (sensor_size[0], sensor_size[1], sensor_size[2]), ordering, tau = tau_cla, surface_dimensions = ts_size, drop_proba = drop_proba, device = device)
                    X, label = X, label.to(device)
                    n_events = X.shape[0]
                    X = X.reshape(n_events, N)
                    optimizer.zero_grad()
                    
                    outputs = classif_layer(X)

                    labels = label*torch.ones(n_events).to(device).to(torch.int64)
                    labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).to(device).to(torch.float32)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    
                    del X, outputs, labels, loss
                    torch.cuda.empty_cache()
                #print(f'New label {label.item()} - Mean loss: {np.round(np.nanmean(np.array(losses)),3)}')
                #plt.plot(losses,'*')
                #plt.show()
            mean_loss_epoch.append(np.nanmean(np.array(losses)))
            print(f'Loss for epoch number {epoch}: {np.round(np.nanmean(np.array(losses)),3)}')
            
            with open(model_path[:-4]+f'_it{epoch}.pkl', 'wb') as file:
                pickle.dump([classif_layer, mean_loss_epoch], file, pickle.HIGHEST_PROTOCOL)
            
            if os.path.exists(model_path[:-4]+f'_it{epoch-1}.pkl'): os.remove(model_path[:-4]+f'_it{epoch-1}.pkl')
        
        with open(model_path, 'wb') as file:
            pickle.dump([classif_layer, mean_loss_epoch], file, pickle.HIGHEST_PROTOCOL)
        os.remove(model_path[:-4]+f'_it{epoch}.pkl')

    return classif_layer, mean_loss_epoch

def predict_mlr(mlrlayer,
                tau_cla,
                loader,
                results_path,
                sensor_size,
                ordering,
                ts_size = None,
                save = True,
                device = 'cuda',
                ts_batch_size = None,
        ):
    
    if os.path.isfile(results_path):
        with open(results_path, 'rb') as file:
            likelihood, true_target, timestamps = pickle.load(file)
    else:    
        N = sensor_size[0]*sensor_size[1]*sensor_size[2]
        if ts_size: N = ts_size[0]*ts_size[1]*sensor_size[2]
        t_index = ordering.index('t')
        
        initial_memory = copy.copy(torch.cuda.memory_allocated())

        classif_layer = mlrlayer.to(device)
        
        with torch.no_grad():
            # needed for previous versions, now it should be ok to remove it
            classif_layer.linear = classif_layer.linear.double()
            #print(classif_layer.linear.weight.dtype)
            likelihood, true_target, timestamps = [], [], []

            for events, label in tqdm(loader):
                events = events.squeeze(0)
                timestamps.append(events[:,t_index])
                if events.shape[0]==0:
                    outputs = torch.Tensor([])
                    print('sample without events')
                elif ts_batch_size and len(events)>ts_batch_size:
                    nb_batch = len(events)//ts_batch_size+1
                    previous_timestamp = []
                    outputs = torch.Tensor([]).to(device)
                    for load_nb in range(nb_batch):
                        X, ind_filtered, previous_timestamp = timesurface(events, (sensor_size[0], sensor_size[1], sensor_size[2]), ordering, tau = tau_cla, surface_dimensions = ts_size, ts_batch_size = ts_batch_size, load_number = load_nb, previous_timestamp = previous_timestamp, device = device)
                        n_events = X.shape[0]
                        X, label = X, label.to(device)
                        X = X.reshape(n_events, N)
                        outputs_splitted = classif_layer(X.double())
                        outputs = torch.vstack([outputs,outputs_splitted]) if outputs.shape[0]>0 else outputs_splitted
                        del X, outputs_splitted
                        torch.cuda.empty_cache()
                else:
                    X, ind_filtered = timesurface(events, (sensor_size[0], sensor_size[1], sensor_size[2]), ordering, tau = tau_cla, surface_dimensions = ts_size, device=device)
                    n_events = X.shape[0]
                    X, label = X, label.to(device)
                    X = X.reshape(n_events, N)
                    outputs = classif_layer(X.double())
                    del X
                likelihood.append(outputs.cpu().numpy())
                true_target.append(label.cpu().numpy())
                del outputs
                torch.cuda.empty_cache()
            if save:
                with open(results_path, 'wb') as file:
                    pickle.dump([likelihood, true_target, timestamps], file, pickle.HIGHEST_PROTOCOL)

    return likelihood, true_target, timestamps

def score_classif_events(likelihood, true_target, n_classes, thres=None):
    
    max_len = 0
    for likeli in likelihood:
        if max_len<likeli.shape[0]:
            max_len=likeli.shape[0]

    matscor = np.zeros([len(true_target),max_len])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    nb_test = len(true_target)
    nb_events = np.zeros([nb_test])
    nb_no_decision = 0
    best_probability = 0

    for likelihood_, true_target_ in zip(likelihood, true_target):
        nb_event_sample = len(likelihood_)
        nb_events[sample] = nb_event_sample
        if nb_event_sample!=0:
            if np.where(likelihood_==np.max(likelihood_))[1][0]==true_target_:
                best_probability += 1
            pred_target = np.zeros(nb_event_sample)
            pred_target[:] = np.nan
            if not thres:
                pred_target = np.argmax(likelihood_, axis = 1)
            else:
                for i in range(nb_event_sample):
                    if np.max(likelihood_[i])>thres:
                        pred_target[i] = np.argmax(likelihood_[i])
            for event in range(nb_event_sample):
                if np.isnan(pred_target[event])==False:
                    matscor[sample,event] = pred_target[event]==true_target_
            if pred_target[-1]==true_target_:
                lastac+=1
            if np.sum(np.isnan(pred_target))==nb_event_sample:
                nb_no_decision += 1
                pred_target = np.random.randint(0,n_classes)
                matscor[sample,:] = pred_target==true_target_
                if pred_target==true_target_:
                    lastac+=1    
        else: 
            nb_no_decision += 1
            pred_target = np.random.randint(0,n_classes)
            matscor[sample,:] = pred_target==true_target_
            if pred_target==true_target_:
                lastac+=1
        sample+=1
    
    if matscor.shape[1]==0:
        meanac = 1/n_classes
        onlinac = 1/n_classes
        lastac = 1/n_classes
    else:
        meanac = np.nanmean(matscor)
        onlinac = np.nanmean(matscor, axis=0)
        lastac/=nb_test
        best_probability/=nb_test
    
    return meanac, onlinac, lastac, best_probability, np.quantile(nb_events, .9), nb_no_decision

def score_classif_time(likelihood, true_target, timestamps, timestep, thres=None, verbose=True):
    
    max_dur = 0
    for time in timestamps:
        if max_dur<time[-1]:
            max_dur=time[-1]
            
    time_axis = np.arange(0,max_dur,timestep)

    matscor = np.zeros([len(true_target),len(time_axis)])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    nb_test = len(true_target)
    
    if verbose: pbar = tqdm(total=len(likelihood))
    
    for likelihood_, true_target_, timestamps_ in zip(likelihood, true_target, timestamps):
        pred_timestep = np.zeros(len(time_axis))
        pred_timestep[:] = np.nan
        for step in range(1,len(pred_timestep)):
            if time_axis[step]<timestamps_.numpy()[-2]:
                indices = np.where((timestamps_.numpy()<time_axis[step])&(timestamps_.numpy()>=time_axis[step-1]))[0]
                mean_likelihood = np.mean(likelihood_[indices,:],axis=0)
            if np.isnan(mean_likelihood).sum()>0:
                pred_timestep[step] = np.nan
            else:
                if not thres:
                    pred_timestep[step] = np.nanargmax(mean_likelihood)
                elif np.max(likelihood_[indices,np.nanargmax(mean_likelihood)])>thres:
                    pred_timestep[step] = np.nanargmax(mean_likelihood)
                else:
                    pred_timestep[step] = np.nan
            if not np.isnan(pred_timestep[step]):
                matscor[sample,step] = pred_timestep[step]==true_target_
        lastev = -1
        while np.isnan(pred_timestep[lastev]):
            lastev -= 1
        if pred_timestep[lastev]==true_target_:
            lastac+=1
        if verbose: pbar.update(1)
        sample+=1
       
    if verbose: pbar.close()
    
    meanac = np.nanmean(matscor)
    onlinac = np.nanmean(matscor, axis=0)
    lastac/=nb_test
    truepos = len(np.where(matscor==1)[0])
    falsepos = len(np.where(matscor==0)[0])
        
    if verbose:
        print(f'Mean accuracy: {np.round(meanac,3)*100}%')
        plt.semilogx(time_axis*1e-3,onlinac, '.');
        plt.xlabel('time (in ms)');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of time');
    
    return meanac, onlinac, lastac, truepos, falsepos

def online_accuracy(mlrlayer,
                tau_cla,
                loader,
                results_path,
                sensor_size,
                ordering,
                n_classes,
                mlr_threshold = None,
                original_accuracy = None,
                original_accuracy_nohomeo = None,
                online_plot = False,
                psycho_plot = False,
                figure_name = None,
                save_likelihood = False,
                device = 'cuda',
                ts_size = None,
                ts_batch_size = None,):
    onlinac_path = results_path[:-4]+f'_onlinac_{mlr_threshold}'
    if not os.path.exists(onlinac_path+'.npz'):
        likelihood, true_target, timestamps = predict_mlr(mlrlayer, tau_cla, loader, results_path, sensor_size, ordering, save=save_likelihood, device=device, ts_size=ts_size, ts_batch_size=ts_batch_size)
        meanac, onlinac, lastac, best_probability, percentile_90, nb_no_decision = score_classif_events(likelihood, true_target, n_classes, thres=mlr_threshold)
        np.savez(onlinac_path, meanac, onlinac, lastac, best_probability, percentile_90, nb_no_decision)
    else: 
        data_stored = np.load(onlinac_path+'.npz')
        meanac = data_stored['arr_0']
        onlinac = data_stored['arr_1']
        lastac = data_stored['arr_2']
        best_probability = data_stored['arr_3']
        percentile_90 = data_stored['arr_4']
        nb_no_decision = data_stored['arr_5']
        
    print(f'Number of chance decisions: {nb_no_decision}')
    print(f'90th quantile for number of events: {percentile_90}')
    print(f'Mean accuracy: {np.round(meanac,3)*100}%')
    print(f'Last accuracy: {np.round(lastac,3)*100}%')
    print(f'Highest probability accuracy: {np.round(best_probability,3)*100}%')

    if online_plot:   
        fig, ax = plt.subplots()
        sampling = (np.logspace(0,np.log10(percentile_90),100)).astype(int)
        ax.semilogx(sampling[:-1],onlinac[sampling[:-1]]*100, '.', label='online HOTS (ours)');
        ax.hlines(1/n_classes*100,0,int(percentile_90), linestyles='dashed', color='k', label='chance level')
        if original_accuracy:
            ax.hlines(original_accuracy*100,0,int(percentile_90), linestyles='dashed', color='g', label='HOTS with homeostasis')
        if original_accuracy_nohomeo:
            ax.hlines(original_accuracy_nohomeo*100,0,int(percentile_90), linestyles='dashed', color='r', label='original HOTS')
        ax.set_xlabel('Number of events', fontsize=16);
        ax.axis([1,int(percentile_90),0,101]);
        #plt.title('LR classification results evolution as a function of the number of events');
        plt.setp(ax.get_xticklabels(),fontsize=12)
        #ax.set_yticks([])
        plt.setp(ax.get_yticklabels(),fontsize=12)
        ax.legend(fontsize=12, loc='lower right');
        ax.set_ylabel('Accuracy (in %)', fontsize=16);
        if figure_name:
            printfig(fig, figure_name)
    
    if psycho_plot: 
        notnan_ind = np.where(np.isnan(matscor)==0)
        event_nb = np.sort(notnan_ind[1])
        alpha, beta = fit_PF(notnan_ind[1], matscor[notnan_ind[0], notnan_ind[1]], init_params=[200, .5])
        
        fig, ax = plt.subplots()
        ax.semilogx(notnan_ind[1], matscor[notnan_ind[0], notnan_ind[1]], 'b.', alpha=.01, label='online HOTS (ours)')
        ax.plot(event_nb, pf(event_nb, alpha, beta))
        sampling = np.arange(0,len(nb_events))
        ax.hlines(1/n_classes,0,int(max_len), linestyles='dashed', color='k', label='chance level')
        ax.set_xlabel('Number of events', fontsize=16);
        #ax.axis([1,int(max_len),-.01,1.01]);
        #plt.title('LR classification results evolution as a function of the number of events');
        plt.setp(ax.get_xticklabels(),fontsize=12)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["False", "True"], fontsize=16)
        ax.legend(fontsize=12, loc='lower right');
        #ax.set_ylabel('Accuracy (in %)', fontsize=16);
        if figure_name:
            printfig(fig, figure_name)
    
    return onlinac, best_probability, meanac, lastac

def NR_jitter(jitter,Rmax,Rmin,jitter0,powa): 
    x = jitter**powa
    semisat = jitter0**powa
    output = Rmax-Rmax*x/(x+jitter0)+Rmin
    return output

def fit_NR(jitter,accuracy,init_params=[1,1/10,1e4,2]):
    popt, pcov = curve_fit(NR_jitter, jitter, accuracy, p0 = init_params)
    Rmax = popt[0]
    Rmin = popt[1]
    semisat = popt[2]
    powa = popt[3]
    return Rmin,Rmax,semisat,powa

def pf(x, alpha, beta):
    return 1. / (1 + np.exp( -(x-alpha)/beta))

def fit_PF(event_nb, success, init_params=[100., 1.]):
    par, mcov = curve_fit(pf, event_nb, success, p0 = init_params)
    return par[0], par[1]

def plotjitter(fig, ax, jit, score, param = [0.8, 22, 4, 0.1], color='red', label='name', nb_class=10, n_epo = 33, fitting = True, logscale = False):
    score_stat = np.zeros([3,len(jit)])
    q = [0.05,0.95]
    for i in range(score.shape[1]):
        mean = np.mean(score[:,i])
        if np.unique(score[:,i]).shape[0]==1:
            score_stat[0,i], score_stat[1,i], score_stat[2,i] = mean, mean, mean
        else:
            paramz = beta.fit(score[:,i]*.9999+.00001, floc=0, fscale = 1)
            score_stat[0,i], score_stat[2,i] = beta.ppf(q, a=paramz[0], b=paramz[1])
            score_stat[1,i] = np.mean(score[:,i])

    if fitting:
        Rmin,Rmax,semisat,powa = fit_NR(jit,score_stat[1,:],init_params=param)
        if logscale:
            jitter_cont = np.logspace(np.min(np.log10(jit)),np.max(np.log10(jit)),100) 
            nr_fit = NR_jitter(jitter_cont,Rmax,Rmin,semisat, powa)
            ax.semilogx(jitter_cont, nr_fit*100, color=color, lw=1)
        else:
            jitter_cont = np.linspace(np.min(jit),np.max(jit),100)
            nr_fit = NR_jitter(jitter_cont,Rmax,Rmin,semisat,powa)
            ax.plot(jitter_cont, nr_fit*100, color=color, lw=1)
    if logscale:
        ax.semilogx(jit, score_stat[1,:]*100, '.',color=color, label=label)
    else:
        ax.plot(jit, score_stat[1,:]*100, '.',color=color, label=label)
    ax.fill_between(jit, score_stat[2,:]*100, score_stat[0,:]*100, facecolor=color, edgecolor=None, alpha=.3)
        
    x_halfsat = []
    if fitting:
        halfsat = (Rmax-1/nb_class)/2+1/nb_class
        if halfsat>nr_fit[-1]:
            ind_halfsat = np.where(nr_fit<halfsat)[0]
            x_halfsat = jitter_cont[ind_halfsat[0]]
    return fig, ax, x_halfsat

def apply_jitter(min_jitter, max_jitter, jitter_type, hots, hots_nohomeo, classif_layer, tau_cla, dataset_name, trainset_output, trainset_output_nohomeo, learning_rate, betas, num_epochs, ts_batch_size = None, drop_proba_mlr = None, filtering_threshold = None, kfold = None, nb_trials = 10, nb_points = 20, mlr_threshold = None, slicing_time_window = 1e6, device = 'cuda', fitting = True, fit_param = None, figure_name = None, verbose = False):
    
    save_likelihood = False
    print(f'device -> {device}')
    
    initial_name = copy.copy(hots.name)
    initial_name_nohomeo = copy.copy(hots_nohomeo.name)
    
    n_classes = len(trainset_output.classes)
    n_output_neurons = len(hots.layers[-1].cumhisto)
    ts_size = [trainset_output.sensor_size[0],trainset_output.sensor_size[1],n_output_neurons]
    
    type_transform = tonic.transforms.NumpyAsType(int)
    if not os.path.exists(hots.record_path+'jitter_results/'):
        os.mkdir(hots.record_path+'jitter_results/')
    if jitter_type=='temporal':
        std_jit_t = np.logspace(min_jitter,max_jitter,nb_points)
        jitter_values = std_jit_t
    else:
        std_jit_s = np.linspace(min_jitter,max_jitter,nb_points)
        var_jit_s = std_jit_s**2
        jitter_values = var_jit_s
        
    scores_jit = np.zeros([nb_trials, len(jitter_values)])
    scores_jit_histo = np.zeros([nb_trials, len(jitter_values)])
    scores_jit_histo_nohomeo = np.zeros([nb_trials, len(jitter_values)])

    torch.set_default_tensor_type("torch.DoubleTensor")

    for trial in tqdm(range(nb_trials)):
        jitter_path = hots.record_path+f'jitter_results/{initial_name}_{nb_trials}_{min_jitter}_{max_jitter}_{kfold}_{nb_points}_{trial}'
        if not os.path.exists(jitter_path+'.npz'):
            scores_jit_single = np.zeros([len(jitter_values)])
            scores_jit_histo_single = np.zeros([len(jitter_values)])
            scores_jit_histo_nohomeo_single = np.zeros([len(jitter_values)])
            for ind_jit, jitter_val in enumerate(jitter_values):
                print(f'For {jitter_type} jitter equal to {jitter_val} - Trial number {trial}')
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
                    spatial_jitter_transform = tonic.transforms.SpatialJitter(sensor_size = trainset_output.sensor_size, var_x = jitter_val, var_y = jitter_val, clip_outliers = True)
                    transform_full = tonic.transforms.Compose([spatial_jitter_transform, type_transform])

                if dataset_name=='poker':
                    testset = tonic.datasets.POKERDVS(save_to='../../Data/', train=False, transform=transform_full)
                    testloader = get_loader(testset, kfold = kfold)
                    hots.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter = jitter, filtering_threshold = filtering_threshold, device = device, verbose=False)
                    hots_nohomeo.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter=jitter, filtering_threshold=filtering_threshold, device = device, verbose=False)
                elif dataset_name=='nmnist':
                    testset = tonic.datasets.NMNIST(save_to='../../Data/', train=False, transform=transform_full)
                    testloader = get_loader(testset, kfold = kfold)
                    hots.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter = jitter, filtering_threshold = filtering_threshold, device = device, verbose=False)
                    hots_nohomeo.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter=jitter, filtering_threshold=filtering_threshold, device = device, verbose=False)
                if dataset_name=='gesture':
                    testset = tonic.datasets.DVSGesture(save_to='../../Data/', train=False, transform=transform_full)
                    testloader = get_sliced_loader(testset, slicing_time_window, dataset_name, False, only_first=True, kfold=kfold)
                    hots.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter = jitter, filtering_threshold = filtering_threshold, ts_batch_size=ts_batch_size, device = device, verbose=False)
                    hots_nohomeo.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter=jitter, filtering_threshold=filtering_threshold, ts_batch_size=ts_batch_size, device = device, verbose=False)
                    
                    
                num_sample_test = len(testloader)

                test_path = hots.record_path+f'output/test/{hots.name}_{num_sample_test}_{jitter}/'
                results_path = hots.record_path+f'LR_results/{hots.name}_{tau_cla}_{num_sample_test}_{learning_rate}_{betas}_{num_epochs}_{drop_proba_mlr}_{jitter}.pkl'
                
                testset_output = HOTS_Dataset(test_path, trainset_output.sensor_size, trainset_output.classes, dtype=trainset_output.dtype, transform=type_transform)
                test_outputloader = get_loader(testset_output, shuffle=False)
                
                if len(test_outputloader)<num_sample_test:
                    print(f'{len(test_outputloader)} is not enough')
                    shutil.rmtree(test_path)
                    #testset = tonic.datasets.DVSGesture(save_to='../../Data/', train=False, transform=transform_full)
                    #testloader = get_sliced_loader(testset, slicing_time_window, dataset_name, False, only_first=True, kfold=kfold)
                    events, label = next(iter(testloader))
                    hots.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter = jitter, filtering_threshold = filtering_threshold, ts_batch_size=ts_batch_size, device = device, verbose=False)
                
                testset_output = HOTS_Dataset(test_path, trainset_output.sensor_size, trainset_output.classes, dtype=trainset_output.dtype, transform=type_transform)
                test_outputloader = get_loader(testset_output, shuffle=False)
                
                onlinac, best_probability, meanac, lastac = online_accuracy(classif_layer, tau_cla, test_outputloader, results_path, ts_size, testset_output.ordering, n_classes, mlr_threshold = mlr_threshold, ts_batch_size = ts_batch_size)

                scores_jit_histo_single[ind_jit] = make_histogram_classification(trainset_output, testset_output, n_output_neurons)
                scores_jit_single[ind_jit] = best_probability

                test_path_nohomeo = hots.record_path+f'output/test/{hots_nohomeo.name}_{num_sample_test}_{jitter}/'
                testset_output_nohomeo = HOTS_Dataset(test_path_nohomeo, trainset_output.sensor_size, trainset_output.classes, dtype=trainset_output.dtype, transform=type_transform)
                
                if len(testset_output_nohomeo)<num_sample_test:
                    print(f'{len(testset_output_nohomeo)} is not enough')
                    shutil.rmtree(test_path_nohomeo)
                    #testset = tonic.datasets.DVSGesture(save_to='../../Data/', train=False, transform=transform_full)
                    #testloader = get_sliced_loader(testset, slicing_time_window, dataset_name, False, only_first=True, kfold=kfold)
                    hots_nohomeo.coding(testloader, trainset_output.ordering, testset.classes, training=False, jitter=jitter, filtering_threshold=filtering_threshold, ts_batch_size=ts_batch_size, device = device, verbose=False)
                    
                testset_output_nohomeo = HOTS_Dataset(test_path_nohomeo, trainset_output.sensor_size, trainset_output.classes, dtype=trainset_output.dtype, transform=type_transform)
                
                scores_jit_histo_nohomeo_single[ind_jit] = make_histogram_classification(trainset_output_nohomeo, testset_output_nohomeo, n_output_neurons)

                if verbose: 
                    print(f'For {jitter_type} jitter equal to {jitter_val}')
                    print(f'Online HOTS accuracy: {best_probability*100} %')
                    print(f'Original HOTS accuracy: {scores_jit_histo_nohomeo_single[ind_jit]*100} %')
                    print(f'HOTS with homeostasis accuracy: {scores_jit_histo_single[ind_jit]*100} %')

            if jitter_type=='spatial':
                jitter_values = std_jit_s
            np.savez(jitter_path, jitter_values, scores_jit_single, scores_jit_histo_single, scores_jit_histo_nohomeo_single)
        else:
            data_stored = np.load(jitter_path+'.npz')
            jitter_values = data_stored['arr_0']
            scores_jit_single = data_stored['arr_1']
            scores_jit_histo_single = data_stored['arr_2']
            scores_jit_histo_nohomeo_single = data_stored['arr_3']

        scores_jit[trial,:] = scores_jit_single
        scores_jit_histo[trial,:] = scores_jit_histo_single
        scores_jit_histo_nohomeo[trial,:] = scores_jit_histo_nohomeo_single

        hots.name = initial_name
        hots_nohomeo.name = initial_name_nohomeo
        
    if jitter_type=='temporal':
        logscale=True
        jitter_values*=1e-3
    else:
        logscale=False

    fig_t, ax_t = plt.subplots(1,1,figsize=(8,5))
    colorz = ['#2ca02c','#1f77b4','#d62728']
    label = 'online HOTS (ours)'
    if fit_param is not None: 
        param_1, param_2, param_3 = fit_param
    else:
        param_1 = [.9, 1/n_classes, 2, 2] # to change to adjust the fit
        param_2 = [.9, 1/n_classes, 2, 2]
        param_3 = [.9, 1/n_classes, 2, 2]
    n_epoch = 33

    fig_t, ax_t, semisat_t = plotjitter(fig_t, ax_t, jitter_values, scores_jit, param = param_1, color=colorz[1], label=label, nb_class=n_classes, n_epo=n_epoch, fitting = fitting, logscale=logscale)
    if fitting:
        print(f'semi saturation level for {label}: {np.round(semisat_t,2)} ms')

    label = 'HOTS with homeostasis'
    fig_t, ax_t, semisat_t = plotjitter(fig_t, ax_t, jitter_values, scores_jit_histo, param = param_2, color=colorz[0], label=label, nb_class=n_classes, n_epo=n_epoch, fitting = fitting, logscale=logscale)
    if fitting:
        print(f'semi saturation level for {label}: {np.round(semisat_t,2)} ms')

    label = 'original HOTS'
    fig_t, ax_t, semisat_t = plotjitter(fig_t, ax_t, jitter_values, scores_jit_histo_nohomeo, param = param_3, color=colorz[2], label=label, nb_class=n_classes, n_epo=n_epoch, fitting = fitting, logscale=logscale)
    if fitting:
        print(f'semi saturation level for {label}: {np.round(semisat_t,2)} ms')

    chance_t = np.ones([len(jitter_values)])*100/n_classes
    ax_t.plot(jitter_values,chance_t, 'k--', label='chance level')
    if jitter_type=='temporal':
        ax_t.axis([1,max(jitter_values),0,100]);
        ax_t.set_xlabel('Standard deviation of temporal jitter (in $ms$)', fontsize=16);
    else:
        ax_t.axis([0,max(jitter_values),0,100]);
        ax_t.set_xlabel('Standard deviation of spatial jitter (in $pixels$)', fontsize=16);
    ax_t.set_ylabel('Accuracy (in %)', fontsize=16);
        
    if figure_name:
        printfig(fig_t, figure_name)
    
    return jitter_values, scores_jit, scores_jit_histo, scores_jit_histo_nohomeo

def make_and_display_ts(events, file_name, trainset, tau, polarity= 'off', nb_frames = 100, ts_batch_size = None, numev_threshold = None, device = 'cuda'):
    
    if os.path.exists(f'figures/{file_name}_{polarity}.gif'):
        return Image(filename=f'figures/{file_name}_{polarity}.gif')

    else:
        if not os.path.exists('figures/'): os.mkdir('figures')
        print('Building .gif ...')
        frame_interval = int(events.shape[0]/nb_frames)
        indices_of_frames = np.arange(0,events.shape[0],frame_interval)
        if ts_batch_size and len(events)>ts_batch_size:
            nb_batch = len(events)//ts_batch_size+1
            previous_timestamp = []
            outputs = torch.Tensor([])
            ind_outputs = torch.Tensor([])
            for load_nb in tqdm(range(nb_batch)):
                all_ts, ind_filtered_timesurface, previous_timestamp = timesurface(events, trainset.sensor_size, trainset.ordering, tau = tau, ts_batch_size = ts_batch_size, load_number = load_nb, previous_timestamp = previous_timestamp, filtering_threshold = numev_threshold, device = device)
                indices_batch = indices_of_frames[np.where((indices_of_frames>=load_nb*ts_batch_size)&(indices_of_frames<(load_nb+1)*ts_batch_size))[0]]
                for event_indice in indices_batch-load_nb*ts_batch_size:
                    plt.imshow(all_ts[event_indice][0,:,:].cpu());
                    plt.axis('off');
                    plt.savefig(f'figures/ts_off_{file_name}_{event_indice+load_nb*ts_batch_size}');
                    plt.imshow(all_ts[event_indice][1,:,:].cpu());
                    plt.axis('off');
                    plt.savefig(f'figures/ts_on_{file_name}_{event_indice+load_nb*ts_batch_size}');
                del all_ts
                torch.cuda.empty_cache()
        else:
            all_ts, ind_filtered = timesurface(events, trainset.sensor_size, trainset.ordering, tau = tau, filtering_threshold = numev_threshold, device = device)
            
            for event_indice in tqdm(indices_of_frames):
                plt.imshow(all_ts[event_indice][0,:,:].cpu());
                plt.axis('off');
                plt.savefig(f'figures/ts_off_{file_name}_{event_indice}');
                plt.imshow(all_ts[event_indice][1,:,:].cpu());
                plt.axis('off');
                plt.savefig(f'figures/ts_on_{file_name}_{event_indice}');

        frames_off = np.stack([iio.imread(f"figures/ts_off_{file_name}_{x}.png") for x in indices_of_frames], axis=0)
        frames_on = np.stack([iio.imread(f"figures/ts_on_{file_name}_{x}.png") for x in indices_of_frames], axis=0)
        iio.imwrite(f'figures/{file_name}_off.gif', frames_off)
        iio.imwrite(f'figures/{file_name}_on.gif', frames_on)

        for x in indices_of_frames:
            os.remove(f"figures/ts_off_{file_name}_{x}.png")
            os.remove(f"figures/ts_on_{file_name}_{x}.png")
            
        return Image(filename=f'figures/{file_name}_{polarity}.gif')
    
    
def plot_kernels(classif_layer, N_output_neurons, sensor_size):
    kernels = classif_layer.linear.weight.data.cpu().numpy()
    fig, ax = plt.subplots(N_output_neurons, kernels.shape[0], figsize=(30, 90))
    for n in range(kernels.shape[0]):
        kernel = kernels[n].reshape(sensor_size[0],sensor_size[1], N_output_neurons)
        for p in range(N_output_neurons):
            ax[p, n].imshow(kernel[:,:,p])


