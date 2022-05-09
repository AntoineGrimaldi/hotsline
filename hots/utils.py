import torch, tonic, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from layer import mlrlayer
from timesurface import timesurface

def get_loader(dataset, kfold = None, kfold_ind = 0, num_workers = 0, shuffle=True, seed=42):
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
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=subsampler, num_workers = num_workers)
    else:
        loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, num_workers = num_workers)
    return loader

def get_sliced_loader(dataset, slicing_time_window, dataset_name, train, only_first = True, kfold = None, kfold_ind = 0, transform = tonic.transforms.NumpyAsType(int), num_workers = 0, shuffle=True, seed=42):
    
    classes = dataset.classes
    targets = dataset.targets

    metadata_path = f'./metadata/{dataset_name}_{int(slicing_time_window*1e-3)}_{only_first}_{train}'

    if only_first:
        slicer = tonic.slicers.SliceAtTimePoints(start_tw = [0], end_tw = [slicing_time_window])
    else:
        slicer = tonic.slicers.SliceByTime(time_window = slicing_time_window, include_incomplete = True)
    sliced_dataset = tonic.SlicedDataset(dataset, transform = transform, slicer = slicer, metadata_path = metadata_path)
    
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
            ttl = value
        elif distinguish_labels:
            x = []
            for c in range(nb_class):
                x.append(values[value][0,np.nonzero(values[value][0,:,c]),c].ravel())
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
        #axs[i].set_xscale("log")
        #axs[i].set_yscale("log")
    return values

class HOTS_Dataset(tonic.dataset.Dataset):
    """Make a dataset from the output of the HOTS network
    """

    def __init__(self, path_to, sensor_size, dtype, train=True, transform=None, target_transform=None):
        super(HOTS_Dataset, self).__init__(
            path_to, transform=transform, target_transform=target_transform
        )
        
        self.dtype = dtype
        self.ordering = dtype.names

        self.location_on_system = path_to
        
        if not os.path.exists(self.location_on_system):
            print('no output, process the samples first')
            return

        self.sensor_size = sensor_size
        
        for path, dirs, files in os.walk(self.location_on_system):
            files.sort()
            if dirs:
                label_length = len(dirs[0])
                self.classes = dirs
                self.int_classes = dict(zip(self.classes, range(len(dirs))))
            for file in files:
                if file.endswith("npy"):
                    self.data.append(np.load(os.path.join(path, file)))
                    n_target = 0
                    indice = path.find(self.classes[n_target])
                    while indice==-1:
                        n_target += 1
                        indice = path.find(self.classes[n_target])
                        
                    self.targets.append(self.int_classes[self.classes[n_target]])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events, target = self.data[index], self.targets[index]
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

def make_histogram_classification(trainset, testset, nb_output_pola, k = 6):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    p_index = trainset.ordering.index('p')
    train_histo_map = torch.zeros([len(trainset),nb_output_pola], device = device)
    train_labels = torch.zeros([len(trainset)], device = device)
    
    dist = torch.nn.PairwiseDistance(p=2)
    score = 0
    
    for sample in range(len(trainset)):
        events, label = trainset[sample]
        histo = torch.bincount(torch.tensor(events[0,:,p_index], device = device))
        train_histo_map[sample,:len(histo)] = histo/histo.sum()
        train_labels[sample] = label
        
    for sample in range(len(testset)):
        histo = torch.zeros([nb_output_pola], device = device)
        events, label = testset[sample]
        histo_bin = torch.bincount(torch.tensor(events[0,:,p_index], device = device))
        histo[:len(histo_bin)] = histo_bin/histo_bin.sum()
        distances = dist(histo, train_histo_map)
        distances_sorted, indices = torch.sort(distances)
        label_sorted = train_labels[indices].clone().detach().int()
        inference = torch.bincount(label_sorted[:k]).argmax()
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
            ts_size,
            ordering,
            n_classes,
            num_workers=0):
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            classif_layer, losses = pickle.load(file)
    
    else:
        criterion = torch.nn.BCELoss(reduction="mean")
        amsgrad = True #or False gives similar results
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device -> {device} - num_workers -> {num_workers}')
        
        N = ts_size[0]*ts_size[1]*ts_size[2]

        classif_layer = mlrlayer(N, n_classes, device=device)
        classif_layer.train()
        optimizer = torch.optim.Adam(
            classif_layer.parameters(), lr=learning_rate, betas=betas, amsgrad=amsgrad
        )

        for epoch in tqdm(range(int(num_epochs))):
            losses = []
            for events, label in loader:
                X, ind_filtered = timesurface(events.squeeze(0).squeeze(0), (ts_size[0], ts_size[1], ts_size[2]), ordering, tau = tau_cla, device=device)
                
                X, label = X.to(device) ,label.to(device)
                X = X.reshape(X.shape[0], N)

                outputs = classif_layer(X)

                n_events = X.shape[0]
                labels = label*torch.ones(n_events).to(device)
                labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).to(device)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        with open(model_path, 'wb') as file:
            pickle.dump([classif_layer, losses], file, pickle.HIGHEST_PROTOCOL)

    return classif_layer, losses

def predict_mlr(mlrlayer,
                tau_cla,
                loader,
                results_path,
                timesurface_size,
                ordering,
                num_workers = 0,
        ):    
    
    if os.path.isfile(results_path):
        with open(results_path, 'rb') as file:
            likelihood, true_target, timestamps = pickle.load(file) 
    else:    
        N = timesurface_size[0]*timesurface_size[1]*timesurface_size[2]
        t_index = ordering.index('t')

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f'device -> {device} - num_workers -> {num_workers}')
            
            logistic_model = mlrlayer.to(device)
            likelihood, true_target, timestamps = [], [], []

            for events, label in tqdm(loader):
                timestamps.append(events[0,0,:,t_index])
                if events.shape[2]==0:
                    outputs = torch.Tensor([])
                else:
                    X, ind_filtered = timesurface(events.squeeze(0).squeeze(0), (timesurface_size[0], timesurface_size[1], timesurface_size[2]), ordering, tau = tau_cla, device=device)
                    X, label = X.to(device) ,label.to(device)
                    X = X.reshape(X.shape[0], N)
                    n_events = X.shape[0]
                    outputs = logistic_model(X)
                likelihood.append(outputs.cpu().numpy())
                true_target.append(label.cpu().numpy())

            with open(results_path, 'wb') as file:
                pickle.dump([likelihood, true_target, timestamps], file, pickle.HIGHEST_PROTOCOL)

    return likelihood, true_target, timestamps

def score_classif_events(likelihood, true_target, thres=None, verbose=True):
    
    max_len = 0
    for likeli in likelihood:
        if max_len<likeli.shape[0]:
            max_len=likeli.shape[0]

    matscor = np.zeros([len(true_target),max_len])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    nancount = 0
    nb_test = len(true_target)

    for likelihood_, true_target_ in zip(likelihood, true_target):
        if len(likelihood_)!=0:
            pred_target = np.zeros(len(likelihood_))
            pred_target[:] = np.nan
            if not thres:
                pred_target = np.argmax(likelihood_, axis = 1)
            else:
                for i in range(len(likelihood_)):
                    if np.max(likelihood_[i])>thres:
                        pred_target[i] = np.argmax(likelihood_[i])
            for event in range(len(pred_target)):
                if np.isnan(pred_target[event])==False:
                    matscor[sample,event] = pred_target[event]==true_target_
            if pred_target[-1]==true_target_:
                lastac+=1
        else:
            matscor[sample,:] = 0
        sample+=1

    meanac = np.nanmean(matscor)
    onlinac = np.nanmean(matscor, axis=0)
    lastac/=nb_test
    truepos = len(np.where(matscor==1)[0])
    falsepos = len(np.where(matscor==0)[0])

    if verbose:
        print(f'Mean accuracy: {np.round(meanac,3)*100}%')
        plt.semilogx(onlinac, '.');
        plt.xlabel('number of events');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of the number of events');
    
    return meanac, onlinac, lastac, truepos, falsepos

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
            indices = np.where((timestamps_.numpy()<=time_axis[step])&(timestamps_.numpy()>time_axis[step-1]))[0]
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