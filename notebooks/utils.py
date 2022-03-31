import torch, tonic
import numpy as np
import matplotlib.pyplot as plt

def timesurface(events, sensor_size, ordering, surface_dimensions=None, tau=5e3, decay="exp", filtering_threshold = None, device="cpu"):
    '''with tonic events is loaded in a standardized format: event -> (x,y,t,p) 
       TODO : use tonic function to apply to_timesurface in a clean way.
    '''
    
    x_index = ordering.index('x')
    y_index = ordering.index('y')
    t_index = ordering.index('t')
    p_index = ordering.index('p')
    
    if surface_dimensions:
        assert len(surface_dimensions) == 2
        assert surface_dimensions[0] % 2 == 1 and surface_dimensions[1] % 2 == 1
        radius_x = surface_dimensions[0] // 2
        radius_y = surface_dimensions[1] // 2
    else:
        radius_x = 0
        radius_y = 0
        surface_dimensions = sensor_size

    timestamp_memory = torch.zeros(
        (sensor_size[2], sensor_size[1] + radius_y * 2, sensor_size[0] + radius_x * 2)
    ).to(device)
    timestamp_memory -= tau * 3 + 1
    all_surfaces = torch.zeros(
        (len(events), sensor_size[2], surface_dimensions[1], surface_dimensions[0])
    ).to(device)
    for index, event in enumerate(events):
        x = int(event[x_index])
        y = int(event[y_index])
        timestamp_memory[int(event[p_index]), y + radius_y, x + radius_x] = event[t_index]
        if radius_x > 0 and radius_y > 0:
            timestamp_context = (
                timestamp_memory[
                    :, y : y + surface_dimensions[1], x : x + surface_dimensions[0]
                ]
                - event[t_index]
            )
        else:
            timestamp_context = timestamp_memory - event[t_index]

        if decay == "lin":
            timesurface = timestamp_context / (3 * tau) + 1
            timesurface[timesurface < 0] = 0
        elif decay == "exp":
            timesurface = torch.exp(timestamp_context / tau)
        all_surfaces[index, :, :, :] = timesurface
        indices = None
    if filtering_threshold:
        indices = torch.nonzero(all_surfaces.sum(dim=(1,2,3))>filtering_threshold).squeeze(1)
        all_surfaces = all_surfaces[indices, :, :, :]
    return all_surfaces, indices

def get_loader(dataset, kfold = None, kfold_ind = 0, num_workers = 0, shuffle=True, batch_size=None, seed=42):
    # creates a loader for the samples of the dataset. If kfold is not None, 
    # then the dataset is splitted into different folds with equal repartition of the classes.
    if kfold:
        subset_indices = []
        subset_size = len(dataset)//kfold
        for i in range(len(dataset.classes)):
            all_ind = np.where(np.array(dataset.targets)==i)[0]
            subset_indices += all_ind[kfold_ind*subset_size//len(dataset.classes):
                            min((kfold_ind+1)*subset_size//len(dataset.classes), len(dataset)-1)].tolist()
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        subsampler = torch.utils.data.SubsetRandomSampler(subset_indices, g_cpu)
        if batch_size:
            loader = torch.utils.data.DataLoader(dataset, shuffle=False, sampler=subsampler, num_workers = num_workers, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=subsampler, num_workers = num_workers)
    else:
        if batch_size:
            loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, num_workers = num_workers, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))
        else: 
            loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, num_workers = num_workers)
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

def plot_kernels(layer, pola, R, width=20):
    kernel = layer.synapses.weight.data.T
    fig = plt.figure(figsize=(width,pola/kernel.shape[1]*width))
    for n in range(len(kernel[0,:])):
        for p in range(pola):
            sub = fig.add_subplot(pola,len(kernel[0,:]),n+len(kernel[0,:])*p+1)
            dico = np.reshape(kernel[p*(2*R+1)**2:(p+1)*(2*R+1)**2,n], [int(np.sqrt(len(kernel)/pola)), int(np.sqrt(len(kernel)/pola))])
            sub.imshow((dico), cmap=plt.cm.plasma)
            sub.axes.get_xaxis().set_visible(False)
            sub.axes.get_yaxis().set_visible(False)
    plt.show()
    
def plot_weight_distribution(layer, bins=np.linspace(0, 1, 50)):
    kernels = layer.synapses.weight.data
    fig, axs = plt.subplots(1,2, figsize=(12, 6))
    n_neurons = kernels.size(dim=0)
    ts_size = int(kernels.size(dim=1)/2)
    for k in range (n_neurons):
        pos_kernels = kernels[k][ts_size:]
        neg_kernels = kernels[k][:ts_size]
        axs[0].hist(neg_kernels, bins=bins, alpha=.1)
        axs[0].set_xlabel('OFF polarities', fontsize=16)
        axs[1].hist(pos_kernels, bins=bins, alpha=.1)
        axs[1].set_xlabel('ON polarities', fontsize=16)
    plt.show()