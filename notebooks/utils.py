import torch

def timesurface(events, sensor_size, ordering, surface_dimensions=None, tau=5e3, decay="exp"):
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
    )
    timestamp_memory -= tau * 3 + 1
    all_surfaces = torch.zeros(
        (len(events), sensor_size[2], surface_dimensions[1], surface_dimensions[0])
    )
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
    return all_surfaces

def get_loader(dataset, kfold = None, kfold_ind = 0, num_workers = 0, shuffle=True, seed=42):
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
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=subsampler, num_workers = num_workers)
    else:
        loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, num_workers = num_workers)
    return loader