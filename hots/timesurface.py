import torch

def timesurface(events, sensor_size, ordering, surface_dimensions=None, tau=5e3, decay="exp", filtering_threshold = None, device="cpu", dtype='torch.float32'):
    '''with tonic events is loaded in a standardized format: event -> (x,y,t,p) 
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
            timesurface[timesurface<torch.exp(torch.tensor(-5))] = 0
        all_surfaces[index, :, :, :] = timesurface
    indices = None
    if filtering_threshold:
        indices = torch.nonzero(all_surfaces.sum(dim=(1,2,3))>filtering_threshold).squeeze(1)
        all_surfaces = all_surfaces[indices, :, :, :]
    return all_surfaces, indices


def snnsurface(events, sensor_size, ordering, surface_dimensions=None, tau=5e3, decay="exp", filtering_threshold = None, device="cpu"):
    '''with tonic events is loaded in a standardized format: event -> (x,y,t,p) 
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
        timestamp_memory[int(event[p_index]), y + radius_y, x + radius_x] += event[t_index]
        
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
            timesurface[timesurface<torch.exp(torch.tensor(-5))] = 0
        all_surfaces[index, :, :, :] = timesurface
        indices = None
    if filtering_threshold:
        indices = torch.nonzero(all_surfaces.sum(dim=(1,2,3))>filtering_threshold).squeeze(1)
        all_surfaces = all_surfaces[indices, :, :, :]
    return all_surfaces, indices