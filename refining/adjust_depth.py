import numpy as np
from skimage.measure import compare_ssim
from PIL import Image
from scipy.signal import convolve2d
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.spatial import cKDTree
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
import multiprocessing as mp
from functools import partial
import threading
from concurrent.futures import ThreadPoolExecutor


def fetch_image(kf, kf_ref, fill='kf', debug=False):
    image_ref = None
    if fill == 'kf':
        image_ref = kf.grayimage.copy()
    elif fill == 'zero':
        image_ref = np.zeros_like(kf.grayimage)
    else:
        raise AttributeError('Unrecognized fill type: {}'.format(fill))

    v_coords = kf_ref.project_pointsworld_to_imagecoord(kf.points_world)
    inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < kf_ref.cam.width - 0.5) &
              (v_coords[:,1] > -0.5) & (v_coords[:,1] < kf_ref.cam.height- 0.5))
    v_coords_inside = v_coords[inside, :]
    u_coords = kf.cam.u_coords
    u_coords_inside = u_coords[inside, :]
    u_coords_outside = u_coords[~inside, :]

    # TO DO: implement image_f in keyframe class
    v_image_gray = kf_ref.grayimage_f.ev(v_coords_inside[:, 1], v_coords_inside[:, 0])
    image_ref[u_coords_inside[:, 1], u_coords_inside[:, 0]] = v_image_gray

    # DEBUG
    if debug:
        width, height = image_ref.shape
        new_im = Image.new('L', (width*3, height))
        new_im.paste(Image.fromarray(kf_ref.grayimage), (0,0))
        new_im.paste(Image.fromarray(image_ref), (width, 0))
        new_im.paste(Image.fromarray(kf.grayimage), (width*2, 0))
        new_im.show()
        exit()        # p.map_async(f, list(range(len(keyframes))))
        # p.close()
        # p.join()
    
    valid_map = np.ones_like(image_ref, dtype=np.float)
    valid_map[u_coords_outside[:,1], u_coords_outside[:,0]] = np.zeros(shape=(len(u_coords_outside)))


    return image_ref, u_coords_outside, valid_map

def compute_error_map(image, image_ref, u_coords_outside=None, fill_value=0.0):
    #error_map = (image - image_ref)**2
    _, error_map = compare_ssim(image, image_ref, full=True)
    error_map = -error_map
    if u_coords_outside is not None:
        error_map[u_coords_outside[:,1], u_coords_outside[:,0]] = np.full((len(u_coords_outside)), fill_value)
    return error_map

def arrange_choices(choices):
    start = len(choices) // 2
    offset = 0
    rv = np.empty_like(choices)
    for i in range(len(rv)):
        rv[i] = choices[start + offset]
        if offset == 0:
            offset = -1
        elif offset < 0:
            offset = -offset
        else:
            offset = -offset - 1
    return rv
        

def windowed_dense_depth_adjustment(keyframes, id, search_range=[-0.2, 0.2], search_steps=21):
    for i in range(len(keyframes)):
        if(i != id):
            continue
        kf = keyframes[i]
        error_maps = np.zeros(shape=(kf.grayimage.shape[0], kf.grayimage.shape[1], search_steps))
        search_choices = np.linspace(search_range[0], search_range[1], num=search_steps)
        search_choices = arrange_choices(search_choices)
        original_depth = kf.depth.copy()
        for d, delta_depth in enumerate(search_choices):
            valid_count_map = np.zeros_like(kf.grayimage, dtype=np.float)
            # adjust depth attempts
            kf.depth = original_depth + delta_depth
            kf.update_points()
            for j in range(len(keyframes)):
                kf_ref = keyframes[j]
                if i != j:
                    weight = 1.0
                    # fetch colors from kf_ref to kf's coordinate
                    image_ref, u_coords_outside, valid_map = fetch_image(kf, kf_ref, fill='zero')
                    valid_count_map += valid_map
                    # compute pixel-wise error and accumulate
                    error_maps[:, :, d] += weight * compute_error_map(kf.grayimage, image_ref, u_coords_outside=u_coords_outside, fill_value=0.0)
            error_maps[:,:,d] /= valid_count_map
            
            # print(np.count_nonzero(np.isnan(error_maps[:,:,d])))
        # get the depth with least error
        best_d = np.argmin(error_maps, axis=2)
        def f(d):
            return search_choices[d]
        vfunc = np.vectorize(f)
        best_delta_depth = vfunc(best_d)
        kf.depth = original_depth + best_delta_depth
        kf.update_points()

def windowed_averaging(keyframes, num_frames_to_refine=np.inf):
    num_frames_to_refine = min(num_frames_to_refine, len(keyframes))
    depths = []
    for i in range(0, num_frames_to_refine, 2):
        kf = keyframes[i]
        average_depth = kf.depth.copy()

        divider_map = np.ones_like(kf.depth, dtype=np.float)
        for j in range(len(keyframes)):
            if i == j:
                continue
            # weight = 1.0 # np.sqrt(1.0 / abs(j-i))
            kf_ref = keyframes[j]
            # compute the project of the depth of kf_ref onto kf
            v_coords, v_depths = kf.project_pointsworld_to_imagecoord(
                kf_ref.points_world, return_depths=True)

            inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < keyframes[i].cam.width - 0.5) &
                (v_coords[:,1] > -0.5) & (v_coords[:,1] < keyframes[i].cam.height- 0.5))
            v_coords = v_coords[inside, :]
            v_depths = v_depths[inside]
            v_coords = np.round(v_coords).astype(np.int)
            average_depth[v_coords[:,1], v_coords[:,0]] += v_depths
            divider_map[v_coords[:,1], v_coords[:,0]] += 1
        average_depth /= divider_map

        depths.append(average_depth)
    for j, i in enumerate(range(0, num_frames_to_refine, 2)):
        keyframes[i].depth = depths[j]
        keyframes[i].update_points()


def f_average_mp(i, keyframes):
    kf = keyframes[i]
    average_depth = kf.depth.copy()
    divider_map = np.ones_like(kf.depth, dtype=np.float)
    for j in range(len(keyframes)):
        if i == j:
            continue
        # weight = 1.0 # np.sqrt(1.0 / abs(j-i))
        kf_ref = keyframes[j]
        # compute the project of the depth of kf_ref onto kf
        v_coords, v_depths = kf.project_pointsworld_to_imagecoord(
            kf_ref.points_world, return_depths=True)

        inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < keyframes[i].cam.width - 0.5) &
            (v_coords[:,1] > -0.5) & (v_coords[:,1] < keyframes[i].cam.height- 0.5))
        v_coords = v_coords[inside, :]
        v_depths = v_depths[inside]
        v_coords = np.round(v_coords).astype(np.int)
        average_depth[v_coords[:,1], v_coords[:,0]] += v_depths
        divider_map[v_coords[:,1], v_coords[:,0]] += 1
    average_depth /= divider_map
    return average_depth


def windowed_averaging_mp(keyframes):
    pool = mp.Pool(2)
    f = partial(f_average_mp, keyframes=keyframes)
    new_depths = pool.map(f, range(len(keyframes)))
    pool.close()
    for i in range(len(keyframes)):
        depth_map = new_depths[i]
        keyframes[i].depth = depth_map
        keyframes[i].update_points()


def windowed_averaging_ofraw(keyframes, id):
    kf = keyframes[id]
    average_depth = kf.depth.copy()
    divider_map = np.ones_like(kf.raw_depth, dtype=np.float)
    for j in range(len(keyframes)):
        if id == j:
            continue
        # weight = 1.0 # np.sqrt(1.0 / abs(j-i))
        kf_ref = keyframes[j]
        # compute the project of the depth of kf_ref onto kf
        v_coords, v_depths = kf.project_pointsworld_to_imagecoord(
            kf_ref.points_world, return_depths=True)

        inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < kf.cam.width - 0.5) &
            (v_coords[:,1] > -0.5) & (v_coords[:,1] < kf.cam.height- 0.5))
        v_coords = v_coords[inside, :]
        v_depths = v_depths[inside]
        v_coords = np.round(v_coords).astype(np.int)
        average_depth[v_coords[:,1], v_coords[:,0]] += v_depths
        divider_map[v_coords[:,1], v_coords[:,0]] += 1
    average_depth /= divider_map
    keyframes[id].depth = average_depth
    keyframes[id].update_points()


def modified_ip_call(ip, *args):
    xi = _ndim_coords_from_arrays(args, ndim=ip.points.shape[1])
    xi = ip._check_call_shape(xi)
    xi = ip._scale_x(xi)
    dist, i = ip.tree.query(xi, n_jobs=-1)
    return dist, ip.values[i]


def f_average(i, keyframes, depths, method='linear'):
    '''
    depths: shared memory 1D
    '''
    print('Processing {}'.format(i))
    kf = keyframes[i]
    depth_map_size = kf.depth.shape[0]*kf.depth.shape[1]

    average_depth = kf.depth.copy()

    divider_map = np.ones_like(kf.depth, dtype=np.float)
    for j in range(len(keyframes)):
        if i == j:
            continue
        weight = 1.0 # np.sqrt(1.0 / abs(j-i))
        kf_ref = keyframes[j]
        # compute the project of the depth of kf_ref onto kf
        v_coords, v_depths = kf.project_pointsworld_to_imagecoord(
            kf_ref.points_world, return_depths=True)
        xi, yi = kf.cam.u_grid
        # Use griddata warpper
        if method == 'linear':
            depth_proj = griddata(v_coords, v_depths, (xi, yi), method='linear')
            outside_map = np.isnan(depth_proj)
        # Directly use NearestNDInterpolator
        else:
            distance_upper_bound = 4
            ip = NearestNDInterpolator(v_coords, v_depths)
            dist, depth_proj = modified_ip_call(ip, (xi, yi))
            outside_map = dist > distance_upper_bound
        
        
        depth_proj[outside_map] = 0
        # weight_map = 1.0 / (np.abs(kf.depth - depth_proj) + 1)
        # weight_map[outside_map] = 0
        average_depth += weight * depth_proj #* weight_map
        divider_map += weight * (~outside_map) #* weight_map
    average_depth /= divider_map

    depths[depth_map_size*i : depth_map_size*(i+1)] = average_depth.flatten()

def f_average_naive(i, keyframes, depths):
    '''
    depths: shared memory 1D
    '''
    print('Processing {}'.format(i))
    kf = keyframes[i]
    depth_map_size = kf.depth.shape[0]*kf.depth.shape[1]

    average_depth = kf.depth.copy()

    divider_map = np.ones_like(kf.depth, dtype=np.float)
    for j in range(len(keyframes)):
        if i == j:
            continue
        # weight = 1.0 # np.sqrt(1.0 / abs(j-i))
        kf_ref = keyframes[j]
        # compute the project of the depth of kf_ref onto kf
        v_coords, v_depths = kf.project_pointsworld_to_imagecoord(
            kf_ref.points_world, return_depths=True)

        inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < keyframes[i].cam.width - 0.5) &
            (v_coords[:,1] > -0.5) & (v_coords[:,1] < keyframes[i].cam.height- 0.5))
        v_coords = v_coords[inside, :]
        v_depths = v_depths[inside]
        v_coords = np.round(v_coords).astype(np.int)
        average_depth[v_coords[:,1], v_coords[:,0]] += v_depths
        divider_map[v_coords[:,1], v_coords[:,0]] += 1

    average_depth /= divider_map
    depths[depth_map_size*i : depth_map_size*(i+1)] = average_depth.flatten()

def windowed_averaging_multithread(keyframes):
    height, width = keyframes[0].depth.shape
    depth_map_size = height * width

    # get a copy of current depth maps
    depths = np.empty(shape=(height*width*len(keyframes)))
    depths = mp.Array('d', depths)
    print('Averaging {} frames in local window [multithreaded]'.format(len(keyframes)))
    
    ps = []
    for i in range(len(keyframes)):
        p = mp.Process(target=f_average, args=(i, keyframes, depths))
        ps.append(p)
        ps[i].start()            
    for i in range(len(keyframes)):
        ps[i].join()
    
    print('Done!')

    print('Updating')
    # update keyframes using the refined copies of depth maps
    # pool = mp.Pool(len(keyframes))
    # f = partial(f_update, keyframes=keyframes, height=height, width=width, depths=np.array(depths))
    # pool.map_async(f, range(len(keyframes)))
    for i in range(len(keyframes)):
        depth_map = np.array(depths[depth_map_size*i : depth_map_size*(i+1)])
        keyframes[i].depth = depth_map.reshape((height, width))
        keyframes[i].points_local = None
        keyframes[i].update_points()

def eliminate_conv_boundary_effect(height, width, num_conv=1, kernel_size=51):
    regularizer = np.ones(shape=(height, width), dtype=np.float)
    cx = width // 2
    cy = height // 2
    for x in range(width):
        for y in range(height):
            d = np.sqrt((x-cx)**2+(y-cy)**2) / (height + width) * 2
            regularizer[y,x] = 1 - 0.9 * d**2
    return regularizer
    # idepth_map = np.ones(shape=(width, height), dtype=np.float)
    # kernel = np.ones(shape=(kernel_size,kernel_size), dtype=np.float)
    # kernel /= (kernel_size**2)
    # for _ in range(num_conv):
    #     idepth_map = convolve2d(idepth_map, kernel, mode='same', boundary='fill', fillvalue=0.0)
    # depth_map = 1.0 / idepth_map
    # return depth_map

    
