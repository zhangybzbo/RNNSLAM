import numpy as np

def windowed_averaging_vote(keyframes, num_frames_to_refine=np.inf):
    num_frames_to_refine = min(num_frames_to_refine, len(keyframes))
    depths = []
    for i in range(0, num_frames_to_refine, 1):
        kf = keyframes[i]
        average_depth = kf.depth.copy()

        divider_map = np.ones_like(kf.depth, dtype=np.float)
        for j in range(len(keyframes)):
            if i == j:
                continue
            kf_ref = keyframes[j]
            # compute the project of the depth of kf_ref onto kf
            v_coords, v_depths = kf.project_pointsworld_to_imagecoord(
                kf_ref.points_world, return_depths=True)

            inside = ((v_coords[:,0] >= 0) & (v_coords[:,0] <= kf.cam.width - 1) &
                (v_coords[:,1] >= 0) & (v_coords[:,1] <= keyframes[i].cam.height - 1))
            v_coords = v_coords[inside, :]
            v_depths = v_depths[inside]
            left = np.floor(v_coords[:, 0]).astype(np.int)
            right = np.ceil(v_coords[:, 0]).astype(np.int)
            bottom = np.floor(v_coords[:, 1]).astype(np.int)
            top = np.ceil(v_coords[:, 1]).astype(np.int)

            # top left
            average_depth[top, left] += v_depths
            divider_map[top, left] += 1
            # top right
            average_depth[top, right] += v_depths
            divider_map[top, right] += 1
            # bottom left
            average_depth[bottom, left] += v_depths
            divider_map[bottom, left] += 1
            # bottom right
            average_depth[bottom, right] += v_depths
            divider_map[bottom, right] += 1
        average_depth /= divider_map
        depths.append(average_depth)

    for j, i in enumerate(range(0, num_frames_to_refine, 1)):
        keyframes[i].depth = depths[j]
        keyframes[i].points_local = None
        keyframes[i].update_points()