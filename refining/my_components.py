import numpy as np
import g2o
import time

from threading import Lock, Thread
# from queue import Queue

# from enum import Enum
from collections import defaultdict

# from covisibility import GraphKeyFrame
# from covisibility import GraphMapPoint
# from covisibility import GraphMeasurement

from scipy.interpolate import interp2d, griddata, Rbf
from scipy.interpolate import RectBivariateSpline

class Camera(object):
    def __init__(self, fx, fy, cx, cy, width, height,
                 frustum_near, frustum_far):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fxy = np.array([fx, fy])
        self.cxy = np.array([cx, cy])
        
        self.intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 0]])
        
        self.frustum_near = frustum_near
        self.frustum_far = frustum_far

        self.width = width
        self.height = height

        self.ux = np.arange(self.width)
        self.ux_homo = (self.ux - self.cx) / self.fx
        self.uy = np.arange(self.height)
        self.uy_homo = (self.uy - self.cy) / self.fy
        self.u_grid = np.meshgrid(self.ux, self.uy)
        self.u_homo_grid = np.meshgrid(self.ux_homo, self.uy_homo)

        # this grid times depth should give points_frame
        self.ground_grid = np.dstack(
            (self.u_homo_grid + [np.ones(shape=(height, width))]))

        self.u_coords = np.concatenate([self.u_grid[0].reshape(-1,1),
            self.u_grid[1].reshape(-1,1)], axis=1)


def pad(points):
    ones = np.ones(shape=(len(points), 1))
    return np.concatenate([points, ones], axis=1)


class Frame(object):
    def __init__(self, idx, cam, depth,
        relative_pose_matrix, preceding_keyframe=None,
        image=None, depth_image=None,
        timestamp=None, sparse_points_depths=None, name=None):
        self.idx = idx

        self.pose_matrix = relative_pose_matrix
        self.relative_pose_matrix = relative_pose_matrix
        
        if preceding_keyframe:  
            self.pose_matrix = relative_pose_matrix.dot(
                preceding_keyframe.pose_matrix)

        # g2o representations
        self.pose = g2o.Isometry3d(
            self.pose_matrix[:3,:3], self.pose_matrix[:3, 3])
        self.orientation = self.pose.orientation()
        self.position = self.pose.position()

        self.cam = cam
        self.timestamp = timestamp
        # transform matrix: shape (3,4)
        self.inv_pose_matrix = np.linalg.inv(self.pose_matrix)
        self.raw_depth = depth.copy()
        self.depth = depth.copy() # depth map, predicted by CNN

        self.image = image
        self.depth_image = depth_image
        self.grayimage = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        self.grayimage_f = RectBivariateSpline(self.cam.uy, self.cam.ux, self.grayimage)

        self.colors = None
        if self.image is not None:
            self.colors = self.image.copy().reshape(-1, 3) / 255
        self.update_points()
        self.raw_points_world = self.points_world.copy()

        self.sparse_points_depths = sparse_points_depths
        # if self.sparse_points_depths is not None:
        #     self.update_sparse_points()
        self.name = name

    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose   
        self.orientation = self.pose.orientation()  
        self.position = self.pose.position()
        
        self.pose_matrix = self.pose.matrix()
        self.inv_pose_matrix = np.linalg.inv(self.pose_matrix)

        self.update_points()
    
    def project_pointsworld_to_imagecoord(self, points_world, return_depths=False):
        '''
        points_world: Nx3, should be world coord
        '''
        if return_depths:
            points, depths = self.project_pointsworld_to_homocoord(points_world, return_depths)
        else:
            points = self.project_pointsworld_to_homocoord(points_world, return_depths)
        # convert to image coord
        points = np.matmul(points, self.cam.intrinsic.T)
        if return_depths:
            return points[:, :2], depths
        else:
            return points[:, :2]

    def project_pointsworld_to_homocoord(self, points_world, return_depths=False):
        '''
        points_world: Nx3, should be world coord
        '''
        points = pad(points_world)
        points = np.matmul(points, self.inv_pose_matrix.T)[: ,:3]
        if return_depths:
            return points / points[:,2,np.newaxis], points[:,2]
        else:
            return points / points[:,2,np.newaxis]

    def sample_depth(self, v_coords):
        v_depth = self.depth_f.ev(v_coords[:, 1], v_coords[:, 0])
        return v_depth.reshape(self.cam.height, self.cam.width)
    
    def sample_depth_v2(self, v_coords, frame):
        v_points = self.sample_points(v_coords, return_pad=True)
        v_points = np.matmul(v_points, frame.inv_pose_matrix.T)
        v_depth = v_points[:, 2].reshape(frame.cam.height, frame.cam.width)
        return v_depth
    
    def sample_depth_by_homo(self, v_homo_coords):
        v_depth = self.depth_homo_f.ev(v_homo_coords[:,1], v_homo_coords[:,0])
        return v_depth.reshape(self.cam.height, self.cam.width)
    
    def sample_points(self, v_coords, return_pad=False):
        v_depth = self.depth_f.ev(v_coords[:, 1], v_coords[:, 0])
        v_homo_coords = pad((v_coords - self.cam.cxy) / self.cam.fxy)
        points_frame = v_homo_coords * v_depth[:, np.newaxis]
        points_world = np.matmul(pad(points_frame), self.pose_matrix.T)
        if return_pad:
            return points_world
        else:
            return points_world[:,3]

    def update_points(self):
        '''
        This function will give keyframe attributes:
            points_world

        Should be called each time after depth is modified
        '''
        if not hasattr(self, 'points_local') or self.points_local is None:
            print('update points from depth')
            self.depth_f = RectBivariateSpline(self.cam.uy, self.cam.ux, self.depth)
            self.depth_homo_f = RectBivariateSpline(
                self.cam.uy_homo, self.cam.ux_homo, self.depth)
            # map the depth values to 3D points (in frame coord)
            self.points_local = self.cam.ground_grid * self.depth[:, :, np.newaxis]
            # compute points in world coordinate
            self.points_local = pad(self.points_local.reshape(-1, 3))
        self.points_world = np.matmul(self.points_local, self.pose_matrix.T)[:, :3]
        self.points_world_rect = self.points_world.reshape(
            self.cam.height, self.cam.width, 3)

    def update_sparse_points(self, refine_dense=False):
        if len(self.sparse_points_depths) == 0:
            self.sparse_points_world = np.empty(shape=(0, 4))
            self.sparse_points_color = np.empty(shape=(0, 3))
            return
        sparse_points_homo = pad((self.sparse_points_depths[:, :2] - self.cam.cxy) / self.cam.fxy)
        sparse_points_frame = sparse_points_homo * self.sparse_points_depths[:, 2, np.newaxis]
        self.sparse_points_world = np.matmul(pad(sparse_points_frame), self.pose_matrix.T)
        idxs = self.sparse_points_depths[:, 0] + self.sparse_points_depths[:, 1] * self.cam.width
        idxs = idxs.astype(int)
        self.sparse_points_color = self.colors[idxs, :]

        self.depth[[self.sparse_points_depths[:,1].astype(int), self.sparse_points_depths[:,0].astype(int)]] = self.sparse_points_depths[:,2]
        self.update_points()

        # Update dense depth map using sparse landmarks
        if refine_dense:
            depth_vec = self.depth.flatten()
            depths_raw = depth_vec[idxs]
            depths_change = self.sparse_points_depths[:, 2] - depths_raw
            valid = abs(depths_change) < 0.5
            spd = self.sparse_points_depths[valid, :]
            depths_change = depths_change[valid]
            xi, yi = self.cam.u_grid

            depths_change_map = griddata(
                (spd[:,0], spd[:,1]), depths_change,
                (xi, yi), method='nearest', fill_value=0)

            self.depth += depths_change_map
            self.update_points()



class Keyframe(Frame):
    _id = 0
    _id_lock = Lock()

    def __init__(self, white_noise_variance, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)

        with Keyframe._id_lock:
            self.id = Keyframe._id
            Keyframe._id += 1
            self.update_points()
            if self.sparse_points_depths is not None:
                self.update_sparse_points()
            self.white_noise_variance = white_noise_variance
        
        self.reference_keyframe = None
        self.reference_constraint = None
        self.preceding_keyframe = None
        self.preceding_constraint = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False
    
    def measurements(self):
        # no measurements implemented
        return []

    def update_reference(self, reference=None):
        if reference is not None:
            self.reference_keyframe = reference
        self.reference_constraint = (
            self.reference_keyframe.pose.inverse() * self.pose)

    def update_preceding(self, preceding=None):
        if preceding is not None:
            self.preceding_keyframe = preceding
        self.preceding_constraint = (
            self.preceding_keyframe.pose.inverse() * self.pose)

    def set_loop(self, keyframe, constraint):
        self.loop_keyframe = keyframe
        self.loop_constraint = constraint

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed 

    def __eq__(self, other):
        return isinstance(other, Keyframe) and self.idx == other.idx

    def __lt__(self, other):
        return isinstance(other, Keyframe) and self.idx < other.idx
    
    def __le__(self, other):
        return isinstance(other, Keyframe) and self.idx <= other.idx

    def __gt__(self, other):
        return isinstance(other, Keyframe) and self.idx > other.idx
    
    def __ge__(self, other):
        return isinstance(other, Keyframe) and self.idx >= other.idx

    def _compute_uncertainty(self, d1, d2):
        return np.abs(d1 - d2)

    def compute_uncertainty_map(self):
        '''
        Compute uncertainty against previous keyframe
        This will give keyframe two new attributes:
            uncertainty_map
            uncertainty_map_f
        Both of them have the same size as self.depth
        '''
        if self.preceding_keyframe is None:
            self.uncertainty_map = np.ones_like(self.depth) * self.white_noise_variance
        else:
            # compute uncertainty map
            v_coords = self.preceding_keyframe.project_pointsworld_to_imagecoord(
                self.points_world)
            # DEBUG:
            # v_coords = self.project_pointsworld_to_imagecoord(self.points_world)
            u_coords = self.cam.u_coords
            inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < self.cam.width - 0.5) &
                    (v_coords[:,1] > -0.5) & (v_coords[:,1] < self.cam.height- 0.5))
            u_coords_outside = u_coords[~inside, :]
            #v_depth = self.preceding_keyframe.sample_depth_v2(v_coords, self)
            v_depth = self.preceding_keyframe.sample_depth(v_coords)
            # DEBUG:
            # v_depth = self.sample_depth_v2(v_coords, self)
            self.uncertainty_map = self._compute_uncertainty(self.depth, v_depth)
            self.uncertainty_map[u_coords_outside[:, 1],
                                 u_coords_outside[:, 0]] = self.white_noise_variance
        self.update_uncertainty_map_f()

    def refine(self, frame):
        '''
        This function refines the depth map and uncertainty map of a keyframe
        using small baseline new frames.
        These frames should be ordinary frames, not keyframes
        '''
        v_coords = frame.project_pointsworld_to_imagecoord(self.points_world)
        u_coords = self.cam.u_coords
        inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < self.cam.width - 0.5) &
                  (v_coords[:,1] > -0.5) & (v_coords[:,1] < self.cam.height- 0.5))
        u_coords_outside = u_coords[~inside, :]
        
        v_depth = frame.sample_depth_v2(v_coords, self)
        # v_depth = frame.sample_depth(v_coords)
        v_uncertainty_map = self._compute_uncertainty(self.depth, v_depth)
        v_uncertainty_map[u_coords_outside[:, 1],
                          u_coords_outside[:, 0]] = self.white_noise_variance

        total_uncertainty_map = self.uncertainty_map + v_uncertainty_map

        new_depth = (v_uncertainty_map * self.depth + 
                     self.uncertainty_map * v_depth) / total_uncertainty_map
        new_uncertainty_map = self.uncertainty_map * v_uncertainty_map / total_uncertainty_map
        new_depth[u_coords_outside[:,1], u_coords_outside[:,0]] = (
            self.depth[u_coords_outside[:,1], u_coords_outside[:,0]])
        new_uncertainty_map[u_coords_outside[:,1], u_coords_outside[:,0]] = (
            self.uncertainty_map[u_coords_outside[:,1], u_coords_outside[:,0]])
        
        self.depth = new_depth
        self.uncertainty_map = new_uncertainty_map
        
        self.update_points()
        self.update_uncertainty_map_f()

    def sample_uncertainty_map(self, v_coords):
        v_uncertainty_map = self.uncertainty_map_f.ev(v_coords[:, 1], v_coords[:, 0])
        v_uncertainty_map = np.abs(v_uncertainty_map)
        return v_uncertainty_map.reshape(self.cam.height, self.cam.width)

    def update_uncertainty_map_f(self):
        self.uncertainty_map_f = RectBivariateSpline(
            self.cam.uy, self.cam.ux, self.uncertainty_map)
    
    def fuse_with_preceding_keyframe(self):
        if self.preceding_keyframe is None:
            return
        # compute uncertainty map
        v_coords = self.preceding_keyframe.project_pointsworld_to_imagecoord(self.points_world)
        u_coords = self.cam.u_coords
        inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < self.cam.width - 0.5) &
                  (v_coords[:,1] > -0.5) & (v_coords[:,1] < self.cam.height- 0.5))
        u_coords_outside = u_coords[~inside, :]

        v_depth = self.preceding_keyframe.sample_depth_v2(v_coords, self)
        # v_depth = self.preceding_keyframe.sample_depth(v_coords)

        v_uncertainty_map = self.preceding_keyframe.sample_uncertainty_map(v_coords)

        new_uncertainty_map = self._compute_uncertainty(self.depth, v_depth)
        new_uncertainty_map[u_coords_outside[:, 1], u_coords_outside[:, 0]] = (
            self.uncertainty_map[u_coords_outside[:, 1], u_coords_outside[:, 0]])
        self.uncertainty_map = new_uncertainty_map

        # compute propagated uncertainty map
        propagated_uncertainty_map = (np.abs(v_depth / self.depth) *
            v_uncertainty_map + self.white_noise_variance)

        # fuse current frame with nearest_keyframe
        total_uncertainty_map = self.uncertainty_map + propagated_uncertainty_map
        new_depth = (propagated_uncertainty_map * self.depth +
            self.uncertainty_map * v_depth) / total_uncertainty_map

        new_uncertainty_map = (self.uncertainty_map * propagated_uncertainty_map /
            total_uncertainty_map)
        new_uncertainty_map[u_coords_outside[:, 1], u_coords_outside[:, 0]] = (
            self.uncertainty_map[u_coords_outside[:, 1], u_coords_outside[:, 0]])
        new_depth[u_coords_outside[:, 1], u_coords_outside[:, 0]] = (
            self.depth[u_coords_outside[:, 1], u_coords_outside[:, 0]])
        self.depth = new_depth
        self.uncertainty_map = new_uncertainty_map

        # update depth map and uncertainty map
        self.update_points()
        self.update_uncertainty_map_f()
    
    def simple_fuse_with_preceding_keyframe(self, weight=0.95):
        if self.preceding_keyframe is None:
            return
        v_coords = self.preceding_keyframe.project_pointsworld_to_imagecoord(
            self.points_world)
        u_coords = self.cam.u_coords
        inside = ((v_coords[:,0] > -0.5) & (v_coords[:,0] < self.cam.width - 0.5) &
                  (v_coords[:,1] > -0.5) & (v_coords[:,1] < self.cam.height- 0.5))
        u_coords_outside = u_coords[~inside, :]

        v_depth = self.preceding_keyframe.sample_depth_v2(v_coords, self)
        # v_depth = self.preceding_keyframe.sample_depth(v_coords)
        new_depth = v_depth

        new_depth[u_coords_outside[:,1], u_coords_outside[:,0]] = (
            self.depth[u_coords_outside[:,1], u_coords_outside[:,0]])
        self.depth = self.depth * (1.0-weight) + new_depth * weight
        #self.depth = new_depth
        self.update_points()
    
    def get_vis_points(self, use_sparse=True):

        if use_sparse:
            if len(self.sparse_points_depths) == 0:
                vis_points = np.empty(shape=(0, 4))
                vis_colors = np.empty(shape=(0, 3))
                return vis_points, vis_colors
            depth_vec = self.sparse_points_depths[:, 2]
            valid = depth_vec < 5000
            vis_points = self.sparse_points_world[valid, :]
            vis_colors = self.sparse_points_color[valid, :]
            return vis_points, vis_colors

        depth_vec = self.depth.reshape(-1)
        # uncertainty_vec = self.uncertainty_map.reshape(-1)
        valid = depth_vec < 5000
        # if self._id <= 5:
        #     valid = depth_vec < 50
        # else:
        #     valid = (depth_vec < 50) & (uncertainty_vec < 10)
        # print(uncertainty_vec)
        # print(valid)
        vis_points = self.points_world[valid, :]
        
        if self.colors is not None:
            vis_colors = self.colors[valid, :]
        else:
            vis_colors = None
        # print('{} points'.format(len(vis_colors)))
        return vis_points, vis_colors

