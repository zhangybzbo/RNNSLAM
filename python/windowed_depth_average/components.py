import numpy as np
import time
from collections import defaultdict

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

        self.cam = cam
        self.timestamp = timestamp
        # transform matrix: shape (3,4)
        self.inv_pose_matrix = np.linalg.inv(self.pose_matrix)
        self.depth = depth.copy() # depth map, predicted by CNN

        self.image = image
        self.depth_image = depth_image

        self.colors = None
        if self.image is not None:
            self.colors = self.image.copy().reshape(-1, 3) / 255
        self.update_points()

        self.name = name
    
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

    def update_points(self):
        '''
        This function will give keyframe attributes:
            points_world

        Should be called each time after depth is modified
        '''
        if not hasattr(self, 'points_local') or self.points_local is None:
            # map the depth values to 3D points (in frame coord)
            self.points_local = self.cam.ground_grid * self.depth[:, :, np.newaxis]
            # compute points in world coordinate
            self.points_local = pad(self.points_local.reshape(-1, 3))
        self.points_world = np.matmul(self.points_local, self.pose_matrix.T)[:, :3]
        self.points_world_rect = self.points_world.reshape(
            self.cam.height, self.cam.width, 3)

    def get_vis_points(self):

        depth_vec = self.depth.reshape(-1)
        valid = depth_vec < 5000
        vis_points = self.points_world[valid, :]
        
        if self.colors is not None:
            vis_colors = self.colors[valid, :]
        else:
            vis_colors = None
        return vis_points, vis_colors