import glob
import numpy as np
from my_viewer_gap_demo import MapViewer
from my_components import Frame, Camera, Keyframe
import time
from PIL import Image
import cv2
import g2o
import argparse
import os
import struct
import shutil
from adjust_depth import windowed_dense_depth_adjustment, eliminate_conv_boundary_effect
from adjust_depth import windowed_averaging, windowed_averaging_mp, windowed_averaging_ofraw
from adjust_depth import windowed_averaging_multithread
import time

import sys
sys.path.append('/home/ruibinma/software/dso/src/RNN')
import RNN_pred_colon_reproj

from my_loopclosing import LoopClosing

class MySystem(object):

    def __init__(self, white_noise_variance, local_window_size=7, viewer=None):
        self.poses = []
        self.trajectory = []
        self.points = np.empty(shape=(0, 3))
        self.colors = np.empty(shape=(0, 3))
        self.sparse_points = np.empty(shape=(0, 3))
        self.sparse_colors = np.empty(shape=(0, 3))
        self.image = None
        self.current_keyframe = None
        self.keyframes = []
        self.local_window_size = local_window_size
        self.white_noise_variance = white_noise_variance
        self.total_uncertainty = 0

        self.allpoints = []
        self.allcolors = []

        self.viewer = viewer

        self.old_trajectory = []
        self.old_keyframes = []
        self.connection_edge = []
        self.old_poses = []
    
    def start_new_chunk(self):
        self.old_keyframes = self.keyframes
        self.old_trajectory = []
        self.old_poses = []
        self.connection_edge = []
        self.keyframes = []
        self.poses = []
        self.trajectory = []
        self.points = np.empty(shape=(0, 3))
        self.colors = np.empty(shape=(0, 3))
        self.sparse_points = np.empty(shape=(0, 3))
        self.sparse_colors = np.empty(shape=(0, 3))
        self.image = None
        self.current_keyframe = None
        self.allpoints = []
        self.allcolors = []
    
    def should_add_keyframe(self, frame):
        if self.current_keyframe is None:
            return True
        if frame.idx - self.current_keyframe.idx >= 1:
            return True
    
    def save_refined_depth(self, kf_tosave, refined_depth_dir):
        refined_depth_path = os.path.join(refined_depth_dir, kf_tosave.name)
        kf_tosave.depth.astype(np.float32).tofile(refined_depth_path)
        print('updated {}'.format(refined_depth_path))
    
    def add_keyframe(self, keyframe, refined_depth_dir=None):

        # simply sequentially display depth map and accumulated camera poses
        self.keyframes.append(keyframe)
        self.current_keyframe = keyframe

    def refresh_display_content(self, add_pose=False, concat=False, deep=False, keep_first=False,
                                display_old_trajectory=False):
        if add_pose and self.current_keyframe:
            self.poses.append(self.current_keyframe.pose_matrix)
            self.trajectory.append(self.current_keyframe.pose_matrix[0:3, 3])

        if deep:
            self.poses = []
            for kf in self.keyframes:
                self.poses.append(kf.pose_matrix)
            self.trajectory = []
            for kf in self.keyframes:
                self.trajectory.append(kf.pose_matrix[0:3, 3])
        if len(self.keyframes) > 0:
            points, colors = self.keyframes[-1].get_vis_points(use_sparse=False)
            points = points[:, :3]
        else:
            points = np.empty(shape=(0, 3))
            colors = np.empty(shape=(0, 3))

        if keep_first:
            points_first, colors_first = self.keyframes[0].get_vis_points(use_sparse=False)
            points_first = points_first[:, :3]
            self.points = np.concatenate([points_first, points], axis=0)
            self.colors = np.concatenate([colors_first, colors], axis=0)
            average_dist = np.mean(np.sqrt(np.sum((points_first - points)**2, axis=1)))
            print("Average point distance = {}".format(average_dist))
        elif concat:
            self.points = np.concatenate([self.points, points], axis=0)
            self.colors = np.concatenate([self.colors, colors], axis=0)
        else:
            self.points = points
            self.colors = colors
        if self.current_keyframe:
            self.image = np.rot90(np.rot90(self.current_keyframe.image))
            # self.image = self.current_keyframe.image
            depth = self.current_keyframe.depth
            max_depth = np.max(depth)
            depth = depth / max_depth * 255
            self.depth = np.rot90(np.rot90(depth.astype(np.uint8)))

        if display_old_trajectory and hasattr(self, 'old_keyframes') and self.old_keyframes:
            print('displaying {} old keyframes'.format(len(self.old_keyframes)))
            self.old_trajectory = []
            for kf in self.old_keyframes:
                self.old_trajectory.append(kf.pose_matrix[0:3, 3])
            self.old_poses = []
            for kf in self.old_keyframes:
                self.old_poses.append(kf.pose_matrix)

        if self.viewer:
            self.viewer.update()
    
    def add_connection_edge(self, edge):
        # should be a list of two 3-D points
        self.connection_edge = edge

    def refresh_display_content_v2(self, add_pose=False):
        if add_pose:
            self.poses.append(self.current_keyframe.pose_matrix)
        
        sparse_points, sparse_colors = self.current_keyframe.get_vis_points(use_sparse=True)
        dense_points, dense_colors = self.current_keyframe.get_vis_points(use_sparse=False)
        sparse_points = sparse_points[:, :3]
        dense_points = dense_points[:, :3]
        print(sparse_points.shape, self.sparse_points.shape)
        self.sparse_points = np.concatenate([sparse_points,self.sparse_points],axis=0)
        self.sparse_colors = np.concatenate([sparse_colors,self.sparse_colors],axis=0)
        
        self.points = np.concatenate([self.sparse_points, dense_points],axis=0)
        self.colors = np.concatenate([self.sparse_colors, dense_colors],axis=0)

        self.image = np.flip(self.current_keyframe.image, axis=0)
        depth = self.current_keyframe.depth
        max_depth = np.max(depth)
        depth = depth / max_depth * 255

        self.depth = np.flip(depth.astype(np.uint8), axis=0)

    def refresh_display_content_window(self, add_pose=False):
        if add_pose:
            self.poses.append(self.current_keyframe.pose_matrix)
        self.trajectory.append(self.current_keyframe.pose_matrix[0:3, 3])


        points = []
        colors = []
        for keyframe in self.keyframes:
            p, c = keyframe.get_vis_points(use_sparse=False)
            points.append(p)
            colors.append(c)
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        points = points[:, :3]


        self.points = points
        self.colors = colors
        self.image = np.rot90(np.rot90(self.current_keyframe.image))
        depth = self.current_keyframe.depth
        max_depth = np.max(depth)
        depth = depth / max_depth * 255

        self.depth = np.rot90(np.rot90(depth.astype(np.uint8)))


def load_depth(file_path, height, width, fx, fy, cx, cy):
    z = np.fromfile(file_path, dtype=np.float32).reshape(height, width)
    grid = np.meshgrid((np.arange(width)  - cx) / fx, (np.arange(height) - cy) / fy)
    points_3D = np.dstack((grid + [np.ones_like(z)])) * z[:,:,np.newaxis]
    return points_3D, z

def presort(file_path):
    poses = {}
    with open(file_path, 'r') as file:
        lines = file.read().split('\n')
        if len(lines[-1]) <= 1:
            del lines[-1]
        for mline in lines:
            m = mline.split(' ')
            assert(len(m) > 12)
            assert(int(m[12]) not in poses)
            poses[int(m[12])] = mline
    keys = sorted(poses.keys())
    with open(file_path, 'w') as file:
        for key in keys:
            file.write('{}\n'.format(poses[key]))

def load_sparse_points_depths(sparse_depth_path, args):
    with open(sparse_depth_path, 'rb') as sparse_file:
        num_sparse_points = struct.unpack('i', sparse_file.read(4))[0]
        print("{} sparse points".format(num_sparse_points))
        sparse_points_depths = []
        for _ in range(num_sparse_points):
            u = struct.unpack('f', sparse_file.read(4))[0]
            v = struct.unpack('f', sparse_file.read(4))[0]
            u = u / 320. * 270.
            v = v / 256. * 216.
            idepth = struct.unpack('f', sparse_file.read(4))[0]
            idepth_scaled = struct.unpack('f', sparse_file.read(4))[0]
            idepth_hessian = struct.unpack('f', sparse_file.read(4))[0]
            maxRelBaseline = struct.unpack('f', sparse_file.read(4))[0]
            numGoodResiduals = struct.unpack('i', sparse_file.read(4))[0]
            depth = 1.0 / idepth
            var = (1.0 / (idepth_hessian+0.01))
            if var * pow(depth, 4) > args.relVarTH:
                continue
            if var > args.absVarTH:
                continue
            if maxRelBaseline < args.minRelativeBS:
                continue

            sparse_point_depth = []
            sparse_point_depth.append(u)
            sparse_point_depth.append(v)
            sparse_point_depth.append(1.0 / idepth_scaled)
            sparse_points_depths.append(sparse_point_depth)
    sparse_points_depths = np.array(sparse_points_depths)
    return sparse_points_depths


def write_all_keyframes(system, refined_depth_dir):
    for kf_tosave in system.keyframes:
        system.save_refined_depth(kf_tosave, args.refined_depth_dir)


def main(args):

    # read camera intrinsics
    with open(args.intrinsic, 'r') as file:
        line = file.readline().split(' ')
        fx = float(line[1])
        fy = float(line[2])
        cx = float(line[3])
        cy = float(line[4])
        line = file.readline().split(' ')
        width = int(line[0])
        height = int(line[1])
        line = file.readline()
        line = file.readline().split(' ')
        dso_width = int(line[0])
        dso_height = int(line[1])
    print("Camera parameters: [{} {}] {} {} {} {}".format(
        width, height, fx, fy, cx, cy))


    cam = Camera(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height,
        frustum_near=0.01, frustum_far=1000.)
    
    image_names = sorted(os.listdir(args.image_dir))
    image_file_paths = [os.path.join(args.image_dir, name) for name in image_names]

    # initialize system and map viewer and loop closure thread
    system = MySystem(white_noise_variance=50., local_window_size=args.local_window_size)
    print("System initialized.")
    if args.use_viewer:
        viewer = MapViewer(system, w=width, h=height)
        print("Viewer thread initialized.")
        system.viewer = viewer
    if args.use_loopclosure:
        loop_closing = LoopClosing(system, None)
        print("Loop closure thread initialized.")
    count_kf = 0
    start_time = time.time()

    # run rnn to get camera pose & depth map for each frame
    net = RNN_pred_colon_reproj.RNN_depth_pred(args.rnnmodel)
    # ===== Bootstrap RNN
    bootstrap_steps = 9
    # make a artificial gap
    break_frame = 50
    break_frame_name = os.path.basename(image_file_paths[break_frame])
    sequence1 = image_file_paths[:break_frame]
    sequence2 = image_file_paths[break_frame:]
    temp = list(reversed(sequence2))
    temp.extend(sequence2[1:])
    sequence2 = temp

    image_file_paths = sequence1

    interruption = [None] * 100
    for seq, image_file_paths in enumerate([sequence1, interruption, sequence2]):

        for i, image_file_path in enumerate(image_file_paths):
            if seq == 1:
                time.sleep(0.05)
                continue
            
            if i == 0:
                net.assign_keyframe_by_path(image_file_path)
            if i < bootstrap_steps:
                print('rnn [{}/{}] bootstrapping'.format(i+1, len(image_file_paths)))
                depth, relative_pose_matrix, _, _ = net.predict(image_file_path)
                net.update()
            else:
                print('rnn [{}/{}]'.format(i+1, len(image_file_paths)))
                depth, relative_pose_matrix, _, _ = net.predict(image_file_path)
                net.update()

                # if system.current_keyframe is not None:
                #     relative_pose_matrix = np.matmul(pose,
                #         np.linalg.inv(system.current_keyframe.pose_matrix))
                # else:
                #     relative_pose_matrix = pose

                image = Image.open(image_file_paths[i])
                image = np.array(image, dtype=np.uint8)
                assert image.shape[0] == depth.shape[0]
                assert image.shape[1] == depth.shape[1]
                keyframe = Keyframe(
                    white_noise_variance=system.white_noise_variance,
                    idx=count_kf, cam=cam, depth=depth, 
                    relative_pose_matrix=relative_pose_matrix,
                    preceding_keyframe=system.current_keyframe,
                    image=image,
                    name=os.path.basename(image_file_path)
                )
                if system.current_keyframe:
                    keyframe.update_preceding(system.current_keyframe)
                    keyframe.update_reference(system.current_keyframe)

                if system.should_add_keyframe(keyframe):
                    # insert keyframe
                    system.add_keyframe(keyframe)
                    count_kf += 1
                    if args.use_viewer:
                        if args.sparse_display:
                            system.refresh_display_content_v2(add_pose=True)
                        else:
                            system.refresh_display_content(add_pose=True)
                else:
                    # use this frame to refine the depth of current keyframe
                    system.current_keyframe.refine(keyframe)

                if seq == 2 and keyframe.name == break_frame_name:
                    print("[old_rnn]")
                    time.sleep(2)
                    depth_in_old, pose_relative_to_matched, _, _ = old_net.predict(image_file_path)
                    matched_keyframe = system.old_keyframes[-1]
                    pose_in_old = np.matmul(matched_keyframe.pose_matrix, pose_relative_to_matched)
                    global_relative_pose = np.matmul(np.linalg.inv(keyframe.pose_matrix), pose_in_old)
                    for k, kf in enumerate(system.keyframes):
                        new_pose = np.matmul(kf.pose_matrix, global_relative_pose)
                        new_pose = g2o.Isometry3d(new_pose)
                        system.keyframes[k].update_pose(new_pose)
                    system.refresh_display_content(deep=True, display_old_trajectory=True)
                    system.add_connection_edge([matched_keyframe.pose_matrix[0:3, 3], pose_in_old[0:3, 3]])


                if args.use_viewer:
                    viewer.update()
                if args.use_loopclosure:
                    loop_closing.add_keyframe(keyframe)
        if seq == 0:
            system.start_new_chunk()
            system.refresh_display_content(deep=True, display_old_trajectory=True)
            if args.use_viewer:
                viewer.update()
            old_net = net
            net = RNN_pred_colon_reproj.RNN_depth_pred(args.rnnmodel)

        if args.use_loopclosure:
            loop_closing.stop()

    
    time_elapsed = time.time() - start_time
    print('Time: {}'.format(time_elapsed))
    print('{:.2f}ms per frame,    {:.2f} frames    per second'.format(time_elapsed*1000/len(image_file_paths), len(image_file_paths)/time_elapsed))
    print('{:.2f}ms per keyframe, {:.2f} keyframes per second'.format(time_elapsed*1000/count_kf, count_kf/time_elapsed))

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-i',
        default='/home/ruibinma/visualize_missing_region/09_image')
    parser.add_argument('--intrinsic',
        default='/home/ruibinma/visualize_missing_region/09_txt/calibration.txt')
    parser.add_argument('--presort_poses', default=False, action='store_true')
    parser.add_argument('--sparse_display', default=False, action='store_true')
    parser.add_argument('--relVarTH', default=0.001, type=float)
    parser.add_argument('--absVarTH', default=0.001, type=float)
    parser.add_argument('--minRelativeBS', default=0.1, type=float)
    parser.add_argument('--use_viewer', default=False, action='store_true')
    parser.add_argument('--use_loopclosure', default=False, action='store_true')
    parser.add_argument('--local_window_size', default=7, type=int)
    parser.add_argument('--rnnmodel')
    args = parser.parse_args()
    main(args)