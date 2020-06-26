import glob
import numpy as np
from my_viewer import MapViewer
from my_components import Frame, Camera, Keyframe
import time
from PIL import Image
import cv2
import argparse
import os
import struct
import shutil
from adjust_depth import windowed_dense_depth_adjustment, eliminate_conv_boundary_effect
from adjust_depth import windowed_averaging_ofraw
from adjust_depth import windowed_averaging_multithread

class MySystem(object):

    def __init__(self, white_noise_variance, local_window_size=7):
        self.poses = []
        self.points = np.empty(shape=(0, 3))
        self.colors = np.empty(shape=(0, 3))
        self.sparse_points = np.empty(shape=(0, 3))
        self.sparse_colors = np.empty(shape=(0, 3))
        self.image = None
        self.current_keyframe = None
        self.keyframes = []
        self.local_window_size = local_window_size
        self.half_window_size = (local_window_size + 1) // 2
        self.white_noise_variance = white_noise_variance
        self.total_uncertainty = 0

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
    
    def add_keyframe(self, frame, refined_depth_dir=None):
        keyframe = Keyframe(
            white_noise_variance=self.white_noise_variance,
            idx=frame.idx, cam=frame.cam, depth=frame.depth, 
            relative_pose_matrix=frame.relative_pose_matrix,
            preceding_keyframe=frame.preceding_keyframe,
            image=frame.image,
            sparse_points_depths=frame.sparse_points_depths,
            name=frame.name
        )
        self.keyframes.append(keyframe)
        if len(self.keyframes) > self.local_window_size:
            # save marginalized frame
            if refined_depth_dir is not None:
                kf_tosave = self.keyframes[0]
                self.save_refined_depth(kf_tosave, refined_depth_dir)
            # end of saving
            del self.keyframes[0]

        # keyframe.compute_uncertainty_map()
        # keyframe.fuse_with_preceding_keyframe()
        # keyframe.simple_fuse_with_preceding_keyframe(0.9)
        # if len(self.keyframes) >= 2:
        #     windowed_dense_depth_adjustment(self.keyframes, len(self.keyframes)-1)

        if len(self.keyframes) >= self.half_window_size:
            windowed_averaging_ofraw(
                self.keyframes, len(self.keyframes) - self.half_window_size)

        self.current_keyframe = keyframe
        print('Keyframe inserted')

    def refresh_display_content(self, add_pose=False, concat=False):
        if add_pose:
            self.poses.append(self.current_keyframe.pose_matrix)

        id = -1
        id = 0
        # if len(self.keyframes) >= 7:
        #     id = 3
        points, colors = self.keyframes[id].get_vis_points(use_sparse=False)
        points = points[:, :3]
        if concat:
            self.points = np.concatenate([self.points, points], axis=0)
            self.colors = np.concatenate([self.colors, colors], axis=0)
        else:
            self.points = points
            self.colors = colors
        self.image = np.rot90(np.rot90(self.current_keyframe.image))
        depth = self.current_keyframe.depth
        max_depth = np.max(depth)
        depth = depth / max_depth * 255

        self.depth = np.rot90(np.rot90(depth.astype(np.uint8)))

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

    def avg_remaining_keyframes(self):
        for i in range(len(self.keyframes)-self.half_window_size+1, len(self.keyframes)):
            print('Averaging remaining: {}'.format(self.keyframes[i].idx))
            windowed_averaging_ofraw(self.keyframes, i)

    def write_all_keyframes(self, refined_depth_dir):
        for kf_tosave in self.keyframes:
            self.save_refined_depth(kf_tosave, args.refined_depth_dir)


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


def main(args):
    if args.presort_poses:
        presort(args.cameras_file_path)
    if args.refined_depth_dir is not None:
        if os.path.exists(args.refined_depth_dir):
            shutil.rmtree(args.refined_depth_dir)
        shutil.copytree(args.depth_dir, args.refined_depth_dir)

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
    # load cameras and depths from 09
    cameras_file_path = args.cameras_file_path
    depth_file_paths = sorted(glob.glob(os.path.join(args.depth_dir, '*.depth.bin')))
    
    image_names = sorted(os.listdir(args.image_dir))
    image_file_paths = [os.path.join(args.image_dir, name) for name in image_names]

    # initialize system and map viewer
    system = MySystem(white_noise_variance=50., local_window_size=7)
    viewer = MapViewer(system, w=width, h=height)
    current_frame_id = -1
    with open(cameras_file_path, 'r') as file:
        for i, depth_file_path in enumerate(depth_file_paths):
            print('[{}/{}]  '.format(i+1,len(depth_file_paths)))
            ##########################
            # predict camera path & depth

            if current_frame_id < i:
                m = file.readline()
                m = m.split(' ')
                if len(m) < 5:
                    print("End of Sequence!")
                    system.avg_remaining_keyframes()
                    if args.refined_depth_dir is not None:
                        system.write_all_keyframes(system, args.refined_depth_dir)
                    break
                if len(m) > 12:
                    current_frame_id = int(m[12])

            if current_frame_id > i:
                print('Waiting for {}'.format(current_frame_id))
                continue

            matrix = np.identity(4)
            matrix[0, :] = m[0:4]
            matrix[1, :] = m[4:8]
            matrix[2, :] = m[8:12]
            if system.current_keyframe is not None:
                relative_pose_matrix = np.matmul(matrix,
                    np.linalg.inv(system.current_keyframe.pose_matrix))
            else:
                relative_pose_matrix = matrix

            _, depth = load_depth(depth_file_path, height, width,
                fx=fx, fy=fy,
                cx=cx, cy=cy)
            
            if image_file_paths is not None:
                image = Image.open(image_file_paths[i])
                try:
                    image = image.resize((depth.shape[1],depth.shape[0]))
                except:
                    image_np = np.array(image)
                    image = Image.fromarray(image_np)
                    image = image.resize((depth.shape[1],depth.shape[0]))
                image = np.array(image)
            else:
                image = None

            sparse_points_depths = None
            if args.sparse_depth_dir is not None:
                sparse_depth_path = os.path.join(args.sparse_depth_dir, '{:06d}.bin'.format(i))
                sparse_points_depths = load_sparse_points_depths(sparse_depth_path, args)


            ##########################
            # end of prediction

            # depth bilateral filtering
            # depth = cv2.bilateralFilter(depth, 9, 50, 300)
            # depth *= regular_depth_map

            frame = Frame(i, cam, depth, relative_pose_matrix,
                system.current_keyframe, image=image, sparse_points_depths=sparse_points_depths,
                name=os.path.basename(depth_file_path))
            if system.should_add_keyframe(frame):
                # insert keyframe
                system.add_keyframe(frame, refined_depth_dir=args.refined_depth_dir)
                if args.sparse_display:
                    system.refresh_display_content_v2(add_pose=True)
                else:
                    system.refresh_display_content_window(add_pose=True)
            else:
                # use this frame to refine the depth of current keyframe
                system.current_keyframe.refine(frame)
            
            if i==len(depth_file_paths)-1 and args.refined_depth_dir is not None:
                system.avg_remaining_keyframes()
                write_all_keyframes(system, args.refined_depth_dir)
            
            viewer.update()

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cameras_file_path', '-c',
        default='/home/ruibinma/visualize_missing_region/09_txt/09.txt')
    parser.add_argument('--depth_dir', '-d',
        default='/home/ruibinma/visualize_missing_region/09')
    parser.add_argument('--sparse_depth_dir', default=None)
    parser.add_argument('--image_dir', '-i',
        default='/home/ruibinma/visualize_missing_region/09_image')
    parser.add_argument('--intrinsic',
        default='/home/ruibinma/visualize_missing_region/09_txt/calibration.txt')
    parser.add_argument('--presort_poses', default=False, action='store_true')
    parser.add_argument('--sparse_display', default=False, action='store_true')
    parser.add_argument('--relVarTH', default=0.001, type=float)
    parser.add_argument('--absVarTH', default=0.001, type=float)
    parser.add_argument('--minRelativeBS', default=0.1, type=float)
    parser.add_argument('--refined_depth_dir', default=None)
    args = parser.parse_args()
    main(args)