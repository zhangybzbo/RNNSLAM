import glob
import numpy as np
from components import Frame, Camera
import time
from PIL import Image
import argparse
import os
import shutil
from adjust_depth import windowed_averaging_vote
import time

class MySystem(object):

    def __init__(self, local_window_size=7):
        self.poses = []
        self.points = np.empty(shape=(0, 3))
        self.colors = np.empty(shape=(0, 3))
        self.image = None
        self.depth_image = None
        self.current_keyframe = None
        self.keyframes = []
        self.local_window_size = local_window_size
        self.total_uncertainty = 0

        self.allpoints = []
        self.allcolors = []
    
    def save_refined_depth(self, kf_tosave, refined_depth_dir):
        refined_depth_path = os.path.join(refined_depth_dir, kf_tosave.name)
        kf_tosave.depth.astype(np.float32).tofile(refined_depth_path)
        print('updated {}'.format(refined_depth_path))
    
    def add_keyframe(self, keyframe, refined_depth_dir=None):

        self.keyframes.append(keyframe)
        if len(self.keyframes) > self.local_window_size:
            # save marginalized frame
            if refined_depth_dir is not None:
                kf_tosave = self.keyframes[0]
                self.save_refined_depth(kf_tosave, refined_depth_dir)
            # end of saving
            del self.keyframes[0]
        else:
            print('')

        pure_computing_start_time = time.time()
        if len(self.keyframes) >= 2 and self.local_window_size > 1:
            windowed_averaging_vote(self.keyframes)
        pure_computing_end_time = time.time()
        self.current_keyframe = keyframe
        return pure_computing_end_time - pure_computing_start_time

    def refresh_display_content_window(self, add_pose=False):
        if add_pose:
            self.poses.append(self.current_keyframe.pose_matrix)

        points = []
        colors = []
        for keyframe in self.keyframes:
            p, c = keyframe.get_vis_points()
            points.append(p)
            colors.append(c)
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        points = points[:, :3]


        self.points = points
        self.colors = colors
        if self.current_keyframe.image is not None:
            self.image = np.flip(self.current_keyframe.image, axis=0)
        if self.current_keyframe.depth_image is not None:
            self.depth_image = np.flip(self.current_keyframe.depth_image, axis=0)
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

def write_all_keyframes(system, refined_depth_dir):
    for kf_tosave in system.keyframes:
        system.save_refined_depth(kf_tosave, args.refined_depth_dir)


def load_calibration(calibration_file):
    with open(calibration_file, 'r') as file:
        line = file.readline().split(' ')
        fx = float(line[1])
        fy = float(line[2])
        cx = float(line[3])
        cy = float(line[4])
        line = file.readline().split(' ')
        width = int(line[0])
        height = int(line[1])

    print("Camera parameters: [{} {}] {} {} {} {}".format(
        width, height, fx, fy, cx, cy))

    cam = Camera(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height,
        frustum_near=0.01, frustum_far=1000.)

    return cam


def main(args):
    if args.presort_poses:
        presort(args.cameras_file_path)
    if args.refined_depth_dir is not None:
        if os.path.exists(args.refined_depth_dir):
            shutil.rmtree(args.refined_depth_dir)
        shutil.copytree(args.depth_dir, args.refined_depth_dir)

    # read camera calibration
    cam = load_calibration(args.intrinsic)

    # cameras and depths paths
    cameras_file_path = args.cameras_file_path
    depth_file_paths = sorted(glob.glob(os.path.join(args.depth_dir, '*.depth.bin')))
    if not depth_file_paths:
        depth_file_paths = sorted(glob.glob(os.path.join(args.depth_dir, '*.bin')))
    
    # image file paths
    image_names = sorted(os.listdir(args.image_dir))
    image_file_paths = [os.path.join(args.image_dir, name) for name in image_names]

    n_images = len(image_file_paths)
    assert len(depth_file_paths) == n_images

    # depth images, these are user provided depth images to show in UI. By default None.
    depth_image_file_paths = None
    if args.depth_image_dir:
        depth_image_names = sorted(os.listdir(args.depth_image_dir))
        depth_image_file_paths = [os.path.join(args.depth_image_dir, name) for name in depth_image_names]

    # initialize system and map viewer
    system = MySystem(local_window_size=args.local_window_size)
    if args.use_viewer:
        print('use_viewer = True, import MapViewer')
        from viewer import MapViewer
        viewer = MapViewer(system, w=cam.width, h=cam.height)

    current_frame_id = -1
    count_kf = 0
    start_time = time.time()
    pure_computing_time = 0.0
    file = open(cameras_file_path, 'r')
    for i in range(n_images):
        print('[{:3d}/{:3d}]  '.format(i+1,n_images), end='')
        # 1) ------- Load Relative Camera pose
        if current_frame_id < i:
            m = file.readline()
            m = m.split(' ')
            if len(m) < 5:
                print("End of Sequence!")
                if args.refined_depth_dir is not None:
                    write_all_keyframes(system, args.refined_depth_dir)
                break
            if len(m) > 12:
                current_frame_id = int(m[12])

        if current_frame_id > i:
            print('')
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

        # 2) ------- Load Depth Map
        depth_file_path = depth_file_paths[i]
        _, depth = load_depth(
            depth_file_path,
            cam.height, cam.width,
            fx=cam.fx, fy=cam.fy,
            cx=cam.cx, cy=cam.cy
        )
        
        # 3) ------- Load Image
        if args.use_viewer and image_file_paths is not None:
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

        # 4) ------- Load Depth Image
        if args.use_viewer and depth_image_file_paths is not None:
            depth_image = Image.open(depth_image_file_paths[i])
            try:
                depth_image = depth_image.resize((depth.shape[1],depth.shape[0]))
            except:
                depth_image_np = np.array(depth_image)
                depth_image = Image.fromarray(depth_image_np)
                depth_image = depth_image.resize((depth.shape[1],depth.shape[0]))
            depth_image = np.array(depth_image)
            depth_image = depth_image[:,:,:3]
        else:
            depth_image = None

        # 5) ------- Instantiate keyframe
        keyframe = Frame(
            idx=i, cam=cam, depth=depth, 
            relative_pose_matrix=relative_pose_matrix,
            preceding_keyframe=system.current_keyframe,
            image=image, depth_image=depth_image,
            name=os.path.basename(depth_file_path)
        )
        # 6) ------- Add Keyframe, Do Windowed Averaging
        pure_computing_time += system.add_keyframe(keyframe, refined_depth_dir=args.refined_depth_dir)
        count_kf += 1

        if args.use_viewer:
            system.refresh_display_content_window(add_pose=True)
            viewer.update()
        
        if i==len(depth_file_paths)-1 and args.refined_depth_dir is not None:
            write_all_keyframes(system, args.refined_depth_dir)
            

    time_elapsed = time.time() - start_time
    print('Total Time: {}'.format(time_elapsed))
    print('{:.2f}ms per frame,    {:.2f} frames    per second'.format(time_elapsed*1000/len(depth_file_paths), len(depth_file_paths)/time_elapsed))
    print('{:.2f}ms per keyframe, {:.2f} keyframes per second'.format(time_elapsed*1000/count_kf, count_kf/time_elapsed))
    print('Pure Computing Time: {}'.format(pure_computing_time))
    print('{:.2f}ms per frame,    {:.2f} frames    per second'.format(pure_computing_time*1000/len(depth_file_paths), len(depth_file_paths)/pure_computing_time))
    print('{:.2f}ms per keyframe, {:.2f} keyframes per second'.format(pure_computing_time*1000/count_kf, count_kf/pure_computing_time))
    file.close()
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cameras_file_path')
    parser.add_argument('--depth_dir')
    parser.add_argument('--image_dir')
    parser.add_argument('--refined_depth_dir')
    parser.add_argument('--intrinsic')
    parser.add_argument('--presort_poses', default=False, action='store_true')
    parser.add_argument('--use_viewer', default=False, action='store_true')
    parser.add_argument('--local_window_size', default=7, type=int)
    parser.add_argument('--depth_image_dir', default=None)
    args = parser.parse_args()
    main(args)
