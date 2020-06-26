import numpy as np
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue
from PIL import Image

def random_image(w, h):   # for debug
    return (np.ones((h, w, 3), 'uint8') * 
        np.random.randint(256, size=3, dtype='uint8'))

class MapViewer(object):
    def __init__(self, system, w, h):
        self.system = system
        self.w = w
        self.h = h


        self.q_camera = Queue()
        self.q_trajectory = Queue()
        self.q_point = Queue()
        self.q_color = Queue()
        self.q_pose = Queue()
        self.q_image = Queue()
        self.q_old_trajectory = Queue()
        self.q_old_camera = Queue()
        self.q_connection_edge = Queue()
        # self.q_depth = Queue()

        self.view_thread = Process(target=self.view)
        self.view_thread.start()

    def update(self, refresh=False):
        if refresh:
            print('************************** refresh')
            
        else:
            cameras = []
            for pose in self.system.poses:
                cameras.append(pose)
            
            if len(cameras) > 0:
                self.q_camera.put(cameras)
            
            old_cameras = []
            for pose in self.system.old_poses:
                cameras.append(pose)
            
            if len(old_cameras) > 0:
                self.q_old_camera.put(old_cameras)

            trajectory = []
            for point in self.system.trajectory:
                trajectory.append(point)
            
            if len(trajectory) > 1:
                self.q_trajectory.put(trajectory)

            old_trajectory = []
            for point in self.system.old_trajectory:
                old_trajectory.append(point)
            
            if len(old_trajectory) > 1:
                self.q_old_trajectory.put(old_trajectory)
            
            connection_edge = []
            for point in self.system.connection_edge:
                connection_edge.append(point)
            if len(connection_edge) > 1:
                self.q_connection_edge.put(connection_edge)
            
            points = self.system.points
            if points is not None:
                self.q_point.put(points)
            
            colors = self.system.colors
            if colors is not None:
                self.q_color.put(colors)

            image = self.system.image
            if image is not None:
                self.q_image.put(image)

            # depth = self.system.depth
            # if depth is not None:
            #     self.q_depth.put(depth)
            if self.system.current_keyframe is not None:
                self.q_pose.put(self.system.current_keyframe.pose_matrix)
    
    def view(self):
        pangolin.CreateWindowAndBind('Viewer', 1024, 768)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        viewpoint_x = 0
        viewpoint_y = -5   # -10
        viewpoint_z = -10   # -0.1
        viewpoint_f = 2000
        camera_width = 0.02

        proj = pangolin.ProjectionMatrix(
            1024, 768, viewpoint_f, viewpoint_f, 512, 389, 0.1, 5000)
        look_view = pangolin.ModelViewLookAt(
            viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
        
        scam = pangolin.OpenGlRenderState(proj, look_view)

        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -1024./768.)
        dcam.SetHandler(pangolin.Handler3D(scam))

        dimg = pangolin.Display('image')
        dimg.SetBounds(0.0, self.h / 768., 0.0, self.w / 1024., float(self.w)/self.h)
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        texture = pangolin.GlTexture(self.w, self.h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        # ddepth = pangolin.Display('depth')
        # ddepth.SetBounds(self.h / 768., self.h / 768. * 2.0, 0.0, self.w / 1024., float(self.w)/float(self.h))
        # ddepth.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        # texture_depth = pangolin.GlTexture(self.w, self.h, gl.GL_LUMINANCE , False, 0, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE)

        cameras = []
        old_cameras = []
        trajectory = []
        old_trajectory = []
        connection_edge = []
        pose = pangolin.OpenGlMatrix()
        points = np.empty(shape=(0, 3))
        colors = np.empty(shape=(0, 3))
        # image = random_image(self.w, self.h)
        image = 255 * np.ones((self.h, self.w , 3), 'uint8')
        # depth = 255 * np.ones((self.h, self.w), 'uint8')

        gl.glPointSize(3)
        gl.glLineWidth(2)
        while not pangolin.ShouldQuit():                
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.75, 0.75, 0.75, 1.0)
            if not self.q_pose.empty():
                pose.m = self.q_pose.get()
            # scam.Follow(pose, True)

            dcam.Activate(scam)
            gl.glLineWidth(2)
            if not self.q_camera.empty():
                cameras = self.q_camera.get()
            gl.glColor3f(0.0, 0.0, 1.0)
            if len(cameras) > 0:
                pangolin.DrawCameras(cameras, camera_width)
            
            if not self.q_old_camera.empty():
                old_cameras = self.q_old_camera.get()
            if len(old_cameras) > 0:
                pangolin.DrawCameras(old_cameras, camera_width)
            gl.glLineWidth(4)
            if not self.q_trajectory.empty():
                trajectory = self.q_trajectory.get()
            if len(trajectory) > 1:
                gl.glColor3f(0.0, 0.0, 0.0)
                pangolin.DrawLine(trajectory)
            
            if not self.q_connection_edge.empty():
                connection_edge = self.q_connection_edge.get()
            if len(connection_edge) > 1:
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawLine(connection_edge)

            if not self.q_old_trajectory.empty():
                old_trajectory = self.q_old_trajectory.get()
            if len(old_trajectory) > 1:
                gl.glColor3f(1.0, 1.0, 0.0)
                pangolin.DrawLine(old_trajectory)

            if not self.q_point.empty():
                points = self.q_point.get()
                
            if not self.q_color.empty():
                colors = self.q_color.get()
            if len(points) > 0:
                pangolin.DrawPoints(points, colors)


            if not self.q_image.empty():
                image = self.q_image.get()
                # print(image.shape, image.dtype)

            # texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            # dimg.Activate()
            # gl.glColor3f(1.0, 1.0, 1.0)
            # texture.RenderToViewport()

            # if not self.q_depth.empty():
            #     depth = self.q_depth.get()
            #     print(depth.shape, depth.dtype)

            # texture_depth.Upload(depth, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE)
            # ddepth.Activate()
            # gl.glColor3f(1.0, 1.0, 1.0)
            # texture_depth.RenderToViewport()

            
            pangolin.FinishFrame()

            

