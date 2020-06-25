from __future__ import division
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

# def save_color_depth(depth, filename):
#     plt.imsave(filename, depth, cmap='plasma')

# def gray2rgb(im, cmap='gray'):
#     cmap = plt.get_cmap(cmap)
#     rgba_img = cmap(im.astype(np.float32))
#     rgb_img = np.delete(rgba_img, 3, 2)
#     return rgb_img

# def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
#     # convert to disparity
#     depth = 1./(depth + 1e-6)
#     if normalizer is not None:
#         depth = depth/normalizer
#     else:
#         depth = depth/(np.percentile(depth, pc) + 1e-6)
#     depth = np.clip(depth, 0, 1)
#     depth = gray2rgb(depth, cmap=cmap)
#     keep_H = int(depth.shape[0] * (1-crop_percent))
#     depth = depth[:keep_H]
#     depth = depth
#     return depth

def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = tf.shape(input=z)[0]
  N = 1
  #z = tf.clip_by_value(z, -np.pi, np.pi)
  #y = tf.clip_by_value(y, -np.pi, np.pi)
  #x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  
  return rotMat

# Checks if a matrix is a valid rotation matrix.
# def isRotationMatrix(R) :
#     Rt = np.transpose(R)
#     shouldBeIdentity = np.dot(Rt, R)
#     I = np.identity(3, dtype = R.dtype)
#     n = np.linalg.norm(I - shouldBeIdentity)
#     return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    #assert(isRotationMatrix(R))

  sy = tf.sqrt(R[:,0,0] * R[:,0,0] +  R[:,1,0] * R[:,1,0])
  eps = tf.constant(1e-6,dtype=tf.float32,shape=sy.get_shape())
  singular = tf.less(sy, eps)
  x = tf.expand_dims(tf.compat.v1.where(singular, tf.atan2(-R[:,1,2], R[:,1,1]), tf.atan2(R[:,2,1] , R[:,2,2])),1)
  y = tf.expand_dims(tf.compat.v1.where(singular, tf.atan2(-R[:,2,0], sy), tf.atan2(-R[:,2,0], sy)),1)
  z = tf.expand_dims(tf.compat.v1.where(singular, tf.constant(0,dtype=tf.float32,shape=sy.get_shape()), tf.atan2(R[:,1,0], R[:,0,0])),1)
  eulerangle = tf.concat([x,y,z],axis=1)
  # for i in range(B.eval()):
  #   sy = tf.sqrt(R[i,0,0] * R[i,0,0] +  R[i,1,0] * R[i,1,0])
  #   eps = tf.constant(1e-6,dtype=float32)
  #   singular = tf.less(sy, eps)

  #   if  not singular :
  #       x = tf.atan2(R[i,2,1] , R[i,2,2])
  #       y = tf.atan2(-R[i,2,0], sy)
  #       z = tf.atan2(R[i,1,0], R[i,0,0])
  #   else :
  #       x = tf.atan2(-R[i,1,2], R[i,1,1])
  #       y = tf.atan2(-R[i,2,0], sy)
  #       z = tf.constant(0,dtype=float32)
  #   if i==0:
  #     eulerangle = tf.expend_dims(tf.concat(x,y,z),0)
  #   else:
  #     eulerangle = tf.concat(eulerangle,tf.expend_dims(tf.concat(x,y,z),0),axis=0)

  return eulerangle


def axis_angle_to_rotation_matrix(axis, angle):


  B = tf.shape(input=angle)[0]
  zeros = tf.zeros([B, 1, 1])
  ones  = tf.ones([B, 1, 1])


  M1 = tf.concat( [zeros,-tf.expand_dims(tf.expand_dims(axis[:,2],-1),-1), tf.expand_dims(tf.expand_dims(axis[:,1],-1),-1)], axis=2)
  M2 = tf.concat( [zeros, zeros,-tf.expand_dims(tf.expand_dims(axis[:,0],-1),-1)], axis=2)
  M3 = tf.concat( [zeros, zeros,zeros], axis=2)

  M = tf.concat([M1, M2, M3], axis=1)

  cp_axis = M-tf.transpose(a=M,perm=[0,2,1])
  #import pdb;pdb.set_trace()
  # ANGLE_SIN = tf.concat([tf.sin(angle),tf.sin(angle),tf.sin(angle)],axis=2)
  # ANGLE_SIN = tf.concat([ANGLE_SIN, ANGLE_SIN, ANGLE_SIN], axis=1)

  # ANGLE_COS = tf.concat([tf.cos(angle),tf.cos(angle),tf.cos(angle)],axis=2)
  # ANGLE_COS = tf.concat([ANGLE_COS, ANGLE_COS, ANGLE_COS], axis=1)

  # ONES = tf.concat([ones,ones,ones],axis=2)
  # ONES = tf.concat([ONES,ONES,ONES],axis=1)

  rotMat = tf.eye(3,batch_shape=[B]) + tf.sin(angle)*cp_axis + (ones - tf.cos(angle))* tf.matmul(cp_axis,cp_axis)
  return rotMat


def pose_vec2mat(vec,format):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  
  batch_size, _ = vec.get_shape().as_list()
  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)

  if format=='eular':
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])

  elif format=='angleaxis':

    axis = tf.slice(vec, [0, 3], [-1, 3])
    angle = tf.expand_dims(tf.norm(tensor=axis,axis=1),-1)
    #import pdb;pdb.set_trace()
    # if( angle > 1.0e-6 ):
    axis /=angle
    angle = tf.expand_dims(angle,-1)
    rot_mat = axis_angle_to_rotation_matrix(axis,angle)

  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)
  return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.linalg.inv(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])

  z_u = tf.reshape(z_u, [batch, height, width, 1])
  
  return tf.transpose(a=pixel_coords, perm=[0, 2, 3, 1]),z_u

def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(a=tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), perm=[1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords

def projective_inverse_warp(img, depth, pose, intrinsics,format='eular'):
  """Inverse warp a source image to the target image plane based on projection.

  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of tx, ty, tz, rx, ry, rz
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  #import pdb;pdb.set_trace()
  batch, height, width, _ = img.get_shape().as_list()

  if format=='eular' or format=='angleaxis':
    pose = pose_vec2mat(pose,format)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords, src_depth = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  #import pdb;pdb.set_trace()
  rigid_flow = src_pixel_coords-tf.transpose(a=pixel_coords[:,0:2,:,:],perm=[0,2,3,1])

  output_img,wmask = bilinear_sampler(img, src_pixel_coords)

  return output_img, wmask, rigid_flow



def random_ROT_warp(img, depth, pose, intrinsics,format='eular'):
  """Inverse warp a source image to the target image plane based on projection.

  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of tx, ty, tz, rx, ry, rz
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  batch, height, width, _ = img.get_shape().as_list()
  # Convert pose vector to matrix
  #import pdb;pdb.set_trace()
  if format=='eular' or format=='angleaxis':
    pose = pose_vec2mat(pose,format)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords, src_depth = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  #out_img,out_depth = extract_image(img,src_pixel_coords,src_depth)

  out_img,_ = bilinear_sampler(img, src_pixel_coords)

  depth = tf.expand_dims(depth, -1)

  out_depth,_ = bilinear_sampler(depth, src_pixel_coords)

  return out_img,out_depth

def extract_image(img,src_pixel_coords,src_depth):

  batch, height, width, _ = img.get_shape().as_list()

  out_img = tf.zeros([batch,height,width,3])
  out_depth = tf.zeros([batch,height,width,1])

  return out_img,out_depth

def optflow_warp(img,flowx,flowy):

  #import pdb;pdb.set_trace()
  batch, height, width, _ = img.get_shape().as_list()
  pixel_coords = meshgrid(batch, height, width,is_homogeneous=False)

  pixel_coords = tf.transpose(a=pixel_coords, perm=[0,2,3,1])
  coords_x, coords_y = tf.split(pixel_coords, [1, 1], axis=3)

  x_n = coords_x+flowx
  y_n = coords_y+flowy

  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  src_pixel_coords = tf.transpose(a=pixel_coords, perm=[0,2,3,1])
  output_img,_ = bilinear_sampler(img, src_pixel_coords)
  return output_img

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        a=tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), perm=[1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.compat.v1.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(input=imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(input=imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    #import pdb;pdb.set_trace()
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])

    wmask = w00+w01+w10+w11

    return output,wmask





def consistent_depth_loss(src_depth,pred_src_depth, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        a=tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), perm=[1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.compat.v1.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = src_depth.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = src_depth.get_shape().as_list()[3]   

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(input=src_depth)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(input=src_depth)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    
    imgs_flat = tf.reshape(src_depth, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    #import pdb;pdb.set_trace()
    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    #wmask = w00+w01+w10+w11

    return tf.abs(pred_src_depth-output)




def solve_scale(static_points,moving_points):

  #import pdb;pdb.set_trace()
  y=0.0
  #for i in range(length(static_points)):
    #y = y+(static_points(:,i)-s*moving_points(:,i)).T*(static_points(:,i)-s*moving_points(:,i))


def depth_optflow(src_pixel_coords):
  
  
  batch, height, width, _ = src_pixel_coords.get_shape().as_list()
  pixel_coords = meshgrid(batch, height, width,is_homogeneous=False)


  
  pixel_coords = tf.transpose(a=pixel_coords, perm=[0,2,3,1])
  coords_x, coords_y = tf.split(pixel_coords, [1, 1], axis=3)


  coords_x_src, coords_y_src = tf.split(src_pixel_coords, [1, 1], axis=3)

  optflowx = coords_x_src-coords_x;
  optflowy = coords_y_src-coords_y;

  return optflowx,optflowy

UNKNOWN_FLOW_THRESH = 1e7
def flow_to_image(flow):
  """
  Convert flow into middlebury color code image
  :param flow: optical flow map
  :return: optical flow image in middlebury color
  """
  u = flow[:, :, 0]
  v = flow[:, :, 1]

  maxu = -999.
  maxv = -999.
  minu = 999.
  minv = 999.

  idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
  u[idxUnknow] = 0
  v[idxUnknow] = 0

  maxu = max(maxu, np.max(u))
  minu = min(minu, np.min(u))

  maxv = max(maxv, np.max(v))
  minv = min(minv, np.min(v))

  rad = np.sqrt(u ** 2 + v ** 2)
  maxrad = max(-1, np.max(rad))

  #print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

  u = u/(maxrad + np.finfo(float).eps)
  v = v/(maxrad + np.finfo(float).eps)

  img = compute_color(u, v)

  idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
  img[idx] = 0

  return np.uint8(img)


# def depth_plasma(depth):
#   cmap = plt.get_cmap('plasma')
#   rgb_depth = cmap(depth)
#   return rgb_depth


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel
