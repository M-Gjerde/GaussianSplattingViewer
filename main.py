import math
import random
import time

import cv2
import glfw
import OpenGL.GL as gl
import glm
import skimage
from PIL import Image
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
import csv
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import torch
_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)
def preprocess_image(img_np):
    # Convert to tensor
    img_tensor = torch.from_numpy(img_np).float()

    # Add batch dimension if not present
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Normalize to [-1, 1]
    img_tensor = (img_tensor / 255.0) * 2 - 1

    # Permute dimensions if required
    if img_tensor.shape[1] > 3:  # assuming channel-last format for numpy array
        img_tensor = img_tensor.permute(0, 3, 1, 2)

    return img_tensor

# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

g_camera = None # middlebury
#g_camera = util.Camera(1280, 1920)
pose_index = 0
use_file = False
manual_save = False
switch_lr_pose = False
theta = 0
phi = 0
radius = 3
animate = False
debug_vector = False
new_camera_pos = False

BACKEND_OGL = 0
BACKEND_CUDA = 1
g_renderer_list = [
    None,  # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.3
g_auto_sort = True
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 6


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def impl_glfw_init():
    window_name = "NeUVF editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def cursor_pos_callback(window, xpos, ypos):
    g_camera.process_mouse(xpos, ypos)


up_vector = glm.vec3(0, -1, 0)  # Assuming 'up' is in the -y direction
front_vector = glm.vec3(0, 0, 1)
#default_forward = glm.vec3(0.0, 0, 1)


def load_camera_positions(camera_pose, bounding_box= None, center = glm.vec3(0, 0, 0), camera_bb = (0, 1, 0, 1, 0, 1)):
    global new_camera_pos, up_vector, front_vector, debug_vector
    qw, qx, qy, qz = float(camera_pose[1]), float(camera_pose[2]), float(camera_pose[3]), float(camera_pose[4])
    x, y, z = float(camera_pose[5]), float(camera_pose[6]), float(camera_pose[7])
    position = glm.vec3(x, y, z)
    baseline = -0.193001 * 5
    id = int(camera_pose[0])
    """
        if bounding_box is not None:
            # Convert bounding box to glm.vec3 for easier comparison
            min_bound = glm.vec3(*bounding_box[0])
            max_bound = glm.vec3(*bounding_box[1])
    
            # Check if the position is inside the bounding box
            is_inside = all([
                min_bound.x <= position.x <= max_bound.x,
                min_bound.y <= position.y <= max_bound.y,
                min_bound.z <= position.z <= max_bound.z,
            ])
    
            # If inside or too close, adjust or discard
            if is_inside:
                # This is a simple strategy: move the position further away along the vector from center to position
                # You might want to customize this logic based on your specific needs
                direction = glm.normalize(position - center)  # Direction from center to position
                adjustment_distance = 5.0  # This is an arbitrary distance; adjust as needed
                new_position = position + direction * adjustment_distance
                if id != last_image_id:
                    print(f"Adjusting position from {position} to {new_position}")
                position = new_position
            else:
                if id != last_image_id:
                    print(f"Camera pose is outputside of the scene bounding box")
    """
    last_image_id = id

    # Use a default forward vector, assuming the camera looks towards the negative Z-axis in its local spac
    rot = qvec2rotmat((qw, qx, qy, qz)).T
    T = np.array([x, y, z])
    world_view_trans = getWorld2View2(rot, T).transpose()

    #f = glm.normalize(rotation * front_vector)
    f_glm = glm.normalize(center - position)

    position.y = -position.y
    position.z = -position.z

    f = glm.normalize(center - position)
    s = glm.normalize(glm.cross(f, up_vector))
    u = glm.cross(s, f)

    viewMat = glm.mat4(1.0)

    viewMat[0][0] = s.x
    viewMat[1][0] = s.y
    viewMat[2][0] = s.z
    viewMat[0][1] = u.x
    viewMat[1][1] = u.y
    viewMat[2][1] = u.z
    viewMat[0][2] = -f.x
    viewMat[1][2] = -f.y
    viewMat[2][2] = -f.z
    viewMat[3][0] = -glm.dot(s, position)
    viewMat[3][1] = -glm.dot(u, position)
    viewMat[3][2] = glm.dot(f, position)



    if debug_vector:
        print("Front vector: ", front_vector)
        print("Position: ", position)
        print("up vector: ", up_vector)
        debug_vector = False

    # Right vector:
    right = glm.normalize(glm.cross(front_vector, up_vector))

    right_pos = position + (right * baseline)
    res = glm.distance(position, right_pos)

    T = glm.mat4(1.0)
    T[3, 0] = baseline
    T[3, 1] = 0
    T[3, 2] = 0
    viewMatRight = T * viewMat

    T = glm.mat4(1.0)
    T[3, 0] = -baseline
    T[3, 1] = 0
    T[3, 2] = 0
    viewMatLeft = T * viewMat

    #rot = qvec2rotmat((qw, qx, qy, qz))
    #trans = np.array([x, y, z])
#
    #Rt = np.zeros((4, 4), dtype=np.float32)
    #Rt[0:3, 0:3] = rot
    #Rt[0:3, 3] = trans
    #Rt[3, 3] = 1
#
    #world_to_view = Rt
    #view_to_world = np.linalg.inv(world_to_view)
    pose_left = {
        "camera_front": front_vector,
        "camera_up": up_vector,
        "camera_position": position,
        "camera_view": viewMatLeft
    }

    pose_center = {
        "camera_front": front_vector,
        "camera_up": up_vector,
        "camera_position": position,
        "camera_view": viewMat
    }

    pose_right = {
        "camera_front": front_vector,
        "camera_up": up_vector,
        "camera_position": right_pos,
        "camera_view": viewMatRight
    }
    return pose_left, pose_center, pose_right


def generate_sphere_positions(radius_l, theta_l, phi_l):
    positions = []
    theta_l = glm.radians(theta_l)
    phi_l = glm.radians(phi_l)
    x = radius_l * np.sin(theta_l) * np.cos(phi_l)
    y = radius_l * np.sin(theta_l) * np.sin(phi_l)
    z = radius_l * np.cos(theta_l)

    position = glm.vec3(x, y, z)
    front_vector = -glm.normalize(position)  # Normalize and invert to face origin
    up_vector = glm.vec3(0, -1, 0)  # Assuming 'up' is in the y-direction

    # Right vector:
    right = glm.normalize(glm.cross(front_vector, up_vector))
    baseline = 0.193001

    right_pos = position + (right * baseline)
    res = glm.distance(position, right_pos)

    pose = {
        "camera_front": front_vector,
        "camera_up": up_vector,
        "camera_position": position,
        "camera_view": glm.mat4(1.0),
    }

    poseRight = {
        "camera_front": front_vector,
        "camera_up": up_vector,
        "camera_position": right_pos,
        "camera_view": glm.mat4(1.0),
    }

    return pose, poseRight


def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)
    g_camera.is_middlemouse_pressed = (button == glfw.MOUSE_BUTTON_MIDDLE and pressed)

    if g_camera.is_middlemouse_pressed:
        # Specify the path to your CSV file
        csv_file_path = 'camera_data.csv'

        # Convert glm.vec3 objects to lists of their components
        camera_front = [g_camera.camera_front.x, g_camera.camera_front.y, g_camera.camera_front.z]
        camera_up = [g_camera.camera_up.x, g_camera.camera_up.y, g_camera.camera_up.z]
        camera_position = [g_camera.camera_position.x, g_camera.camera_position.y, g_camera.camera_position.z]

        # Data to append, flattened into a single list
        data_to_append = camera_front + camera_up + camera_position

        # Open the CSV file in append mode ('a') and use a csv.writer to write
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the data as a new row in the CSV file
            writer.writerow(data_to_append)


def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)


def key_callback(window, key, scancode, action, mods):
    global pose_index, use_file, manual_save, switch_lr_pose, theta, phi, radius, animate, debug_vector, new_camera_pos
    speed = 5
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)
        elif key == glfw.KEY_N:
            pose_index += 1
        elif key == glfw.KEY_M:
            pose_index -= 1
        elif key == glfw.KEY_I:
            switch_lr_pose = not switch_lr_pose
        elif key == glfw.KEY_X:
            manual_save = True
        elif key == glfw.KEY_P:
            use_file = not use_file
        elif key == glfw.KEY_B:
            debug_vector = not debug_vector
        elif key == glfw.KEY_UP or key == glfw.KEY_W:
            g_camera.camera_position += g_camera.camera_front
            theta -= 1.0 * speed
        elif key == glfw.KEY_DOWN or key == glfw.KEY_S:
            theta += 1.0 * speed
            g_camera.camera_position -= g_camera.camera_front
        elif key == glfw.KEY_LEFT or key == glfw.KEY_A:
            phi += 1.0 * speed
            g_camera.camera_position -= glm.normalize(glm.cross(g_camera.camera_front, g_camera.camera_up))
        elif key == glfw.KEY_RIGHT or key == glfw.KEY_D:
            phi -= 1.0 * speed
            g_camera.camera_position += glm.normalize(glm.cross(g_camera.camera_front, g_camera.camera_up))
        elif key == glfw.KEY_0:
            radius += (1.0 / speed)
            g_camera.camera_position -= glm.normalize(glm.cross(g_camera.camera_front, g_camera.camera_up))
        elif key == glfw.KEY_1:
            radius -= (1.0 / speed)
            g_camera.camera_position += glm.normalize(glm.cross(g_camera.camera_front, g_camera.camera_up))
        elif key == glfw.KEY_3:
            animate = not animate
            g_camera.camera_position -= glm.normalize(glm.cross(g_camera.camera_front, g_camera.camera_up))
        elif key == glfw.KEY_4:
            new_camera_pos = not new_camera_pos
            g_camera.camera_position += glm.normalize(glm.cross(g_camera.camera_front, g_camera.camera_up))
    print(f"theta: {theta}, phi: {phi}, radius: {radius}")


def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False


def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False


def update_activated_renderer_state(gaus: util_gau.GaussianData):
    global use_file
    camera_front = glm.vec3(float(0), float(0), float(0))
    camera_up = glm.vec3(float(0), float(0), float(1))
    camera_position = glm.vec3(float(-3), float(0), float(1))

    # Convert row data to a dictionary and append to the list
    pose = {
        "camera_front": camera_front,
        "camera_up": camera_up,
        "camera_position": camera_position,
        "camera_view": glm.mat4(1.0),
    }

    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera, use_file, pose)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera, use_file, pose)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)


def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)


def read_camera_poses_from_csv(csv_file_path):
    # Initialize an empty list to store the dictionaries
    poses_list = []

    # Open the CSV file for reading
    with open(csv_file_path, 'r') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)

        # Optional: If your CSV file includes headers, you can skip the first row
        # next(reader, None)  # Uncomment this line if your CSV has a header row

        # Loop through each row in the CSV file
        for row in reader:
            # Ensure the row has at least 9 columns for camera_front, camera_up, and camera_position
            if len(row) >= 9:
                try:
                    # Parse the vector components from the row
                    camera_front = glm.vec3(float(row[0]), float(row[1]), float(row[2]))
                    camera_up = glm.vec3(float(row[3]), float(row[4]), float(row[5]))
                    camera_position = glm.vec3(float(row[6]), float(row[7]), float(row[8]))

                    # Convert row data to a dictionary and append to the list
                    pose = {
                        "camera_front": camera_front,
                        "camera_up": camera_up,
                        "camera_position": camera_position
                    }
                    poses_list.append(pose)
                except ValueError as e:
                    print(f"Error converting row to float: {row}. Error: {e}")
                    continue  # Skip rows with conversion errors

    return poses_list


last_update_time = 0


def update_sphere_positions(radius, theta, phi):
    global last_update_time

    # Get the current time
    current_time = time.time()

    # Check if at least 0.03 seconds have passed since the last update
    if current_time - last_update_time >= 0.03:
        # Update the last update time
        last_update_time = current_time

        # Perform the update
        theta += 5
        phi = np.floor(theta / 360) * 5

        if phi > 35:
            phi = 0
    else:
        # If not enough time has passed, do not update theta and phi
        # Just return the current values
        pass

    return radius, theta, phi


def main(trained_model = None, colmap_poses = None):
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables, pose_index, manual_save, switch_lr_pose, theta, phi, radius, use_file

    width = 1920
    height = 1280
    # settings
    camera_poses = []
    if colmap_poses is not None:
        use_file = True
        imagesFilePath = os.path.join(colmap_poses, "images.txt")
        file = open(imagesFilePath, "r")
        line_no = 0
        for line in file.readlines():
            if line.startswith("#"):
                continue
            if line_no % 2 == 1:
                line_no += 1
                continue

            elements = line.split()
            # Extract the needed information
            # Assuming the first 8 values are IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ
            image_id, qw, qx, qy, qz, x, y, z, camera_id, fileName = elements
            camera_poses.append(elements)

            line_no += 1
        print(f"Found correct number of camera_poses: {len(camera_poses) == 100}")

        camerasFilePath = os.path.join(colmap_poses, "cameras.txt")
        cameraFile = open(camerasFilePath, "r")
        for line in cameraFile.readlines():
            if line.startswith("#"):
                continue
            elements = line.split()
            id = int(elements[0])
            width = int(elements[2])
            height = int(elements[3])
            fx, fy, cx, cy = float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])

    height = 522
    width = 1160
    g_camera = util.Camera(height, width)

    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)

    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)

    #    from renderer_cuda import CUDARenderer
    #    g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]

    g_renderer_idx = BACKEND_OGL
    g_renderer = g_renderer_list[g_renderer_idx]


    #width, height = glfw.get_framebuffer_size(window)

    # Step 1: Create an FBO and a texture attachment
    fbo_center = gl.glGenFramebuffers(1)
    fbo_center_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbo_center_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_center)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbo_center_texture, 0)
    fbo_right = gl.glGenFramebuffers(1)
    fbo_right_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbo_right_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_right)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbo_right_texture, 0)
    # Step 1: Create an FBO and a texture attachment
    fbo_disparity = gl.glGenFramebuffers(1)
    fbo_disparity_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbo_disparity_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, width, height, 0, gl.GL_RED, gl.GL_FLOAT, None)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_disparity)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbo_disparity_texture, 0)

    fbo_left = gl.glGenFramebuffers(1)
    fbo_left_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbo_left_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_left)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbo_left_texture, 0)

    # Check FBO status
    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
        print("Error: Framebuffer is not complete.")
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    outputFolder = "out"

    scene_folder = "0000"
    if trained_model is not None:
        scene_folder = os.path.split(trained_model)[-1]

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    if not os.path.exists(f"{outputFolder}/{scene_folder}"):
        os.mkdir(f"{outputFolder}/{scene_folder}")
    if not os.path.exists(f"{outputFolder}/{scene_folder}/right"):
        os.mkdir(f"{outputFolder}/{scene_folder}/right")
    if not os.path.exists(f"{outputFolder}/{scene_folder}/depth"):
        os.mkdir(f"{outputFolder}/{scene_folder}/depth")
    if not os.path.exists(f"{outputFolder}/{scene_folder}/left"):
        os.mkdir(f"{outputFolder}/{scene_folder}/left")
    if not os.path.exists(f"{outputFolder}/{scene_folder}/center"):
        os.mkdir(f"{outputFolder}/{scene_folder}/center")


    saved_image = [False for x in camera_poses]

    camera_bb = 0
    minX = 10
    maxX = 0
    minY = 10
    maxY = 0
    minZ = 10
    maxZ = 0

    for pose in camera_poses:
        x, y, z = float(pose[5]), float(pose[6]), float(pose[7])
        if minX > x:
            minX = x
        if minY > y:
            minY = y
        if minZ > z:
            minZ = z
        if maxX < x:
            maxX = x
        if maxY < y:
            maxY = y
        if maxZ < z:
            maxZ = z

    camera_bb = (minX, maxX, minY, maxY, minZ, maxZ)

    skip_frame = True
    if len(camera_poses) > 0:
        poseLeft, pose, poseRight = load_camera_positions(camera_poses[pose_index], camera_bb = camera_bb)
    else:
        pose, poseRight = generate_sphere_positions(radius, theta, phi)

    # gaussian data
    #
    if trained_model is not None:
        gaussians_path = os.path.join(trained_model, "point_cloud/iteration_30000/point_cloud.ply")
        gaussians, bounding_box, center = util_gau.load_ply(gaussians_path)
    else:
        gaussians, bounding_box, center = util_gau.naive_gaussian()

    g_renderer.update_gaussian_data(gaussians)
    g_renderer.sort_and_update(g_camera, use_file, pose)
    update_activated_renderer_state(gaussians)
    frame_counter = 0  # Initialize frame counter
    csv_file_path = f"{outputFolder}/{scene_folder}/blur_score.csv"
    file = open(csv_file_path, 'a', newline='')
    writer = csv.writer(file)
    # Optional: Write headers to the CSV file
    writer.writerow(['Rendered ID', 'Blur score 0 - Low and 1- High'])

    # Check SSIM with training images
    num_blur_checks = 5
    if colmap_poses is not None:
        path = Path(colmap_poses)
        images_path = path.parents[1] / "images"
        # List all image files (assuming .jpg and .png files for this example)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        random.shuffle(image_files)
        # Open the CSV file in write mode ('w') or append mode ('a') as needed
        # CSV file to store the results
        values = []
        try:
            for x in range(num_blur_checks):
                # Select an image file sequentially from the shuffled list
                if len(image_files) > 0:  # Check if there are still images left to select
                    selected_image = image_files.pop(0)  # Remove the first image from the list to avoid repetition
                    refImage = cv2.imread(str(selected_image))  # Ensure to convert Path object to string
                    refImage_gray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
                    blur_ref = skimage.measure.blur_effect(refImage_gray, h_size=11)
                    values.append(blur_ref)
                    print("Reference blur score: ", blur_ref)
                else:
                    print("No more unique images available.")
                    break  # or handle according to your needs
            print("Average blur score: ", sum(values) / len(values))
            writer.writerow(["Reference:", str(sum(values) / len(values))])
        except Exception as e:
            print(e)
            writer.writerow(["Reference:", 0])

    file.close()

    # Compute camera central


    while not glfw.window_should_close(window):
        file = open(csv_file_path, 'a', newline='')
        writer = csv.writer(file)

        skip_frames = True
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        frame_counter += 1
        # if animate:
        #    radius, theta, phi = update_sphere_positions(radius, theta, phi)
        #

        # Check if 30 frames have passed
        if frame_counter == 30 and animate:
            pose_index += 1  # Move to the next camera pose
            frame_counter = 0  # Reset the frame counter

        # Ensure pose_index doesn't exceed your camera_poses list length
        if 0 < len(camera_poses) <= pose_index:
            glfw.set_window_should_close(window, True)
            continue

        # pose, poseRight = generate_sphere_positions(radius, theta, phi)
        if len(camera_poses) > 0:
            poseLeft, pose, poseRight = load_camera_positions(camera_poses[pose_index], bounding_box, center ,camera_bb = camera_bb )

        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_intrin_lazy()

        if frame_counter == 29:
            skip_frames = False

        if switch_lr_pose:
            g_renderer.update_camera_pose(g_camera, use_file, poseRight)
        else:
            g_renderer.update_camera_pose(g_camera, use_file, poseLeft)

        g_renderer.draw()

        if len(camera_poses) > 0 and (manual_save or saved_image[pose_index] is False) and not skip_frames or manual_save:
            print(f"Saving from Image ID: {camera_poses[pose_index][0]}. Rendered image name: {pose_index}.png")
            ######### LEFT FBO

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_left)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            g_renderer.update_camera_pose(g_camera, use_file, poseLeft)

            g_renderer.draw()

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_left)  # Ensure FBO is bound if not already
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            # Convert pixels to a numpy array and then to an image
            imageLeft = Image.frombytes("RGB", (width, height), pixels)

            ######### CENTER FBO
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_center)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            g_renderer.update_camera_pose(g_camera, use_file, pose)
            g_renderer.draw()

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_center)  # Ensure FBO is bound if not already
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            # Convert pixels to a numpy array and then to an image
            imageCenter = Image.frombytes("RGB", (width, height), pixels)
            # OpenGL's origin is in the bottom-left corner and PIL's is in the top-left.
            # We need to flip the imageLeft vertically.


            ###### RIGHT FBO

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_right)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            g_renderer.update_camera_pose(g_camera, use_file, poseRight)
            g_renderer.draw()

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_right)  # Ensure FBO is bound if not already
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            # Convert pixels to a numpy array and then to an image
            imageRight = Image.frombytes("RGB", (width, height), pixels)
            # OpenGL's origin is in the bottom-left corner and PIL's is in the top-left.
            # We need to flip the imageRight vertically.

            ####### DEPTH FBO
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_disparity)  # Ensure default framebuffer is active for ImGui
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_disparity)  # Ensure default framebuffer is active for ImGui
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            g_renderer.set_is_depth_image(1)  # depth
            g_renderer.update_camera_pose(g_camera, use_file, poseLeft)

            g_renderer.set_render_mod(-1)  # depth
            g_renderer.draw()

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_disparity)  # Ensure FBO is bound if not already
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RED, gl.GL_FLOAT)
            g_renderer.set_is_depth_image(0)  # depth

            # Convert pixels to a numpy array and then to an image
            imageDepth = Image.frombytes("I;16", (width, height), pixels)

            # Convert the raw data to a NumPy array
            disparity_map = np.frombuffer(pixels, dtype=np.float32).reshape((height, width))
            disparity_map_scaled = ((disparity_map * width) * 56.495).astype(np.uint16)
            disparity_map_rotated = cv2.rotate(disparity_map_scaled, cv2.ROTATE_180)
            cv2.imwrite(f"{outputFolder}/{scene_folder}/depth/{pose_index}.png", disparity_map_rotated)

            #image_array = np.array(imageDepth, dtype=np.uint16)
            #image_scaled = image_array * 1160
            #image_scaled = np.clip(image_scaled, 0, 65535).astype(np.uint16)
            #imageDepth = Image.fromarray(image_scaled)

            # OpenGL's origin is in the bottom-left corner and PIL's is in the top-left.
            # We need to flip the imageDepth vertically.

            imageLeft_np = np.array(imageLeft)
            imageLeft_gray = cv2.cvtColor(imageLeft_np, cv2.COLOR_BGR2GRAY)
            blur_ref = skimage.measure.blur_effect(imageLeft_gray, h_size=23)
            print(f"Rendered blur score: {blur_ref}")
            writer.writerow([str(pose_index), blur_ref])

            imageDepth = imageDepth.transpose(Image.FLIP_TOP_BOTTOM)
            imageLeft = imageLeft.transpose(Image.FLIP_TOP_BOTTOM)
            imageRight = imageRight.transpose(Image.FLIP_TOP_BOTTOM)
            imageCenter = imageCenter.transpose(Image.FLIP_TOP_BOTTOM)

            #imageDepth.save(f"{outputFolder}/{scene_folder}/depth/{pose_index}.png")
            imageRight.save(f"{outputFolder}/{scene_folder}/right/{pose_index}.png")
            imageLeft.save(f"{outputFolder}/{scene_folder}/left/{pose_index}.png")
            imageCenter.save(f"{outputFolder}/{scene_folder}/center/{pose_index}.png")

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)  # Ensure default framebuffer is active for ImGui
            g_renderer.set_render_mod(g_render_mode - 3)
            saved_image[pose_index] = True
            manual_save = False
            skip_frame = True
            print(f"Generated {pose_index} images of scene: {os.path.split(trained_model)[-1]}")

        skip_frame = False
        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()

        if g_show_control_win:
            if imgui.begin("Control", True):
                # rendering backend
                changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)])
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    update_activated_renderer_state(gaussians)

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")
                imgui.text(f"# of Gaus = {len(gaussians)}")
                if imgui.button(label='open ply'):
                    file_path = filedialog.askopenfilename(title="open ply",
                                                           initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                                                           filetypes=[('ply file', '.ply')]
                                                           )
                    if file_path:
                        try:
                            gaussians, bounding_box, center = util_gau.load_ply(file_path)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera, use_file, pose)
                        except RuntimeError as e:
                            pass

                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )

                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()

                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True

                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)

                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 3)

                # sort button
                if imgui.button(label='sort Gaussians'):
                    g_renderer.sort_and_update(g_camera, use_file, pose)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox(
                    "auto sort", g_auto_sort,
                )
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera, use_file, pose)

                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3;
                    stride = nrChannels * width;
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])
                    # save intermediate information
                    # np.savez(
                    #     "save.npz",
                    #     gau_xyz=gaussians.xyz,
                    #     gau_s=gaussians.scale,
                    #     gau_rot=gaussians.rot,
                    #     gau_c=gaussians.sh,
                    #     gau_a=gaussians.opacity,
                    #     viewmat=g_camera.get_view_matrix(),
                    #     projmat=g_camera.get_project_matrix(),
                    #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                    # )
                imgui.end()

        if g_show_camera_win:
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
            )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/output")
            imgui.text("Use control panel to change setting")
            imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        file.close() # for incremental updates

    file.close()
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    parser.add_argument('--gs_model', type=str)
    parser.add_argument('--colmap_poses', type=str)

    args = parser.parse_args()
    try:
        main(args.gs_model, args.colmap_poses)
    except Exception as e:
        print(f"An error occurred: {e}")