import math

import glfw
import OpenGL.GL as gl
import glm
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

# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#g_camera = util.Camera(1988, 2964) # middlebury
g_camera = util.Camera(1280, 1920)
pose_index = 0
use_file = False
manual_save = False
switch_lr_pose = False

BACKEND_OGL = 0
BACKEND_CUDA = 1
g_renderer_list = [
    None,  # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = True
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 6


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
    global pose_index, use_file, manual_save, switch_lr_pose
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)
        elif key == glfw.KEY_N:
            pose_index += 1
        elif key == glfw.KEY_I:
            switch_lr_pose = not switch_lr_pose
        elif key == glfw.KEY_X:
            manual_save = True
        elif key == glfw.KEY_P:
            use_file = not use_file
        elif key == glfw.KEY_UP or key == glfw.KEY_W:
            g_camera.camera_position += g_camera.camera_front;
        elif key == glfw.KEY_DOWN or key == glfw.KEY_S:
            g_camera.camera_position -= g_camera.camera_front;
        elif key == glfw.KEY_LEFT or key == glfw.KEY_A:
            g_camera.camera_position -= glm.normalize(glm.cross(g_camera.camera_front, g_camera.camera_up))
        elif key == glfw.KEY_RIGHT or key == glfw.KEY_D:
            g_camera.camera_position += glm.normalize(glm.cross(g_camera.camera_front, g_camera.camera_up))


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
        "camera_position": camera_position
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


def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables, pose_index, manual_save, switch_lr_pose

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

    from renderer_cuda import CUDARenderer
    g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]

    g_renderer_idx = BACKEND_OGL
    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)
    width, height = glfw.get_framebuffer_size(window)

    # Step 1: Create an FBO and a texture attachment
    fbo_left = gl.glGenFramebuffers(1)
    fbo_left_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbo_left_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_left)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbo_left_texture, 0)
    fbo_right = gl.glGenFramebuffers(1)
    fbo_right_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbo_right_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_right)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbo_right_texture, 0)
    # Step 1: Create an FBO and a texture attachment
    fbo_depth = gl.glGenFramebuffers(1)
    fbo_depth_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbo_depth_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R16, width, height, 0, gl.GL_RED, gl.GL_UNSIGNED_SHORT, None)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_depth)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbo_depth_texture, 0)

    # Check FBO status
    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
        print("Error: Framebuffer is not complete.")
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    # Example usage
    csv_file_path = 'camera_data.csv'  # Specify the path to your CSV file

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w") as file:
            file.close()
    camera_poses = read_camera_poses_from_csv(csv_file_path)

    if len(camera_poses) < 1:
        camera_front = glm.vec3(float(0), float(0), float(0))
        camera_up = glm.vec3(float(0), float(0), float(1))
        camera_position = glm.vec3(float(-3), float(0), float(1))

        # Convert row data to a dictionary and append to the list
        pose = {
            "camera_front": camera_front,
            "camera_up": camera_up,
            "camera_position": camera_position
        }
        camera_poses.append(pose)
        print(camera_poses)

    saved = [False for x in camera_poses]
    camera_right_poses = []
    for pose in camera_poses:
        front = pose["camera_front"]
        up = pose["camera_up"]
        pos = pose["camera_position"]

        # Right vector:
        right = glm.normalize(glm.cross(front, up))
        baseline = 0.193001

        right_pos = pos + (right * baseline)

        res = glm.distance(pos, right_pos)

        print(f"Absolute baseline: {res}")

        pose = {
            "camera_front": front,
            "camera_up": up,
            "camera_position": right_pos
        }

        camera_right_poses.append(pose)

    if not os.path.exists("out"):
        os.mkdir("out")
    if not os.path.exists("out/right"):
        os.mkdir("out/right")
    if not os.path.exists("out/depth"):
        os.mkdir("out/depth")
    if not os.path.exists("out/left"):
        os.mkdir("out/left")
    # settings

    while not glfw.window_should_close(window):
        pose = camera_poses[pose_index]
        poseRight = camera_right_poses[pose_index]
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_intrin_lazy()

        if switch_lr_pose:
            g_renderer.update_camera_pose(g_camera, use_file, poseRight)
        else:
            g_renderer.update_camera_pose(g_camera, use_file, pose)

        g_renderer.draw()

        if saved[pose_index] is False or manual_save:
            ######### LEFT FBO
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_left)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            g_renderer.draw()

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_left)  # Ensure FBO is bound if not already
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            # Convert pixels to a numpy array and then to an image
            image = Image.frombytes("RGB", (width, height), pixels)
            # OpenGL's origin is in the bottom-left corner and PIL's is in the top-left.
            # We need to flip the image vertically.
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(f"out/left/{pose_index}.png")
            ###### RIGHT FBO

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_right)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            g_renderer.update_camera_pose(g_camera, use_file, poseRight)
            g_renderer.draw()

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_right)  # Ensure FBO is bound if not already
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            # Convert pixels to a numpy array and then to an image
            image = Image.frombytes("RGB", (width, height), pixels)
            # OpenGL's origin is in the bottom-left corner and PIL's is in the top-left.
            # We need to flip the image vertically.
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(f"out/right/{pose_index}.png")

            ####### DEPTH FBO
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_depth)  # Ensure default framebuffer is active for ImGui
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_depth)  # Ensure default framebuffer is active for ImGui
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            g_renderer.set_render_mod(-1)  # depth
            g_renderer.draw()

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_depth)  # Ensure FBO is bound if not already
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RED , gl.GL_UNSIGNED_SHORT)

            # Convert pixels to a numpy array and then to an image
            image = Image.frombytes("I;16", (width, height), pixels)
            # OpenGL's origin is in the bottom-left corner and PIL's is in the top-left.
            # We need to flip the image vertically.
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(f"out/depth/{pose_index}.png")

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)  # Ensure default framebuffer is active for ImGui
            g_renderer.set_render_mod(g_render_mode - 3)
            saved[pose_index] = True
            manual_save = False

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
                            gaussians = util_gau.load_ply(file_path)
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
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    main()
