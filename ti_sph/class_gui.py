import taichi as ti
import numpy as np


@ti.kernel
def pos_normalizer(num: ti.template(), global_pos: ti.template(), pos_lb: ti.template(), pos_rt: ti.template(), output_normalized_pos: ti.template()):
    for i in range(num[None]):
        output_normalized_pos[i] = (
            global_pos[i] - pos_lb[None]) / (pos_rt[None] - pos_lb[None])

# UNDONE AND USELESS
# @ti.kernel
# def part_size_normalizer(part_size: ti.template(), pos_lb: ti.template(), pos_rt: ti.template(), gui_res: ti.template(),  relaxing_factor: ti.template(), output_normalized_part_size: ti.template()):
#     output_normalized_part_size = part_size[None] / (pos_lb[None] - pos_rt[None]) * gui_res[None] * relaxing_factor[None]


class Gui():
    def __init__(self, config_gui):
        self.window = ti.ui.Window(
            "Fluid Simulation", (config_gui.res[None][0], config_gui.res[None][1]), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.make_camera()
        self.camera.position(
            config_gui.cam_pos[None][0], config_gui.cam_pos[None][1], config_gui.cam_pos[None][2])
        self.camera.lookat(
            config_gui.cam_look[None][0], config_gui.cam_look[None][1], config_gui.cam_look[None][2])
        self.camera.fov(55)
        self.canvas_color = (
            (config_gui.canvas_color[None][0], config_gui.canvas_color[None][1], config_gui.canvas_color[None][2]))
        self.ambient_color = (
            config_gui.ambient_light_color[None][0], config_gui.ambient_light_color[None][1], config_gui.ambient_light_color[None][2])
        self.point_light_pos = (
            config_gui.point_light_pos[None][0], config_gui.point_light_pos[None][1], config_gui.point_light_pos[None][2])
        self.point_light_color = (
            config_gui.point_light_color[None][0], config_gui.point_light_color[None][1], config_gui.point_light_color[None][2])
        # self.dispaly_radius = config.part_size[1] * 0.5

        # Toggles
        self.show_bound = False
        self.show_help = True
        self.show_run_info = True
        self.op_system_run = False
        self.op_write_file = False
        self.op_refresh_window = True

    def monitor_listen(self):
        self.camera.track_user_inputs(
            self.window, movement_speed=0.03, hold_key=ti.ui.RMB)

        if self.show_help:
            self.window.GUI.begin("options", 0.05, 0.3, 0.2, 0.2)
            self.window.GUI.text("h: help")
            self.window.GUI.text("w: front")
            self.window.GUI.text("s: back")
            self.window.GUI.text("a: left")
            self.window.GUI.text("d: right")
            self.window.GUI.text("RMB: rotate")
            self.window.GUI.text("b: display boundary")
            self.window.GUI.text("r: run system")
            self.window.GUI.text("f: write file")
            self.window.GUI.text("c: refresh window")
            self.window.GUI.end()

        if self.window.get_event(ti.ui.PRESS):
            # run
            if self.window.event.key == 'r':
                self.op_system_run = not self.op_system_run
                print("start to run:", self.op_system_run)

            if self.window.event.key == 'f':
                self.op_write_file = not self.op_write_file
                print("write file:", self.op_write_file)

            if self.window.event.key == 'b':
                self.show_bound = not self.show_bound
                print("show boundary:", self.show_bound)

            if self.window.event.key == 'i':
                self.show_run_info = not self.show_run_info
                print("show run information:", self.show_run_info)

            if self.window.event.key == 'h':
                self.show_help = not self.show_help
                print("show help:", self.show_help)

            if self.window.event.key == 'c':
                self.op_refresh_window = not self.op_refresh_window
                print("refresh window:", self.op_refresh_window)

    def env_set_up(self):
        self.canvas.set_background_color(self.canvas_color)

    def scene_setup(self):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light(self.ambient_color)
        self.scene.point_light(
            pos=self.point_light_pos, color=self.point_light_color)

    def scene_add_parts(self, obj, length):
        truncate_pos(obj)
        self.scene.particles(
            obj.basic.pos, per_vertex_color=obj.color.color_vec, radius=length/2)

    def scene_render(self):
        self.canvas.scene(self.scene)  # Render the scene
        self.window.show()

@ti.kernel
def truncate_pos(obj: ti.template()):
    for i in range(obj.info.part_num[None]-obj.info.stack_top[None]):
        obj.basic.pos[obj.info.part_num[None]-i-1][0] = 25535

# def ti2numpy_color(num, obj_color):
#     return obj_color.to_numpy()[:num]


# def set_unused_par(obj, config):
#     # temporary method: throw the unused particles away so they aren't rendered
#     unused_pos = ti.Vector.field(config.dim[None], float, ())
#     unused_pos.from_numpy(
#         np.array([533799.0] * config.dim[None], dtype=np.float32))
#     sub_set_unused_par(obj, unused_pos)


# @ti.kernel
# def sub_set_unused_par(obj: ti.template(), unused_pos: ti.template()):
#     for i in range(obj.part_num[None], obj.max_part_num):
#         obj.pos[i] = unused_pos[None]
#         obj.gui_2d_pos[i] = unused_pos[None]



