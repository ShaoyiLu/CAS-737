'''
only need to install the following modules, code should run
pip install PyOpenGL
pip install glfw
'''
import sys
from OpenGL import GL as gl
import glfw
from OpenGL.GLU import *

# Rendering-related Code ------------------------------------------------------------
'''for capture screen shots'''
def dump_framebuffer_to_ppm(ppm_name, fb_width, fb_height):
    pixelChannel = 3
    pixels = gl.glReadPixels(0, 0, fb_width, fb_height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    fout = open(ppm_name, "w")
    fout.write('P3\n{} {}\n255\n'.format(int(fb_width), int(fb_height)))
    for i in range(0, fb_height):
        for j in range(0, fb_width):
            cur = pixelChannel * ((fb_height - i - 1) * fb_width + j)
            fout.write('{} {} {} '.format(int(pixels[cur]), int(pixels[cur+1]), int(pixels[cur+2])))
        fout.write('\n')
    fout.flush()
    fout.close()

screen_width, screen_height = 1024, 512
x_mid = x_size = screen_width/2
y_mid = y_size = screen_height/2
ss_id = 0 # screenshot id

''' create window'''
if not glfw.init():
    sys.exit(1)

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
title = 'Curve Canvas'
window = glfw.create_window(screen_width, screen_height, title, None, None)
if not window:
    print('GLFW Window Failed')
    sys.exit(2)
glfw.make_context_current(window)
gl.glClearColor(0.3, 0.4, 0.5, 0)

shaders = {
    gl.GL_VERTEX_SHADER: '''\
    #version 330 core
    layout(location = 0) in vec3 aPos;
    void main() {
        gl_Position = vec4(aPos, 1);
    }
''',
    gl.GL_FRAGMENT_SHADER: '''\
    #version 330 core
    out vec3 color;
    void main() {
      color = vec3(0.9,0.8,0.7);
    }
'''}

program_id = gl.glCreateProgram()
shader_ids = []
for shader_type, shader_src in shaders.items():
    shader_id = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader_id, shader_src)
    gl.glCompileShader(shader_id)
    # check if compilation was successful
    result = gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS)
    nlog = gl.glGetShaderiv(shader_id, gl.GL_INFO_LOG_LENGTH)
    if nlog:
        logmsg = gl.glGetShaderInfoLog(shader_id)
        print("Shader Error", logmsg)
        sys.exit(1)
    gl.glAttachShader(program_id, shader_id)
    shader_ids.append(shader_id)

gl.glLinkProgram(program_id)
result = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
nlog = gl.glGetProgramiv(program_id, gl.GL_INFO_LOG_LENGTH)
if nlog:
    logmsg = gl.glGetProgramInfoLog(program_id)
    print("Link Error", logmsg)
    sys.exit(1)
gl.glUseProgram(program_id)

def draw_keys(keys_data):
    key_number = int(len(keys_data)/3)
    if key_number == 0:
        return
    vertex_array_id = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vertex_array_id)
    attr_id = 0
    vertex_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
    array_type = (gl.GLfloat * len(keys_data))
    sizeof_float = ctypes.sizeof(ctypes.c_float)
    gl.glBufferData(gl.GL_ARRAY_BUFFER,
                    len(keys_data) * sizeof_float,
                    array_type(*keys_data),
                    gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(
        attr_id,  # attribute 0.
        3,  # components per vertex attribute
        gl.GL_FLOAT,  # type
        False,  # to be normalized?
        0,  # stride
        None  # array buffer offset
    )
    gl.glEnableVertexAttribArray(attr_id)
    gl.glPointSize(10)
    gl.glDrawArrays(gl.GL_POINTS, 0, key_number)

def draw_curve(curve_data, DRAW_MODE=gl.GL_POINTS):
    if curve_data == None:
        return
    curve_pt_number = int(len(curve_data)/3)
    print('curve pt num:', curve_pt_number)
    if curve_pt_number == 0:
        return

    vertex_array_id = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vertex_array_id)
    attr_id = 0
    vertex_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
    array_type = (gl.GLfloat * len(curve_data))
    sizeof_float = ctypes.sizeof(ctypes.c_float)
    gl.glBufferData(gl.GL_ARRAY_BUFFER,
                    len(curve_data) * sizeof_float,
                    array_type(*curve_data),
                    gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(
        attr_id,  # attribute 0.
        3,  # components per vertex attribute
        gl.GL_FLOAT,  # type
        False,  # to be normalized?
        0,  # stride
        None  # array buffer offset
    )
    gl.glEnableVertexAttribArray(attr_id)
    gl.glPointSize(5)
    gl.glDrawArrays(DRAW_MODE, 0, curve_pt_number)

def draw_control_polygon(key_points):
    key_number = len(key_points)
    segment_number = int((key_number) / 12)
    for i in range(0, segment_number):
        keys_on_one_polygon = key_points[4*i*3:4*(i+1)*3]
        draw_curve(keys_on_one_polygon, gl.GL_LINE_STRIP)
    #last strip
    if key_number > segment_number * 12:
        keys_on_one_polygon = key_points[segment_number * 12:]
        draw_curve(keys_on_one_polygon, gl.GL_LINE_STRIP)

# end of Rendering-related Code ------------------------------------------------------------

class Point3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other): return Point3(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other): return self.__add__(-other)

    def __mul__(self, other): return Point3(self.x*other, self.y*other, self.z*other)

    @classmethod
    def point_list_to_num_list(cls, point_list):
        if point_list == None or len(point_list) <= 0:
            return
        num_list = []
        for i in range(0, len(point_list)):
            pt = point_list[i]
            num_list += [pt.x, pt.y, pt.z]
        return num_list

    @classmethod
    def num_list_to_point_list(cls, num_list):
        if num_list == None or len(num_list) <= 0:
            return
        pt_count = int(len(num_list) / 3)
        point_list = []
        for i in range(0, pt_count):
            pt = Point3(num_list[3*i], num_list[3*i+1], num_list[3*i+2])
            point_list.append(pt)
        return point_list


key_points = []
def mouse_button_callback(window, button, action, mods):
    global key_points
    x_pos, y_pos = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        print('right button at', x_pos, y_pos)
        key_points.clear()
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        x_click = (x_pos-x_mid)/x_size
        y_click = -(y_pos-y_mid)/y_size
        key_points += [x_click, y_click, 0]
        print('left button at', x_click, y_click)

def cursor_pos_callback(window, x_pos, y_pos):
    x_hover = (x_pos - x_mid) / x_size
    y_hover = -(y_pos - y_mid) / y_size
    print(x_hover, y_hover)

# bezier curve interpolation ---------------------------------
def gen_polyline_points(key_points):
    curve_points = key_points
    return curve_points

def interpolate_bezier_position_at(t, p0, p1, p2, p3):
    b0 = pow(1 - t, 3)
    b1 = 3*t * pow(1 - t, 2)
    b2 = 3*pow(t,2)*(1-t)
    b3 = pow(t,3)
    result = p0*b0 + p1*b1 + p2*b2 + p3*b3
    return result

# the new catmull rom method ---------------------------------
def catmull_rom(t, p0, p1, p2, p3):
    c0 = -0.5*pow(t, 3) + pow(t, 2) - 0.5*t
    c1 = 1.5*pow(t, 3) - 2.5*pow(t, 2) + 1
    c2 = -1.5*pow(t, 3) + 2*pow(t, 2) + 0.5*t
    c3 = 0.5*pow(t, 3) - 0.5*pow(t, 2)
    result = p0*c0 + p1*c1 + p2*c2 + p3*c3
    return result

def gen_catmull_rom_points(key_points):
    key_number = len(key_points)
    if key_number < 2:
        return
    step_size = 0.01
    step_count = int(1/step_size)

    segment_number = key_number - 1

    print('seg num', segment_number)
    positions_on_curve = []

    for i in range(0, segment_number):
        #segment keys

        # First point
        if i == 0:
            p0 = key_points[i]
            p1 = key_points[i]
            if i + 1 < key_number:
                p2 = key_points[i + 1]
            else:
                p2 = key_points[-1]
            if i + 2 < key_number:
                p3 = key_points[i + 2]
            else:
                p3 = key_points[-1]

        # Last point
        elif i == key_number - 1:
            if i - 2 >= 0:
                p0 = key_points[i - 2]
            else:
                p0 = key_points[0]
            p1 = key_points[i - 1]
            p2 = key_points[i]
            p3 = key_points[i]

        # Middle point
        else:
            p0 = key_points[i - 1]
            p1 = key_points[i]
            p2 = key_points[i + 1]
            if i + 2 < key_number:
                p3 = key_points[i + 2]
            else:
                p3 = key_points[-1]

        for s in range(0, step_count + 1):
            t = s * step_size
            one_position_on_curve = catmull_rom(t, p0, p1, p2, p3)
            positions_on_curve.append(one_position_on_curve)

    return positions_on_curve

# run-time UI loop ---------------------------------------------------
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_pos_callback)
positions_on_curve = []
while (
    glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and
    not glfw.window_should_close(window)
):

    #press key p will capture screen shot
    if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS:
        print ("Capture Window ", ss_id)
        buffer_width, buffer_height = glfw.get_framebuffer_size(window)
        ppm_name = "CurveCanvas-ss" + str(ss_id) + ".ppm"
        dump_framebuffer_to_ppm(ppm_name, buffer_width, buffer_height)
        ss_id += 1

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    if len(key_points) > 0:
        draw_keys(key_points)
        positions_on_curve = gen_polyline_points(Point3.num_list_to_point_list(key_points))
        draw_control_polygon(Point3.point_list_to_num_list(positions_on_curve))
        positions_on_curve = gen_catmull_rom_points(Point3.num_list_to_point_list(key_points))
        draw_curve(Point3.point_list_to_num_list(positions_on_curve))

    glfw.swap_buffers(window)
    glfw.poll_events()

#release resource
for shader_id in shader_ids:
    gl.glDetachShader(program_id, shader_id)
    gl.glDeleteShader(shader_id)
gl.glUseProgram(0)
gl.glDeleteProgram(program_id)
