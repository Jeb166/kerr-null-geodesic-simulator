"""ModernGL renderer; exclusively consumes immutable display geometry."""

import numpy as np

from .scene import horizon_mesh, ribbon_vertices, singularity_marker
from .visuals import direction_arrows, ray_color, ray_markers, ray_opacity


SURFACE_VERTEX = """
#version 330
in vec3 in_position;
uniform mat4 u_view_projection;
void main() { gl_Position = u_view_projection * vec4(in_position, 1.0); }
"""
SURFACE_FRAGMENT = """
#version 330
uniform vec4 u_color;
out vec4 out_color;
void main() { out_color = u_color; }
"""
RIBBON_VERTEX = """
#version 330
in vec3 in_start;
in vec3 in_end;
in float in_t;
in float in_side;
uniform mat4 u_view_projection;
uniform vec2 u_viewport;
uniform float u_width;
void main() {
    vec4 p0 = u_view_projection * vec4(in_start, 1.0);
    vec4 p1 = u_view_projection * vec4(in_end, 1.0);
    // Camera-facing line width in physical pixels. Avoid pathological
    // perspective division when both endpoints are behind the camera.
    if (p0.w <= 0.0 && p1.w <= 0.0) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        return;
    }
    vec2 ndc0 = p0.xy / max(p0.w, 0.0001);
    vec2 ndc1 = p1.xy / max(p1.w, 0.0001);
    vec2 screen_dir = (ndc1 - ndc0) * u_viewport;
    float length_dir = length(screen_dir);
    vec2 normal = length_dir > 0.00001 ? vec2(-screen_dir.y, screen_dir.x) / length_dir : vec2(0.0, 1.0);
    vec4 clip = mix(p0, p1, in_t);
    clip.xy += normal * in_side * (u_width / u_viewport) * clip.w;
    gl_Position = clip;
}
"""

MARKER_VERTEX = """
#version 330
in vec3 in_center;
in vec2 in_corner;
uniform mat4 u_view_projection;
uniform vec2 u_viewport;
uniform float u_radius;
out vec2 v_uv;
void main() {
    vec4 clip = u_view_projection * vec4(in_center, 1.0);
    if (clip.w <= 0.0) { gl_Position = vec4(0,0,-2,1); return; }
    clip.xy += 2.0 * in_corner * (u_radius / u_viewport) * clip.w;
    gl_Position = clip;
    v_uv = in_corner;
}
"""
MARKER_FRAGMENT = """
#version 330
in vec2 v_uv;
uniform vec4 u_color;
uniform int u_glyph;
out vec4 out_color;
void main() {
    float r = length(v_uv);
    float square = max(abs(v_uv.x), abs(v_uv.y));
    float diamond = abs(v_uv.x)+abs(v_uv.y);
    bool ink = false;
    vec3 color = u_color.rgb;
    if (u_glyph == 1) { // START: solid ray color with a white pin
        ink = r < .88;
        if (r < .24) color = vec3(1.0);
    } else if (u_glyph == 2) { // finite escape: hollow ring
        ink = r > .56 && r < .88;
    } else if (u_glyph == 3) { // Sigma threshold: solid red circle
        ink = r < .84;
        color = vec3(1.0, .22, .20);
    } else if (u_glyph == 4) { // partial: outlined amber square
        ink = square < .86 && square > .56;
        color = vec3(1.0, .69, .23);
    } else if (u_glyph == 5) { // failed diagnostics: red X
        ink = square < .84 && abs(abs(v_uv.x)-abs(v_uv.y)) < .23;
        color = vec3(1.0, .22, .20);
    } else if (u_glyph == 6) { // numerical error: red diamond + center X
        ink = (diamond > .62 && diamond < .94)
              || (square < .42 && abs(abs(v_uv.x)-abs(v_uv.y)) < .16);
        color = vec3(1.0, .30, .30);
    } else if (u_glyph == 7) { // horizon: small diamond
        ink = diamond < .86;
    } else if (u_glyph == 8) { // disk: small square
        ink = square < .73;
    } else if (u_glyph == 9) { // chart: small outlined ring
        ink = r > .52 && r < .81;
    }
    if (!ink) discard;
    out_color = vec4(color, u_color.a);
}
"""

ARROW_VERTEX = """
#version 330
in vec3 in_start;
in vec3 in_end;
in float in_fraction;
in vec2 in_pixel_offset;
uniform mat4 u_view_projection;
uniform vec2 u_viewport;
void main() {
    vec4 first = u_view_projection * vec4(in_start, 1.0);
    vec4 last = u_view_projection * vec4(in_end, 1.0);
    if (first.w <= 0.0 || last.w <= 0.0) {
        gl_Position = vec4(0,0,-2,1); return;
    }
    vec2 tangent = (last.xy/last.w - first.xy/first.w) * u_viewport;
    tangent /= max(length(tangent), 0.0001);
    vec2 normal = vec2(-tangent.y, tangent.x);
    vec4 clip = mix(first, last, in_fraction);
    vec2 offset = tangent*in_pixel_offset.x + normal*in_pixel_offset.y;
    clip.xy += 2.0*offset/u_viewport*clip.w;
    gl_Position = clip;
}
"""


def _marker_vertices(position):
    x, y, z = position
    return np.array([(x, y, z, u, v) for u, v in
                     ((-1.,-1.),(1.,-1.),(-1.,1.),(-1.,1.),(1.,-1.),(1.,1.))],
                    dtype=np.float32)


def _arrow_vertices(positions):
    vertices = []
    for arrow in direction_arrows(positions):
        for side in (-1., 1.):
            # Two short ribbon arms, triangular tip points toward last sample.
            head = np.array([0., 0.]); tail = np.array([-12., 7.*side])
            normal = np.array([-(tail-head)[1], (tail-head)[0]])
            normal *= 1.3 / np.linalg.norm(normal)
            for offset in (head+normal, head-normal, tail+normal,
                           tail+normal, head-normal, tail-normal):
                vertices.append((*arrow.start, *arrow.end, arrow.fraction, *offset))
    return np.asarray(vertices, dtype=np.float32).reshape((-1, 9))


class SceneRenderer:
    def __init__(self, ctx, scene):
        import moderngl

        self.gl = moderngl
        self.ctx = ctx
        self.surface = ctx.program(vertex_shader=SURFACE_VERTEX,
                                   fragment_shader=SURFACE_FRAGMENT)
        self.ribbon = ctx.program(vertex_shader=RIBBON_VERTEX,
                                  fragment_shader=SURFACE_FRAGMENT)
        self.marker_program = ctx.program(vertex_shader=MARKER_VERTEX,
                                          fragment_shader=MARKER_FRAGMENT)
        self.arrow_program = ctx.program(vertex_shader=ARROW_VERTEX,
                                         fragment_shader=SURFACE_FRAGMENT)
        self.meshes = []
        self.rays = []
        surface = horizon_mesh(scene.horizon, scene.spin)
        if len(surface):
            self.horizon = self._surface(surface)
        else:
            self.horizon = None
        marker = singularity_marker(scene.singularity)
        if len(marker) == 1:
            # A small, always visible red cross marks the Schwarzschild point.
            cross = [np.array([[-0.09, 0, 0], [0.09, 0, 0]]),
                     np.array([[0, -0.09, 0], [0, 0.09, 0]]),
                     np.array([[0, 0, -0.09], [0, 0, 0.09]])]
            self.singularity = [self._line(ribbon_vertices(points)) for points in cross]
        else:
            self.singularity = [self._line(ribbon_vertices(marker))]
        self.axis = self._line(ribbon_vertices(
            np.array([[0., 0., -5.], [0., 0., 5.]])))
        for index, ray in enumerate(scene.rays):
            data = ribbon_vertices(ray.positions)
            line = self._line(data) if len(data) else None
            arrows = self._arrows(_arrow_vertices(ray.positions))
            markers = []
            color = ray_color(index)
            for marker in ray_markers(ray, index):
                markers.append((self._marker(marker.position), marker.glyph,
                                marker.rgba, marker.radius))
            self.rays.append((line, arrows, markers, color.rgb,
                              ray_opacity(ray.reliability)))

    def _surface(self, vertices):
        buffer = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.vertex_array(self.surface, [(buffer, "3f", "in_position")])
        self.meshes.append((vao, buffer))
        return vao, len(vertices)

    def _line(self, vertices):
        buffer = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.vertex_array(self.ribbon, [
            (buffer, "3f 3f 1f 1f", "in_start", "in_end", "in_t", "in_side")])
        self.meshes.append((vao, buffer))
        return vao, len(vertices)

    def _marker(self, position):
        buffer = self.ctx.buffer(_marker_vertices(position).tobytes())
        vao = self.ctx.vertex_array(self.marker_program, [
            (buffer, "3f 2f", "in_center", "in_corner")])
        self.meshes.append((vao, buffer))
        return vao

    def _arrows(self, vertices):
        if not len(vertices):
            return None
        buffer = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.vertex_array(self.arrow_program, [
            (buffer, "3f 3f 1f 2f", "in_start", "in_end", "in_fraction", "in_pixel_offset")])
        self.meshes.append((vao, buffer))
        return vao, len(vertices)

    def draw(self, scene, view_projection, framebuffer, *, show_markers=True):
        gl = self.gl
        ctx = self.ctx
        framebuffer.use()             # Qt's current QOpenGLWidget FBO, not FBO 0
        width, height = framebuffer.size  # physical pixels, including DPI scaling
        if width <= 0 or height <= 0:
            return
        ctx.viewport = (0, 0, width, height)
        framebuffer.clear(0.24, 0.25, 0.27, 1.0, depth=1.0)
        matrix_bytes = np.asarray(view_projection, dtype=np.float32).T.tobytes()
        self.surface["u_view_projection"].write(matrix_bytes)
        self.ribbon["u_view_projection"].write(matrix_bytes)
        self.ribbon["u_viewport"].value = (float(width), float(height))
        self.marker_program["u_view_projection"].write(matrix_bytes)
        self.marker_program["u_viewport"].value = (float(width), float(height))
        self.arrow_program["u_view_projection"].write(matrix_bytes)
        self.arrow_program["u_viewport"].value = (float(width), float(height))
        ctx.enable(gl.DEPTH_TEST | gl.BLEND)
        # Qt composites the widget FBO: retain opaque framebuffer alpha even
        # when semi-transparent horizon fragments blend into its RGB values.
        ctx.blend_func = (gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE)
        ctx.depth_mask = True
        if self.horizon is not None:
            # Depth test but no depth writes: the surface remains translucent.
            ctx.depth_mask = False
            self.surface["u_color"].value = (0.28, 0.58, 0.75, 0.16)
            self.horizon[0].render(gl.TRIANGLES, vertices=self.horizon[1])
        # Draw reference geometry and rays last, without depth rejection, so
        # ray segments inside the translucent marker cannot disappear.
        ctx.disable(gl.DEPTH_TEST)
        self.ribbon["u_color"].value = (0.67, 0.81, 0.86, 0.78)
        self.ribbon["u_width"].value = 1.5
        self.axis[0].render(gl.TRIANGLES, vertices=self.axis[1])
        self.ribbon["u_color"].value = (0.99, 0.16, 0.19, 1.0)
        self.ribbon["u_width"].value = 4.0
        for vao, count in self.singularity:
            vao.render(gl.TRIANGLES, vertices=count)
        self.ribbon["u_width"].value = 3.0
        for line, arrows, markers, rgb, opacity in self.rays:
            if line:
                self.ribbon["u_color"].value = (*rgb, opacity)
                line[0].render(gl.TRIANGLES, vertices=line[1])
            if show_markers:
                if arrows:
                    self.arrow_program["u_color"].value = (*rgb, max(.70, opacity))
                    arrows[0].render(gl.TRIANGLES, vertices=arrows[1])
                for vao, glyph, color, radius in markers:
                    self.marker_program["u_glyph"].value = glyph
                    self.marker_program["u_color"].value = color
                    self.marker_program["u_radius"].value = radius
                    vao.render(gl.TRIANGLES, vertices=6)
        ctx.depth_mask = True

    def release(self):
        for vao, buffer in self.meshes:
            vao.release()
            buffer.release()
        self.ribbon.release()
        self.surface.release()
        self.marker_program.release()
        self.arrow_program.release()
