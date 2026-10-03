"""Qt window/input adapter; physics solves run outside the Qt render loop."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from html import escape
from time import perf_counter

import numpy as np

from kerr_geodesics import IntegrationSettings, RayDefinition, solve_rays
from kerr_geodesics._native import HAS_NUMBA

from .camera import FreeCamera
from .scene import DisplayScene, geometry_scene, trajectory_to_display
from .visuals import ray_color, termination_reason


def diagnostic_definitions():
    """Fixed local launch definitions, separately rebuilt for the selected spin."""
    return (
        ("Exterior escape", RayDefinition([4., .1, .5], [1., 0., 0.])),
        ("Inclined horizon return", RayDefinition([4., .5, .6], [-1., 0., 0.])),
        ("Signed-r disk passage", RayDefinition([.8, .1, 3.], [0., 0., -1.])),
    )


def _solve_static_scene(spin, backend):
    rays = diagnostic_definitions()
    results = solve_rays((definition for _, definition in rays), spin,
                         IntegrationSettings(escape_radius=8., max_affine_parameter=20.),
                         backend=backend)
    return geometry_scene(spin, (
        trajectory_to_display(name, result)
        for (name, _), result in zip(rays, results)))


def launch(spin=0.95, backend="auto"):
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QSurfaceFormat
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
        from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QWidget
        import moderngl
    except ImportError as exc:
        raise RuntimeError("viewer requires the optional PySide6 + ModernGL dependencies") from exc

    from .renderer import SceneRenderer

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication.instance() or QApplication([])
    initial_scene = geometry_scene(spin)
    compute = ThreadPoolExecutor(max_workers=1, thread_name_prefix="static-geodesics")
    future = compute.submit(_solve_static_scene, spin, backend)

    class ViewerWidget(QOpenGLWidget):
        def __init__(self):
            super().__init__()
            self.setFocusPolicy(Qt.StrongFocus)
            self.setMouseTracking(True)
            self.scene: DisplayScene = initial_scene
            self.camera = FreeCamera()
            self.renderer = None
            self.ctx = None
            self._qt_framebuffer = None
            self._qt_framebuffer_id = -1
            self._keys = set()
            self._last_mouse = None
            self._last_tick = perf_counter()
            self.fps = 0.
            self._frames = 0
            self._fps_start = perf_counter()
            self.debug_markers = True
            self.frame_timer = QTimer(self)
            self.frame_timer.timeout.connect(self.tick)
            self.frame_timer.start(16)

        def initializeGL(self):
            self.ctx = moderngl.create_context(require=330)
            self.renderer = SceneRenderer(self.ctx, self.scene)

        def resizeGL(self, width, height):
            # QOpenGLWidget recreates its FBO on resize; detect it in paintGL.
            self._qt_framebuffer_id = -1

        def paintGL(self):
            if self.ctx is None:
                return
            qt_id = self.defaultFramebufferObject()
            if qt_id != self._qt_framebuffer_id:
                self._qt_framebuffer = self.ctx.detect_framebuffer()
                self._qt_framebuffer_id = qt_id
            width, height = self._qt_framebuffer.size
            aspect = width/max(height, 1)
            view_projection = self.camera.projection(aspect) @ self.camera.view()
            self.renderer.draw(self.scene, view_projection, self._qt_framebuffer,
                               show_markers=self.debug_markers)
            self._frames += 1
            now = perf_counter()
            if now - self._fps_start >= .5:
                self.fps = self._frames/(now-self._fps_start)
                self._frames = 0
                self._fps_start = now

        def set_scene(self, new_scene):
            self.scene = new_scene
            if self.ctx is not None:
                self.makeCurrent()
                self.renderer.release()
                self.renderer = SceneRenderer(self.ctx, new_scene)
                self.doneCurrent()
            self.update()

        def tick(self):
            now = perf_counter()
            dt = min(now-self._last_tick, .05)
            self._last_tick = now
            keys = self._keys
            self.camera.move(forward=(Qt.Key_W in keys)-(Qt.Key_S in keys),
                             lateral=(Qt.Key_D in keys)-(Qt.Key_A in keys),
                             vertical=(Qt.Key_E in keys)-(Qt.Key_Q in keys), dt=dt)
            self.update()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.window().close()
            elif event.key() == Qt.Key_M and not event.isAutoRepeat():
                self.debug_markers = not self.debug_markers
                self.update()
            else:
                self._keys.add(event.key())

        def keyReleaseEvent(self, event):
            self._keys.discard(event.key())

        def focusOutEvent(self, event):
            self._keys.clear()
            super().focusOutEvent(event)

        def mousePressEvent(self, event):
            self._last_mouse = event.position()
            self.setFocus()

        def mouseReleaseEvent(self, event):
            self._last_mouse = None

        def mouseMoveEvent(self, event):
            current = event.position()  # Qt logical pixels: DPI-independent rotation
            if self._last_mouse is not None:
                dx = current.x()-self._last_mouse.x()
                dy = current.y()-self._last_mouse.y()
                if event.buttons() & Qt.RightButton:
                    self.camera.turn(dx, dy)
                elif event.buttons() & Qt.LeftButton:
                    self.camera.orbit(dx, dy)
            self._last_mouse = current

        def wheelEvent(self, event):
            steps = event.angleDelta().y()/120.
            self.camera.speed = float(np.clip(self.camera.speed*(1.18**steps), .03, 30.))

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Kerr null geodesics — v0.3A-r1 viewer")
            self.view = ViewerWidget()
            self.panel = QLabel()
            self.panel.setMinimumWidth(330)
            self.panel.setMaximumWidth(410)
            self.panel.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            self.panel.setWordWrap(True)
            self.panel.setTextFormat(Qt.RichText)
            self.panel.setStyleSheet("QLabel { color: #dddddd; background: #303136; padding: 12px; }")
            root = QWidget()
            layout = QHBoxLayout(root)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.view, 1)
            layout.addWidget(self.panel)
            self.setCentralWidget(root)
            self.resize(1150, 740)
            self.check_timer = QTimer(self)
            self.check_timer.timeout.connect(self.check_compute)
            self.check_timer.start(80)
            self._loaded = False
            self._error = None
            self.update_panel()

        def check_compute(self):
            if future.done() and not self._loaded:
                self._loaded = True
                try:
                    self.view.set_scene(future.result())
                except Exception as exc:
                    self._error = str(exc)
            self.update_panel()

        def update_panel(self):
            scene = self.view.scene
            horizon = scene.horizon
            def radius(value):
                return "—" if value is None else f"{value:.6f}"
            lines = [f"<b>a = {spin:g}</b> · {escape(horizon.regime.value)}",
                     f"r+ = {radius(horizon.r_plus)} · r- = {radius(horizon.r_minus)}",
                     f"horizon: {'yes' if horizon.exists else 'no'} · "
                     f"rays: {len(scene.rays)}/{len(diagnostic_definitions())}",
                     f"backend: {escape(backend)} (Numba: {'yes' if HAS_NUMBA else 'no'})",
                     f"FPS: {self.view.fps:.1f} · speed: {self.view.camera.speed:.2f}",
                     f"Debug markers: {'ON' if self.view.debug_markers else 'OFF'} (M)",
                     "<small>Right drag look · Left drag orbit · WASD fly · Q/E up/down<br>"
                     "Wheel speed · Esc close</small>"]
            if not self._loaded:
                lines.append("Solving static rays…")
            if self._error is not None:
                lines.append("Physics solve failed: "+escape(self._error))
            for index, ray in enumerate(scene.rays):
                color = ray_color(index)
                events = ", ".join(escape(event.label) for event in ray.events) or "none"
                branches = ", ".join(f"{branch:+d}" for branch in sorted(set(ray.branches)))
                charts = ", ".join(escape(chart) for chart in dict.fromkeys(ray.charts))
                reason = escape(termination_reason(ray.fate, ray.escape_radius))
                lines.append(
                    f"<div style='margin-top: 12px; border-top: 1px solid #61636b;'>"
                    f"<b><span style='color:{color.hex};'>■ {escape(ray.name)}</span></b><br>"
                    f"{escape(ray.fate)} / <b>{escape(ray.reliability)}</b><br>"
                    f"<small>END: {reason}<br>r branch: {branches} · chart: {charts}<br>"
                    f"events: {events}</small></div>")
            self.panel.setText("<br>".join(lines))

        def closeEvent(self, event):
            self.view.frame_timer.stop()
            compute.shutdown(wait=False, cancel_futures=True)
            super().closeEvent(event)

    window = MainWindow()
    window.show()
    return app.exec()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Static Kerr physics trajectory viewer")
    parser.add_argument("--spin", type=float, default=.95)
    parser.add_argument("--backend", choices=("auto", "compiled", "reference"), default="auto")
    args = parser.parse_args(argv)
    return launch(args.spin, args.backend)
