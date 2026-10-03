"""Headless viewer geometry and optional EGL framebuffer validation.

The EGL path tests real ModernGL programs with a software OpenGL driver;
it cannot verify the Qt window, Windows graphics drivers or interactive FPS.
"""

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from kerr_viewer.app import _solve_static_scene
from kerr_viewer.camera import FreeCamera
from kerr_viewer.scene import horizon_mesh, singularity_marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egl", action="store_true", help="create an EGL ModernGL framebuffer")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    records = []
    ctx = None
    if args.egl:
        import moderngl
        from kerr_viewer.renderer import SceneRenderer
        ctx = moderngl.create_standalone_context(require=330, backend="egl")
    for spin in (0., .7, 1., 1.05, 1.2):
        scene = _solve_static_scene(spin, "auto")
        mesh = horizon_mesh(scene.horizon, spin)
        marker = singularity_marker(scene.singularity)
        assert (len(mesh) > 0) == (abs(spin) <= 1.)
        assert (len(marker) == 1) == (spin == 0.)
        assert len(scene.rays) == 3 and all(len(ray.positions) > 2 for ray in scene.rays)
        record = {"spin": spin, "regime": scene.horizon.regime.value,
                  "horizon_triangles": len(mesh)//3,
                  "singularity": scene.singularity.kind,
                  "rays": [{"fate": ray.fate, "reliability": ray.reliability,
                            "charts": sorted(set(ray.charts)), "branches": sorted(set(ray.branches))}
                           for ray in scene.rays]}
        if ctx is not None:
            renderer = SceneRenderer(ctx, scene)
            texture = ctx.texture((640, 480), 4)
            depth = ctx.depth_renderbuffer((640, 480))
            fbo = ctx.framebuffer(color_attachments=[texture], depth_attachment=depth)
            renderer.draw(scene, FreeCamera.projection(640/480) @ FreeCamera().view(), fbo)
            pixels = np.frombuffer(fbo.read(components=4, alignment=1), dtype=np.uint8).reshape(480, 640, 4)
            changed = np.any(pixels[:, :, :3] != pixels[0, 0, :3], axis=2)
            record["non_background_pixels"] = int(np.count_nonzero(changed))
            record["framebuffer_alpha_opaque"] = bool(np.all(pixels[:, :, 3] == 255))
            assert record["non_background_pixels"] > 500
            assert record["framebuffer_alpha_opaque"]
            renderer.draw(scene, FreeCamera.projection(640/480) @ FreeCamera().view(),
                          fbo, show_markers=False)
            clean = np.frombuffer(fbo.read(components=4, alignment=1),
                                  dtype=np.uint8).reshape(480, 640, 4)
            record["debug_overlay_changed_pixels"] = int(np.count_nonzero(
                np.any(pixels[:, :, :3] != clean[:, :, :3], axis=2)))
            assert record["debug_overlay_changed_pixels"] > 0
            fbo.release(); depth.release(); texture.release(); renderer.release()
        records.append(record)
        print(record)
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "context": "EGL software OpenGL" if args.egl else "CPU geometry only",
              "cases": records}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
