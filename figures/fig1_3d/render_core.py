"""ISBI 2027 Figure 1A, enlarged core along the beam (depth to scale): probe on the skin, scalp and muscle 5 mm,
temporal bone 2.9 mm, brain 4.2 cm, voxel 27 mm3 at 5 cm. A half block cut through the beam axis, so the beam lies on
its front face. Output: core_render.png (transparent) + core_render.json (pixel anchors for the labels).
Run (Blender 4.2.23, from figures/fig1_3d/): blender -b --factory-startup -P render_core.py -- [samples] [width_px]"""
import sys, json
import numpy as np
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

D = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/"   # figures/fig1_3d/ of this repository
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SAMPLES = int(argv[0]) if argv else 96; WPX = int(argv[1]) if len(argv) > 1 else 900; HPX = int(round(WPX * 2.0))
U = 0.01
SCALP, BONE, DEPTH, W, T = 5.0, 2.9, 50.0, 22.0, 20.0         # mm; voxel centre at DEPTH
ZB = -(DEPTH + 5.0)                                             # bottom of the core
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
sc = bpy.context.scene

def principled(name, rgb, rough=0.5, alpha=1.0, emit=None, emit_s=1.0, spec=0.3, noise=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; p = nt.nodes["Principled BSDF"]
    p.inputs["Roughness"].default_value = rough; p.inputs["Specular IOR Level"].default_value = spec
    if noise > 0:                                               # subtle texture so the layers do not look like plastic
        nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 260.0; nz.inputs["Detail"].default_value = 6
        ramp = nt.nodes.new("ShaderNodeValToRGB"); ramp.color_ramp.elements[0].color = (*(np.array(rgb) * (1 - noise)), 1)
        ramp.color_ramp.elements[1].color = (*rgb, 1); nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"]); nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    else:
        p.inputs["Base Color"].default_value = (*rgb, 1)
    if emit is not None: p.inputs["Emission Color"].default_value = (*emit, 1); p.inputs["Emission Strength"].default_value = emit_s
    if alpha < 1: p.inputs["Alpha"].default_value = alpha; m.blend_method = "BLEND"
    return m

def box(name, x0, x1, y0, y1, z0, z1, m, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1); ob = bpy.context.active_object; ob.name = name
    ob.scale = ((x1 - x0) * U, (y1 - y0) * U, (z1 - z0) * U); ob.location = ((x0 + x1) / 2 * U, (y0 + y1) / 2 * U, (z0 + z1) / 2 * U)
    if bevel > 0: bv = ob.modifiers.new("bevel", "BEVEL"); bv.width = bevel * U; bv.segments = 4
    ob.data.materials.append(m); return ob

C_SKIN = (0.86, 0.58, 0.44); C_BONE = (0.95, 0.92, 0.82); C_BRAIN = (0.72, 0.66, 0.68); C_PROBE = (0.10, 0.11, 0.13); C_BEAM = (0.16, 0.45, 0.80)
box("scalp", -W / 2, W / 2, 0, T, -SCALP, 0, principled("skin", C_SKIN, 0.5, noise=0.08))
box("bone", -W / 2, W / 2, 0, T, -SCALP - BONE, -SCALP, principled("bone", C_BONE, 0.6, noise=0.12))
box("brain", -W / 2, W / 2, 0, T, ZB, -SCALP - BONE, principled("brain", C_BRAIN, 0.55, noise=0.10))
box("probe", -13, 13, -3, T - 3, 1.2, 10.0, principled("probe", C_PROBE, 0.35, spec=0.5), bevel=1.2)
box("lens", -11, 11, -1.5, T - 4.5, 0.05, 1.25, principled("lens", (0.55, 0.57, 0.60), 0.4))
# beam on the front face (y = 0): 20-mm aperture narrowing to the voxel
tri = np.array([[-10, -0.15, 0], [10, -0.15, 0], [0.9, -0.15, -DEPTH], [-0.9, -0.15, -DEPTH]])
me = bpy.data.meshes.new("beam"); me.from_pydata((tri * U).tolist(), [], [[0, 1, 2, 3]]); me.update()
bm = bpy.data.objects.new("beam", me); sc.collection.objects.link(bm); bm.data.materials.append(principled("beam", C_BEAM, 0.3, alpha=0.55, emit=C_BEAM, emit_s=0.7))
box("voxel", -1.5, 1.5, -1.6, 1.4, -DEPTH - 1.5, -DEPTH + 1.5, principled("voxel", (1.0, 0.78, 0.10), 0.3, emit=(1.0, 0.70, 0.05), emit_s=2.0))

tgt = np.array([0.0, T / 2, -24.0])
cdir = np.array([0.55, -0.80, 0.30]); cdir /= np.linalg.norm(cdir)
cd = bpy.data.cameras.new("cam"); cd.type = "ORTHO"; cd.ortho_scale = 76 * U; cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam)
cam.location = Vector(((tgt + cdir * 400) * U).tolist())
cam.rotation_mode = "QUATERNION"; cam.rotation_quaternion = (Vector((tgt * U).tolist()) - cam.location).to_track_quat("-Z", "Y")
sc.camera = cam
def area(name, loc, power, size):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = power; ld.size = size
    lo = bpy.data.objects.new(name, ld); sc.collection.objects.link(lo); lo.location = Vector(loc)
    lo.rotation_mode = "QUATERNION"; lo.rotation_quaternion = (Vector((tgt * U).tolist()) - lo.location).to_track_quat("-Z", "Y")
area("key", ((tgt + np.array([-150, -260, 320])) * U).tolist(), 75, 2.0)
area("fill", ((tgt + np.array([320, -120, 40])) * U).tolist(), 22, 2.5)
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1); w.node_tree.nodes["Background"].inputs[1].default_value = 0.45

sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
sc.render.film_transparent = True; sc.view_settings.view_transform = "Standard"
sc.render.resolution_x = WPX; sc.render.resolution_y = HPX; sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"; sc.render.filepath = D + "core_render.png"
bpy.ops.render.render(write_still=True)
def px(p):
    co = world_to_camera_view(sc, cam, Vector((np.asarray(p, float) * U).tolist())); return [co.x * WPX, (1 - co.y) * HPX]
xr = W / 2                                                         # anchors on the right edge of the front face
anc = {"probe": px([13, -3, 6]), "scalp": px([xr, 0, -SCALP / 2]), "bone": px([xr, 0, -SCALP - BONE / 2]), "brain": px([xr, 0, -28]),
       "voxel": px([1.5, -1.6, -DEPTH]), "top_left": px([-W / 2, 0, 10]), "bottom_left": px([-W / 2, 0, ZB]), "skin_left": px([-W / 2, 0, 0]),
       "probe_top": px([0, 7, 10])}
json.dump({"size_px": [WPX, HPX], "anchors": anc}, open(D + "core_render.json", "w"), indent=1)
print("render done", anc)
