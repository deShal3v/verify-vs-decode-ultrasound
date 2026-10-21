"""ISBI 2027 Figure 1A, context view: back half of S1's head (skin) cut along a coronal-ish plane through the beam axis,
the real T1 slice on the cut, the probe on the left temporal window, the focused beam and the 5-cm voxel.
Inputs: context_S1.npz, slice_S1.png (prep_head.py). Outputs: context_render.png (transparent) + context_render.json.
Run (Blender 4.2.23, from figures/fig1_3d/): blender -b --factory-startup -P render_context.py -- [samples] [width_px]"""
import sys, json
import numpy as np
import bpy
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view

D = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/"   # figures/fig1_3d/ of this repository
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SAMPLES = int(argv[0]) if argv else 96; WPX = int(argv[1]) if len(argv) > 1 else 1400; HPX = int(round(WPX * 1.0))
U = 0.01
Z = np.load(D + "context_S1.npz")
pf, vf, b, e1, e2, nn = (Z[k].astype(float) for k in ("probe", "voxel", "beam_dir", "e1", "e2", "nn"))
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
sc = bpy.context.scene

def principled(name, rgb, rough=0.5, alpha=1.0, emit=None, emit_s=1.0, spec=0.3):
    m = bpy.data.materials.new(name); m.use_nodes = True; p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*rgb, 1); p.inputs["Roughness"].default_value = rough
    p.inputs["Specular IOR Level"].default_value = spec
    if emit is not None: p.inputs["Emission Color"].default_value = (*emit, 1); p.inputs["Emission Strength"].default_value = emit_s
    if alpha < 1: p.inputs["Alpha"].default_value = alpha; m.blend_method = "BLEND"
    return m

def add_mesh(name, V, F, m, smooth=True):
    me = bpy.data.meshes.new(name); me.from_pydata((V * U).tolist(), [], F.tolist()); me.update()
    if smooth: me.shade_smooth()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob.data.materials.append(m); return ob

SKIN = (0.80, 0.60, 0.50); PROBE = (0.10, 0.11, 0.13); BEAM = (0.16, 0.45, 0.80)
hd = add_mesh("head", Z["head_v"], Z["head_f"], principled("skin", SKIN, 0.5))
sm = hd.modifiers.new("smooth", "CORRECTIVE_SMOOTH"); sm.iterations = 6; sm.factor = 0.6
hd.data.materials[0].use_backface_culling = False

# T1 slice: emission texture with alpha, 0.2 mm in front of the cut
C = Z["slice_corners"].astype(float) + nn[None] * 0.2
me = bpy.data.meshes.new("slice"); me.from_pydata((C * U).tolist(), [], [[0, 1, 2, 3]]); me.update()
uv = me.uv_layers.new();
for li, (u_, v_) in zip(range(4), [(0, 1), (1, 1), (1, 0), (0, 0)]): uv.data[li].uv = (u_, v_)
sl = bpy.data.objects.new("slice", me); sc.collection.objects.link(sl)
ms = bpy.data.materials.new("slice"); ms.use_nodes = True; nt = ms.node_tree; nt.nodes.clear()
tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(D + "slice_S1.png"); tex.interpolation = "Linear"; tex.extension = "CLIP"
em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 1.0
tr = nt.nodes.new("ShaderNodeBsdfTransparent"); mix = nt.nodes.new("ShaderNodeMixShader"); outn = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(tex.outputs["Color"], em.inputs["Color"]); nt.links.new(tex.outputs["Alpha"], mix.inputs[0])
nt.links.new(tr.outputs[0], mix.inputs[1]); nt.links.new(em.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], outn.inputs[0])
ms.blend_method = "BLEND"; sl.data.materials.append(ms)

# probe on the temple: thin along the beam, aperture along e2
rot = Matrix((e2.tolist(), nn.tolist(), (-b).tolist())).transposed()       # local x = e2, y = nn, z = outward
bpy.ops.mesh.primitive_cube_add(size=1); pr = bpy.context.active_object; pr.name = "probe"
pr.matrix_world = Matrix.Translation(Vector(((pf - b * 7.0) * U).tolist())) @ rot.to_4x4() @ Matrix.Diagonal((30 * U, 22 * U, 14 * U, 1))
bv = pr.modifiers.new("bevel", "BEVEL"); bv.width = 0.025; bv.segments = 4
pr.data.materials.append(principled("probe", PROBE, 0.35, spec=0.5)); bpy.ops.object.shade_smooth()

# beam: translucent triangle on the slice, 20-mm aperture to the voxel
off = nn * 0.5
tri = np.array([pf + e2 * 10 + off, pf - e2 * 10 + off, vf + off])
me = bpy.data.meshes.new("beam"); me.from_pydata((tri * U).tolist(), [], [[0, 1, 2]]); me.update()
bm = bpy.data.objects.new("beam", me); sc.collection.objects.link(bm); bm.data.materials.append(principled("beam", BEAM, 0.3, alpha=0.55, emit=BEAM, emit_s=0.8))

# voxel: 3-mm square on the slice (27 mm3, drawn to scale), bright
bpy.ops.mesh.primitive_cube_add(size=3 * U, location=Vector(((vf + nn * 1.6) * U).tolist())); vx = bpy.context.active_object
vx.rotation_mode = "QUATERNION"; vx.rotation_quaternion = rot.to_quaternion()
vx.data.materials.append(principled("voxel", (1.0, 0.78, 0.10), 0.3, emit=(1.0, 0.70, 0.05), emit_s=2.0))

# camera: in front of the cut, a little lateral (left) and above
tgt = pf + e1 * 74 + e2 * 16
cdir = 0.86 * nn - 0.30 * e1 + 0.40 * np.array([0, 0, 1.0]); cdir /= np.linalg.norm(cdir)
cd = bpy.data.cameras.new("cam"); cd.lens = 85; cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam)
cam.location = Vector(((tgt + cdir * 600) * U).tolist())
cam.rotation_mode = "QUATERNION"; cam.rotation_quaternion = (Vector((tgt * U).tolist()) - cam.location).to_track_quat("-Z", "Y")
sc.camera = cam
def area(name, loc, power, size):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = power; ld.size = size
    lo = bpy.data.objects.new(name, ld); sc.collection.objects.link(lo); lo.location = Vector(loc)
    lo.rotation_mode = "QUATERNION"; lo.rotation_quaternion = (Vector((tgt * U).tolist()) - lo.location).to_track_quat("-Z", "Y")
area("key", ((tgt + cdir * 300 + np.array([-250, 0, 350])) * U).tolist(), 160, 2.5)
area("fill", ((tgt + cdir * 300 + np.array([300, 0, -50])) * U).tolist(), 50, 3.0)
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1); w.node_tree.nodes["Background"].inputs[1].default_value = 0.4

sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
sc.render.film_transparent = True; sc.view_settings.view_transform = "Standard"
sc.render.resolution_x = WPX; sc.render.resolution_y = HPX; sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"; sc.render.filepath = D + "context_render.png"
bpy.ops.render.render(write_still=True)
def px(p):
    co = world_to_camera_view(sc, cam, Vector((np.asarray(p) * U).tolist())); return [co.x * WPX, (1 - co.y) * HPX]
anc = {"probe": px(pf - b * 7.0), "voxel": px(vf), "beam_mid": px(pf + b * 25), "zoom_a": px(pf + b * 0 + e2 * 12), "zoom_b": px(vf - e2 * 12)}
json.dump({"size_px": [WPX, HPX], "anchors": anc}, open(D + "context_render.json", "w"), indent=1)
print("render done", anc)
