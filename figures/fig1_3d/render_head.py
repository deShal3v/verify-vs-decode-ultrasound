"""ISBI 2027 Figure 1A: Blender (Cycles, CPU) render of the S1 temple cut-away from head_S1.npz (prep_head.py).
Writes head_render.png (transparent film) and head_render.json (pixel positions of the label anchors).
Run (Blender 4.2.23, from figures/fig1_3d/): blender -b --factory-startup -P render_head.py -- [samples] [width_px]"""
import sys, json, math
import numpy as np
import bpy, bmesh
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view

D = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/"   # figures/fig1_3d/ of this repository
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SAMPLES = int(argv[0]) if len(argv) > 0 else 96
WPX = int(argv[1]) if len(argv) > 1 else 1600
HPX = int(round(WPX * 0.60))
U = 0.01                                                        # 1 Blender unit = 100 mm
Z = np.load(D + "head_S1.npz")
pf, vf, b, n = (Z[k].astype(float) for k in ("probe", "voxel", "beam_dir", "cut_n"))
side = np.cross(n, b); side /= np.linalg.norm(side)

for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
sc = bpy.context.scene

def mat(name, rgb, rough=0.55, alpha=1.0, emit=None, sss=0.0, spec=0.3):
    m = bpy.data.materials.new(name); m.use_nodes = True; p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*rgb, 1); p.inputs["Roughness"].default_value = rough
    p.inputs["Specular IOR Level"].default_value = spec
    if sss > 0:
        p.inputs["Subsurface Weight"].default_value = sss; p.inputs["Subsurface Radius"].default_value = (1.0, 0.5, 0.35)
        p.inputs["Subsurface Scale"].default_value = 0.004
    if emit is not None:
        p.inputs["Emission Color"].default_value = (*emit, 1); p.inputs["Emission Strength"].default_value = 1.0
    if alpha < 1:
        p.inputs["Alpha"].default_value = alpha; m.blend_method = "BLEND"
    return m

def mesh_obj(name, V, F, material, smooth=True):
    me = bpy.data.meshes.new(name); me.from_pydata((V * U).tolist(), [], F.tolist()); me.update()
    if smooth: me.shade_smooth()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob.data.materials.append(material)
    return ob

SKIN = (0.80, 0.60, 0.50); BONE = (0.93, 0.89, 0.78); PROBE = (0.10, 0.11, 0.13); BEAM = (0.16, 0.45, 0.80)
sk = mesh_obj("scalp", Z["scalp_v"], Z["scalp_f"], mat("skin_tmp", SKIN))
wa = sk.data.attributes.new("win", "FLOAT", "POINT"); wa.data.foreach_set("value", Z["scalp_win"].astype(np.float32))
ms = bpy.data.materials.new("skin"); ms.use_nodes = True; nt = ms.node_tree; ps = nt.nodes["Principled BSDF"]
at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "win"; mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"
mx.inputs[6].default_value = (*SKIN, 1); mx.inputs[7].default_value = (0.55, 0.72, 0.90, 1)
nt.links.new(at.outputs["Fac"], mx.inputs[0]); nt.links.new(mx.outputs[2], ps.inputs["Base Color"])
ps.inputs["Roughness"].default_value = 0.5; ps.inputs["Specular IOR Level"].default_value = 0.3
sk.data.materials.clear(); sk.data.materials.append(ms)
bn = mesh_obj("bone", Z["bone_v"], Z["bone_f"], mat("bone", BONE, 0.6))
# brain coloured by T1: grey matter (~70) pink-grey, white matter (~110) cream
br = mesh_obj("brain", Z["brain_v"], Z["brain_f"], mat("brain_tmp", (0.8, 0.7, 0.7)))
t = np.clip((Z["brain_t1"] - 62.0) / (108.0 - 62.0), 0, 1)
GM = np.array([0.74, 0.55, 0.55]); WM = np.array([0.96, 0.93, 0.88]); col = GM[None] * (1 - t[:, None]) + WM[None] * t[:, None]
ca = br.data.color_attributes.new("t1", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.concatenate([col, np.ones((len(col), 1))], 1).ravel().astype(np.float32))
mb = bpy.data.materials.new("brain"); mb.use_nodes = True; nt = mb.node_tree; pb = nt.nodes["Principled BSDF"]
attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "t1"; nt.links.new(attr.outputs["Color"], pb.inputs["Base Color"])
pb.inputs["Roughness"].default_value = 0.55; pb.inputs["Specular IOR Level"].default_value = 0.25
br.data.materials.clear(); br.data.materials.append(mb)

for ob in (sk, bn):
    sm = ob.modifiers.new('smooth', 'CORRECTIVE_SMOOTH'); sm.iterations = 6; sm.factor = 0.6

# probe: bevelled box, face on the scalp, 20 mm aperture across the beam's side axis
rot = Matrix((side.tolist(), n.tolist(), (-b).tolist())).transposed()     # local x = side, y = up, z = outward
bpy.ops.mesh.primitive_cube_add(size=1); pr = bpy.context.active_object; pr.name = "probe"
pr.matrix_world = Matrix.Translation(Vector((pf - b * 7.5) * U)) @ rot.to_4x4() @ Matrix.Diagonal((22 * U, 30 * U, 14 * U, 1))
bv = pr.modifiers.new("bevel", "BEVEL"); bv.width = 0.03; bv.segments = 4
pr.data.materials.append(mat("probe", PROBE, 0.35, spec=0.5))
bpy.ops.object.shade_smooth()

# beam: focused cone from the 20-mm aperture to the voxel, translucent
L = float(np.linalg.norm(vf - pf))
bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=10 * U, radius2=1.2 * U, depth=L * U)
cn = bpy.context.active_object; cn.name = "beam"
zb = Vector(b.tolist()); q = Vector((0, 0, 1)).rotation_difference(zb)
cn.rotation_mode = "QUATERNION"; cn.rotation_quaternion = q; cn.location = Vector(((pf + vf) / 2 * U).tolist())
mbeam = mat("beam", BEAM, 0.3, alpha=0.38, emit=BEAM); cn.data.materials.append(mbeam)

# voxel: 3-mm cube (27 mm3, to scale), bright, aligned with the beam frame
bpy.ops.mesh.primitive_cube_add(size=3 * U, location=Vector((vf * U).tolist())); vx = bpy.context.active_object; vx.name = "voxel"
vx.rotation_mode = "QUATERNION"; vx.rotation_quaternion = q
vx.data.materials.append(mat("voxel", (1.0, 0.78, 0.10), 0.3, emit=(1.0, 0.70, 0.05)))

# camera: anterior-lateral-superior of the left temple, looking at the shelf
tgt = (pf * 0.55 + vf * 0.45)
cdir = np.array([-0.78, 0.30, 0.55]); cdir /= np.linalg.norm(cdir)
cam_d = bpy.data.cameras.new("cam"); cam_d.lens = 62; cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam)
cam.location = Vector(((tgt + cdir * 600) * U).tolist())
cam.rotation_mode = "QUATERNION"; cam.rotation_quaternion = (Vector((tgt * U).tolist()) - cam.location).to_track_quat("-Z", "Y")
sc.camera = cam

# light: key above-front-left, soft fill, white world
def area(name, loc, power, size):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = power; ld.size = size
    lo = bpy.data.objects.new(name, ld); sc.collection.objects.link(lo); lo.location = Vector(loc)
    lo.rotation_mode = "QUATERNION"; lo.rotation_quaternion = (Vector((tgt * U).tolist()) - lo.location).to_track_quat("-Z", "Y")
area("key", ((tgt + np.array([-200, 150, 420])) * U).tolist(), 140, 2.0)
area("fill", ((tgt + np.array([-380, -250, 60])) * U).tolist(), 45, 3.0)
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True; w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.35

sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
sc.render.film_transparent = True; sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"
sc.render.resolution_x = WPX; sc.render.resolution_y = HPX; sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
sc.render.filepath = D + "head_render.png"
bpy.ops.render.render(write_still=True)

# anchors (pixel, origin top-left)
def px(p_mm):
    co = world_to_camera_view(sc, cam, Vector((np.asarray(p_mm) * U).tolist()))
    return [co.x * WPX, (1 - co.y) * HPX]
S_MM = 5.0; B_MM = 2.9; up = n * 0.3
anc = {"probe": px(pf - b * 8.5 + n * 12), "scalp": px(pf + b * S_MM / 2 + up + side * 6), "bone": px(pf + b * (S_MM + B_MM / 2) + up + side * 6),
       "brain": px(pf + b * 26 + up + side * 9), "voxel": px(vf), "beam": px(pf + b * 18 + up), "window": px(Z["scalp_v"][np.argmax(Z["scalp_win"] * Z["scalp_v"][:, 2])])}
json.dump({"size_px": [WPX, HPX], "anchors": anc}, open(D + "head_render.json", "w"), indent=1)
print("render done", anc)
