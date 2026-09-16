"""Convert the pinned RoboLab scene to MJCF without importing Isaac Sim.

Meshes, UVs, textures, object transforms and masses come from the authored USD.
The concave bowl is decomposed offline; its collision geometry remains hollow.
Generated files are cached under models/mujoco (not committed).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "models/mujoco/robolab"
OUTPUT = ROOT / "models/mujoco/rubiks_cube_bowl"


def numbers(value) -> str:
    return " ".join(f"{x:.10g}" for x in np.asarray(value).ravel())


def source_constants() -> dict:
    """Read data constants without executing the Isaac-only configuration module."""
    wanted = {"FRONT_CAM_R", "FRONT_CAM_POS", "FRONT_CAM_K", "FRONT_CAM_W", "FRONT_CAM_H",
              "WRIST_CAM_POS", "WRIST_CAM_RESOLUTION", "FRANKA_HOME_QPOS"}
    tree = ast.parse((ROOT / "core/sim/robolab_franka.py").read_text())
    result = {}
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id in wanted:
            value = node.value
            if target.id == "FRANKA_HOME_QPOS":
                result[target.id] = {ast.literal_eval(k): ast.literal_eval(v)
                                     for k, v in zip(value.keys, value.values)
                                     if not isinstance(v, ast.Name)}
            else:
                result[target.id] = ast.literal_eval(value)
        elif isinstance(target, ast.Tuple):
            values = ast.literal_eval(node.value)
            for key, value in zip(target.elts, values):
                if isinstance(key, ast.Name) and key.id in wanted:
                    result[key.id] = value
    if set(result) != wanted:
        raise ValueError("The source Franka config changed; review its data constants.")
    return result


def _transform(matrix) -> dict:
    a = np.asarray(matrix)
    scale = np.linalg.norm(a[:3, :3], axis=1)
    quat = Rotation.from_matrix((a[:3, :3] / scale[:, None]).T).as_quat()
    return {"pos": numbers(a[3, :3]), "quat": numbers(np.r_[quat[3], quat[:3]])}


def _triangles(counts, indices):
    faces, corners, face_ids = [], [], []
    offset = 0
    for face_id, count in enumerate(counts):
        for j in range(1, int(count) - 1):
            corner = [offset, offset + j, offset + j + 1]
            corners.append(corner)
            faces.append(np.asarray(indices)[corner])
            face_ids.append(face_id)
        offset += int(count)
    return np.asarray(faces), np.asarray(corners), np.asarray(face_ids)


def _write_obj(path, vertices, faces, uv=None):
    # Compact subsets (the source fixture has 740k per-face vertices).
    unique, inverse = np.unique(faces.ravel(), return_inverse=True)
    vv = vertices[unique]
    ff = inverse.reshape(-1, 3) + 1
    with path.open("w") as f:
        for v in vv:
            f.write("v " + numbers(v) + "\n")
        if uv is None:
            for face in ff:
                f.write("f " + " ".join(map(str, face)) + "\n")
        else:
            for point in uv.reshape(-1, 2):
                f.write("vt " + numbers(point) + "\n")
            for i, face in enumerate(ff):
                f.write("f " + " ".join(f"{v}/{3*i+j+1}" for j, v in enumerate(face)) + "\n")


class Converter:
    def __init__(self, output: Path):
        self.output = output
        self.meshes = output / "meshes"
        self.meshes.mkdir(parents=True, exist_ok=True)
        self.root = ET.Element("mujoco", model="RoboLab_RubiksCubeTask")
        self.asset = ET.SubElement(self.root, "asset")
        self.world = ET.SubElement(self.root, "worldbody")
        self.materials = {}
        self.report = {"source_revision": (SOURCE / "REVISION").read_text().strip(),
                       "objects": {}, "approximations": [
                           "Omniverse MDL shaders use diffuse-color fallbacks; object textures are preserved.",
                           "MuJoCo contact/friction solvers differ from PhysX.",
                           "Concave bowl uses deterministic CoACD convex pieces; inertias are mesh-derived."]}
        self.index = 0

    def material(self, prim) -> str:
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        key = str(mat.GetPath()) if mat else "unbound"
        if key in self.materials:
            return self.materials[key]
        name = f"material_{len(self.materials)}"
        self.materials[key] = name
        color, texture = (0.65, 0.65, 0.65), None
        if mat:
            for shader in Usd.PrimRange(mat.GetPrim()):
                if shader.IsA(UsdShade.Shader):
                    for inp in UsdShade.Shader(shader).GetInputs():
                        label, value = str(inp.GetBaseName()), inp.Get()
                        if label in ("diffuse_color_constant", "diffuseColor", "diffuse_tint"):
                            color = tuple(value)
                        elif label in ("diffuse_texture", "file") and value:
                            texture = value.resolvedPath
        if "Oak" in key:
            color = (0.64, 0.46, 0.27)
        elif "RustedMetal" in key:
            color = (0.26, 0.25, 0.24)
        attributes = {"name": name, "rgba": numbers([*color, 1]), "specular": "0.15"}
        if texture:
            tex_name = name + "_texture"
            ET.SubElement(self.asset, "texture", name=tex_name, type="2d", file=texture)
            attributes["texture"] = tex_name
        ET.SubElement(self.asset, "material", **attributes)
        return name

    def mesh(self, body, prim, relative, dynamic: bool, name: str):
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ np.asarray(relative))[:, :3]
        faces, corners, face_ids = _triangles(mesh.GetFaceVertexCountsAttr().Get(),
                                             mesh.GetFaceVertexIndicesAttr().Get())
        proxy = UsdGeom.Imageable(prim).ComputePurpose() == "proxy"
        uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        tex = None
        if uv and uv.HasValue():
            coords = np.asarray(uv.ComputeFlattened())
            tex = coords[corners] if uv.GetInterpolation() == "faceVarying" else coords[faces]
        subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        if not proxy:
            parts = [(prim, np.ones(len(faces), dtype=bool))] if not subsets else [
                (part.GetPrim(), np.isin(face_ids, part.GetIndicesAttr().Get())) for part in subsets]
            for material_prim, mask in parts:
                if not mask.any():
                    continue
                self.index += 1
                asset_name = f"visual_{name}_{self.index}"
                path = self.meshes / f"{asset_name}.obj"
                _write_obj(path, vertices, faces[mask], None if tex is None else tex[mask])
                ET.SubElement(self.asset, "mesh", name=asset_name, file=str(path.resolve()))
                ET.SubElement(body, "geom", type="mesh", mesh=asset_name, group="2", mass="0",
                              contype="0", conaffinity="0", material=self.material(material_prim))
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            return vertices
        pieces = [(vertices, faces)]
        if name == "bowl":
            import coacd
            import trimesh

            cache = self.output / "bowl_collision.npz"
            if cache.exists():
                data = np.load(cache)
                pieces = [(data[f"v{i}"], data[f"f{i}"]) for i in range(int(data["count"]))]
            else:
                print("[convert] Decomposing the original hollow bowl...", flush=True)
                coacd.set_log_level("warn")
                clean = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                pieces = coacd.run_coacd(coacd.Mesh(clean.vertices, clean.faces), threshold=0.015,
                                        max_convex_hull=64, preprocess_resolution=80,
                                        mcts_iterations=40, mcts_nodes=10, seed=0)
                saved = {"count": np.array(len(pieces))}
                for i, (v, f) in enumerate(pieces):
                    saved.update({f"v{i}": v, f"f{i}": f})
                np.savez_compressed(cache, **saved)
            self.report["bowl_collision_parts"] = len(pieces)
        for index, (verts, triangles) in enumerate(pieces):
            self.index += 1
            asset_name = f"collision_{name}_{self.index}"
            path = self.meshes / f"{asset_name}.obj"
            _write_obj(path, verts, triangles)
            ET.SubElement(self.asset, "mesh", name=asset_name, file=str(path.resolve()))
            ET.SubElement(body, "geom", name=asset_name, type="mesh", mesh=asset_name,
                          group="3", rgba="0.4 0.4 0.4 0", friction="1 0.005 0.0001",
                          condim="4", solref="0.004 1", solimp="0.95 0.99 0.001")
        return vertices

    def convert(self):
        scene = SOURCE / "assets/scenes/rubiks_cube_bowl.usda"
        stage = Usd.Stage.Open(str(scene.resolve()))
        transforms = UsdGeom.XformCache()
        for name in ("table", "bowl", "rubiks_cube", "franka_table"):
            prim = stage.GetPrimAtPath(f"/World/{name}")
            world_pose = transforms.GetLocalToWorldTransform(prim)
            dynamic = name in ("table", "bowl", "rubiks_cube")
            body = ET.SubElement(self.world, "body", name=name, **_transform(world_pose))
            if dynamic:
                ET.SubElement(body, "freejoint", name=name + "_joint")
            all_vertices = []
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                relative = transforms.GetLocalToWorldTransform(child) * world_pose.GetInverse()
                if child.IsA(UsdGeom.Mesh):
                    all_vertices.append(self.mesh(body, child, relative, dynamic, name))
                elif child.IsA(UsdGeom.Cube):
                    matrix = np.asarray(relative)
                    scale = np.linalg.norm(matrix[:3, :3], axis=1)
                    size = float(UsdGeom.Cube(child).GetSizeAttr().Get()) * scale / 2
                    ET.SubElement(body, "geom", name=f"{name}_{child.GetName()}", type="box",
                                  size=numbers(size), **_transform(relative),
                                  material=self.material(child), friction="1 0.005 0.0001")
                elif child.IsA(UsdGeom.Cylinder):
                    cylinder = UsdGeom.Cylinder(child)
                    ET.SubElement(body, "geom", name=f"{name}_{child.GetName()}", type="cylinder",
                                  size=numbers([cylinder.GetRadiusAttr().Get(), cylinder.GetHeightAttr().Get()/2]),
                                  **_transform(relative), material=self.material(child))
            mass = prim.GetAttribute("physics:mass").Get()
            if mass:
                # Reweight collision geoms by volume; preserve the authored total mass.
                geoms = [g for g in body.findall("geom") if g.get("group") == "3"]
                import trimesh
                volumes = []
                for geom in geoms:
                    entry = self.asset.find(f"mesh[@name='{geom.get('mesh')}']")
                    mesh = trimesh.load_mesh(entry.get("file"))
                    volumes.append(max(abs(mesh.convex_hull.volume), 1e-12))
                for geom, volume in zip(geoms, volumes):
                    geom.set("mass", str(float(mass) * volume / sum(volumes)))
            entry = {"pose": _transform(world_pose), "dynamic": dynamic, "authored_mass_kg": mass}
            if all_vertices:
                vertices = np.concatenate(all_vertices)
                entry["local_bounds"] = [vertices.min(0).tolist(), vertices.max(0).tolist()]
                if name in ("bowl", "rubiks_cube"):
                    # Match RoboLab hull_check.py (float32 hull-vertex centroid,
                    # and open-top planes with normal.z < 0.7), without torch.
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(vertices)
                    entry["hull_centroid"] = vertices[hull.vertices].astype(np.float32).mean(0).tolist()
                    if name == "bowl":
                        planes = hull.equations.astype(np.float32)
                        entry["open_top_planes"] = planes[planes[:, 2] < 0.7].tolist()
            self.report["objects"][name] = entry
        ground = stage.GetPrimAtPath("/World/GroundPlane")
        ground_visible = UsdGeom.Imageable(ground).ComputeVisibility() != "invisible"
        ET.SubElement(self.world, "geom", name="ground", type="plane", size="2 2 0.1",
                      pos=numbers(transforms.GetLocalToWorldTransform(ground).ExtractTranslation()),
                      rgba=numbers([0.35, 0.35, 0.35, float(ground_visible)]))
        self.report["ground_visible"] = ground_visible
        ET.indent(self.root)
        ET.ElementTree(self.root).write(self.output / "scene.xml", encoding="unicode")
        self.report["source_manifest_sha256"] = hashlib.sha256((SOURCE / "manifest.json").read_bytes()).hexdigest()
        self.report["franka"] = source_constants()
        self.convert_robot()
        (self.output / "provenance.json").write_text(json.dumps(self.report, indent=2) + "\n")
        print(f"[convert] Scene ready: {self.output}", flush=True)

    def convert_robot(self):
        from scripts.trajectory.real2sim.robolab.make_short_finger_asset import (
            BRACKET_COLOR, TIP_COLOR, bracket_arrays, mesh_arrays, read_binary_stl,
        )

        original = ROOT / "models/mujoco/franka_emika_panda/panda.xml"
        robot = ET.parse(original).getroot()
        robot.find("compiler").set("meshdir", str(original.parent / "assets"))
        robot.remove(robot.find("keyframe"))
        assets = robot.find("asset")
        ET.SubElement(assets, "material", name="yellow_tip", rgba=numbers([*TIP_COLOR, 1]))
        ET.SubElement(assets, "material", name="finger_bracket", rgba=numbers([*BRACKET_COLOR, 1]))
        triangles, _ = read_binary_stl(ROOT / "assets/panda_short_finger.stl")
        for side, mirror in (("left", False), ("right", True)):
            finger = robot.find(f".//body[@name='{side}_finger']")
            # Isaac's two finger link frames share the hand's orientation.
            # Menagerie rotates the right link 180 degrees; remove that rotation
            # and reverse its slider axis to retain the same world-space motion.
            finger.attrib.pop("quat", None)
            # The original USD declares density 1000 for the finger assembly.
            finger.remove(finger.find("inertial"))
            finger.find("joint").set("axis", "0 -1 0" if mirror else "0 1 0")
            for geom in list(finger.findall("geom")):
                finger.remove(geom)
            pts, faces, _ = mesh_arrays(triangles, mirror)
            bracket_pts, bracket_faces, _ = bracket_arrays(mirror)
            for label, vv, ff, material in (("tip", pts, faces, "yellow_tip"),
                                           ("bracket", bracket_pts, bracket_faces, "finger_bracket")):
                name = f"{side}_{label}"
                path = self.meshes / f"{name}.obj"
                _write_obj(path, vv, ff)
                ET.SubElement(assets, "mesh", name=name, file=str(path.resolve()))
                ET.SubElement(finger, "geom", mesh=name, type="mesh", material=material,
                              contype="0", conaffinity="0", group="2", mass="0")
            # The original Isaac asset uses one convex hull over bracket and tip.
            combined = np.vstack([pts, bracket_pts])
            combined_faces = np.vstack([faces, bracket_faces + len(pts)])
            name = f"{side}_short_finger_collision"
            path = self.meshes / f"{name}.obj"
            _write_obj(path, combined, combined_faces)
            ET.SubElement(assets, "mesh", name=name, file=str(path.resolve()))
            ET.SubElement(finger, "geom", name=name, type="mesh", mesh=name, group="3",
                          contype="2", conaffinity="1", friction="1 0.005 0.0001",
                          density="1000", condim="4", solref="0.004 1", solimp="0.95 0.99 0.001")
        for body in robot.findall(".//body"):
            body.set("gravcomp", "1")
        default = robot.find(".//default[@class='collision']/geom")
        default.set("contype", "2")
        default.set("conaffinity", "1")
        # Match the actuator gains and effort bounds in FrankaPandaCfg.
        for actuator in robot.find("actuator"):
            if actuator.get("joint", "").startswith("joint"):
                actuator.set("gainprm", "400")
                actuator.set("biasprm", "0 -400 -80")
        actuators = robot.find("actuator")
        actuators.remove(actuators.find("general[@name='actuator8']"))
        for i in (1, 2):
            ET.SubElement(actuators, "position", name=f"finger_actuator{i}",
                          joint=f"finger_joint{i}", kp="2000", kv="100",
                          ctrlrange="0 0.04", forcerange="-200 200")
        robot.remove(robot.find("tendon"))
        robot.remove(robot.find("equality"))
        hand = robot.find(".//body[@name='hand']")
        ET.SubElement(hand, "site", name="eef", pos="0 0 0", size="0.002", rgba="0 0 0 0")
        ET.indent(robot)
        ET.ElementTree(robot).write(self.output / "panda.xml", encoding="unicode")
        self.report["robot_sources"] = {
            "arm": "MuJoCo Menagerie Panda (Franka URDF inertias and kinematics)",
            "fingers": "assets/panda_short_finger.stl + original make_short_finger_asset.py transforms",
            "fingers_sha256": hashlib.sha256((ROOT / "assets/panda_short_finger.stl").read_bytes()).hexdigest(),
            "finger_collision": "Convex hull of original bracket and yellow tip, as in Isaac asset",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    Converter(args.output).convert()


if __name__ == "__main__":
    main()
