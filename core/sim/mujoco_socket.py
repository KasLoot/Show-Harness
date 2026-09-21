"""Generate an open round socket from compact scene geometry metadata.

MuJoCo mesh collision uses convex hulls. Separate convex annular wedges preserve
the bore, while noncolliding continuous surfaces hide the wedge boundaries.
"""
import math
import xml.etree.ElementTree as ET


def _numbers(values) -> str:
    return " ".join(f"{float(value):.16g}" for value in values)


def add_round_socket(scene: ET.Element) -> None:
    """Add convex walls and blue/black visual surfaces to the plug scene in place."""
    socket = scene.find(".//body[@name='socket']")
    asset = scene.find("asset")
    if socket is None or asset is None:
        raise ValueError("Round socket requires a socket body and scene asset section")
    entry = socket.find("site[@name='socket_entry']")
    seat = socket.find("site[@name='socket_seat']")
    floor = socket.find("geom[@name='socket_floor']")
    if entry is None or seat is None or floor is None:
        raise ValueError("Round socket requires entry/seat sites and a box floor")
    radius = float(entry.get("size").split()[0])
    lower = float(seat.get("pos").split()[2])
    upper = float(entry.get("pos").split()[2])
    half_x, half_y, half_z = map(float, floor.get("size").split())
    floor_center_z = float(floor.get("pos").split()[2])
    if (entry.get("type") != "cylinder" or seat.get("type") != "cylinder"
            or floor.get("type") != "box"
            or float(seat.get("size").split()[0]) != radius
            or not 0 < radius < min(half_x, half_y)
            or not lower < upper
            or not math.isclose(floor_center_z + half_z, lower, abs_tol=1e-9)):
        raise ValueError("Inconsistent round socket radius, housing extents, or floor depth")

    # Include all rectangle corners so each wall section is convex. At 96 angular
    # intervals the polygonal bore differs from its nominal radius by under 7 um.
    corner = math.atan2(half_y, half_x)
    angles = sorted({*(math.tau * i / 96 for i in range(97)),
                     corner, math.pi - corner, math.pi + corner, math.tau - corner})
    solid_faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for i in range(4):
        j = (i + 1) % 4
        solid_faces.extend(((i, j, j + 4), (i, j + 4, i + 4)))
    visual = {"exterior": ([], [], []), "interior": ([], [], [])}

    def add_quad(surface, points, normals):
        vertices, vertex_normals, faces = visual[surface]
        start = len(vertices)
        vertices.extend(points)
        vertex_normals.extend(normals)
        faces.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))

    def point(angle, distance=None):
        c, s = math.cos(angle), math.sin(angle)
        if distance is None:
            distance = min(half_x / max(abs(c), 1e-12), half_y / max(abs(s), 1e-12))
        return [distance * c, distance * s]

    for index, (a, b) in enumerate(zip(angles[:-1], angles[1:])):
        inner_a, outer_a, outer_b, inner_b = point(a, radius), point(a), point(b), point(b, radius)
        polygon = [inner_a, outer_a, outer_b, inner_b]
        vertices = [[*p, z] for z in (lower, upper) for p in polygon]
        name = f"socket_wall_{index:03d}"
        ET.SubElement(asset, "mesh", name=name,
                      vertex=_numbers(v for p in vertices for v in p),
                      face=" ".join(str(v) for face in solid_faces for v in face))
        ET.SubElement(socket, "geom", name=name, type="mesh", mesh=name,
                      rgba="0.08 0.35 0.9 0", friction="0.4 0.005 0.0001", group="3")

        # Only the exposed top and housing sides are rendered, not the internal
        # radial wedge faces. The side shell covers the black floor's outer sides.
        add_quad("exterior", [[*p, upper] for p in polygon], [[0, 0, 1]] * 4)
        dx, dy = outer_b[0] - outer_a[0], outer_b[1] - outer_a[1]
        length = math.hypot(dx, dy)
        normal = [dy / length, -dx / length, 0]
        # A one-micrometre visual offset avoids z-fighting with the physical floor.
        edge_a = [outer_a[i] + 1e-6 * normal[i] for i in range(2)]
        edge_b = [outer_b[i] + 1e-6 * normal[i] for i in range(2)]
        bottom = floor_center_z - half_z
        add_quad("exterior", [[*edge_a, bottom], [*edge_b, bottom],
                              [*edge_b, upper], [*edge_a, upper]], [normal] * 4)
        normal_a = [-math.cos(a), -math.sin(a), 0]
        normal_b = [-math.cos(b), -math.sin(b), 0]
        add_quad("interior", [[*inner_a, lower], [*inner_a, upper],
                              [*inner_b, upper], [*inner_b, lower]],
                 [normal_a, normal_a, normal_b, normal_b])

    for surface, (vertices, normals, faces) in visual.items():
        name = f"socket_{surface}_visual"
        ET.SubElement(asset, "mesh", name=name,
                      vertex=_numbers(v for p in vertices for v in p),
                      normal=_numbers(v for p in normals for v in p),
                      face=" ".join(str(v) for face in faces for v in face))
        color = "0.08 0.35 0.9 1" if surface == "exterior" else "0.004 0.005 0.006 1"
        ET.SubElement(socket, "geom", name=name, type="mesh", mesh=name,
                      rgba=color, contype="0", conaffinity="0", mass="0", group="2")
