"""Skeleton parts for the STS3215 Microduck, v0 (fit-test geometry): legs, trunk, neck and head brackets.

Every part is a union of boxes and cylinders minus cuts (holes, servo keep-outs), written in world mm at the zero
joint pose of ``layout.json``. ``fusion_build.py`` turns ``build/parts.json`` into Fusion bodies and STLs, and
``check_rom.py`` checks the STLs for collisions over the joint ranges.

Joint style: the child link is a clevis screwed to both the horn and the idler disc; the parent holds the servo
in a cradle around the back of the case, screwed through the 2 + 2 rear cover holes. Left leg is designed here;
the right leg is its mirror in y.

Run: uv run python cad/sts3215/parts.py
"""

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
LAYOUT = json.loads((HERE / "layout.json").read_text())

# ---- STS3215 geometry, servo frame (mm): output shaft along +Z at (12.5, 0), horn on +Z -------------------------------
SHAFT = np.array([12.5, 0.0])
CASE_X, CASE_Y = 22.7, 12.4          # case half-length / half-width
COVER_Z = 15.9                        # top and bottom covers
HORN_FACE, IDLER_FACE = 18.7, -17.7   # outer faces of the horn and the idler disc
TOP_PAD = ((-22.0, 3.6), 7.0, 17.0)   # raised pad on the horn side: x range, half-width, top z
BOT_PAD = ((-17.4, -4.1), 9.2, -17.7)  # raised pad on the idler side
PLUG = ((-3.6, 0.4), 9.6, -24.0)      # cable plugs on the idler side (pins at x -1.6; plug size assumed)
HORN_HOLES = [(12.5 + sx * 4.9, sy * 4.9) for sx in (-1, 1) for sy in (-1, 1)]
TOP_SCREWS = [(-16.5, 10.3), (-16.5, -10.3)]
BOT_SCREWS = [(-20.3, 10.3), (-20.3, -10.3)]

# ---- parameters -------------------------------------------------------------------------------------------------------
CLR = 0.3          # servo-to-part clearance
WALL = 3.0         # cradle wall
ARM_T = 3.5        # clevis arm thickness
ARM_R = 11.0       # clevis disc radius (keep < 12 to clear the cable plugs)
M2_CLEAR = 2.2     # M2 clearance hole (cradle screws into the case)
HORN_HOLE = 3.2    # M3 clearance: the horn's holes are 2.5 mm in the servo model, the M3 tap-drill size
YAW_WALL = 4.5     # hip yaw link walls (thicker: they take the cap screws)
PILOT = 1.7        # pilot hole for an M2 self-tapping screw
RELIEF_D = 7.0     # centre relief for the horn screw head and the idler hub
ROTOR_R = 13.0     # nothing of the holding link inside this radius beyond the covers
CRADLE_FRONT = -5.0  # cradle side walls reach this far forward (servo x); the shaft is at 12.5
YAW_CLIP_R = 22.0   # inner side of the hip yaw link stays inside this radius (yaw axes are 45.6 apart)
SHIN_TOP_R = 21.0   # shin stays inside this radius of the knee axis above the ankle cradle

# battery tray on the back of the trunk: 2S pack of two 18650 cells lying across (along y), stacked in z
CELL_D, CELL_L = 18.5, 65.5   # bare 18650 (no protection board; the BMS goes outside the tray)
CELL_CLR = 0.4                # cell-to-pocket clearance
POCKET_L = CELL_L + 1.5       # room for the nickel strip and the balance leads at the ends
TRAY_WALL = 2.0
TRAY_PLATE = 3.0              # front plate against the trunk's back face
LIP_T = 2.5                   # hook lip resting on the trunk deck
M2_HEAD = (4.2, 1.8)          # counterbore for an M2 pan head: diameter, depth
BOARD_HOLES = (58.0, 23.0)    # controller mounting pattern on the tray's back (Raspberry Pi Zero size, assumed)
BOSS = (6.0, 5.0)             # board standoff boss: diameter, height
TRAY_LAYOUT = "flat"          # "flat": cells along y, side by side in x; "across": cells along y, stacked in z; "upright": cells along z, side by side in y
TRAY_DZ = -8.0                # pocket top relative to the deck top (negative: below it)
TRAY_BOT_CHAMFER = 5.0        # 45 deg chamfer on the bottom outer edges (thigh clearance in hip yaw + roll)
TRAY_TOP_CHAMFER = 7.0        # 45 deg chamfer on the top back edge (head clearance with the neck tipped back)


def box(lo, hi, R=np.eye(3), T=None):
    """Box from corners lo/hi given in the frame (R, T) -- world if T is None."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    c = (lo + hi) / 2
    if T is not None:
        R, c = T[:3, :3], T[:3, :3] @ c + T[:3, 3]
    return dict(t="box", c=c.tolist(), R=np.asarray(R).tolist(), h=((hi - lo) / 2).tolist())


def cyl(c, axis, r, z0, z1, T=None):
    """Cylinder along ``axis`` through point c, from z0 to z1 along the axis (frame T, or world)."""
    c, axis = np.asarray(c, float), np.asarray(axis, float)
    mid = c + axis * (z0 + z1) / 2
    if T is not None:
        mid, axis = T[:3, :3] @ mid + T[:3, 3], T[:3, :3] @ axis
    return dict(t="cyl", c=mid.tolist(), a=(axis / np.linalg.norm(axis)).tolist(), r=r, h=abs(z1 - z0) / 2)


class Servo:
    def __init__(self, joint):
        self.T = np.array(LAYOUT["servos"][joint]["transform_mm"])

    def p(self, x, y, z):
        """Servo-frame point -> world."""
        return self.T[:3, :3] @ np.array([x, y, z], float) + self.T[:3, 3]

    def box(self, lo, hi):
        return box(lo, hi, T=self.T)

    def cyl(self, x, y, r, z0, z1):
        return cyl((x, y, 0.0), (0, 0, 1), r, z0, z1, T=self.T)

    # -- features ---------------------------------------------------------------------------------------------------
    def keepout(self):
        """Case envelope + clearance; every part touching this servo subtracts it."""
        c = CLR
        (px0, px1), pw, pz = TOP_PAD
        (bx0, bx1), bw, bz = BOT_PAD
        (gx0, gx1), gw, gz = PLUG
        return [self.box((-CASE_X - c, -CASE_Y - c, -COVER_Z - c), (CASE_X + c, CASE_Y + c, COVER_Z + c)),
                self.box((px0 - c, -pw - c, COVER_Z), (px1 + c, pw + c, pz + c)),
                self.box((bx0 - c, -bw - c, bz - c), (bx1 + c, bw + c, -COVER_Z)),
                self.box((gx0 - c, -gw - c, gz), (gx1 + c, gw + c, -COVER_Z))]

    def rotor_keepout(self, r=ROTOR_R, horn=True, idler=True):
        """Space the child's clevis arms turn in; the link holding this servo stays out of it."""
        out = []
        if horn:
            out.append(self.cyl(*SHAFT, r, COVER_Z, HORN_FACE + 20))
        if idler:
            out.append(self.cyl(*SHAFT, r, IDLER_FACE - 20, -COVER_Z))
        return out

    def cradle(self, top=True):
        """Pocket around the back of the case, screwed through the rear cover holes. Returns (solids, cuts).
        top=False leaves out the horn-side plate (and its 2 screws)."""
        c, t = CLR, WALL
        xb, xbi = -CASE_X - c - t, -CASE_X - c
        w, wo = CASE_Y + c, CASE_Y + c + t
        zt0, zt1 = TOP_PAD[2] + c, TOP_PAD[2] + c + t
        zs = zt1 if top else COVER_Z                 # how far the walls reach on the horn side
        zb0, zb1 = BOT_PAD[2] - c - t, BOT_PAD[2] - c
        xp = -11.5
        solids = []
        for s in (1, -1):
            solids.append(self.box((xb, s * w, -COVER_Z), (CRADLE_FRONT, s * wo, COVER_Z)))   # side wall
            solids.append(self.box((xb, s * w, zb0), (xp, s * wo, zs)))                       # side wall, full height
        solids += [self.box((xb, -wo, zb0), (xbi, wo, zs)),                                    # back wall
                   self.box((xb, -wo, zb0), (xp, wo, zb1))]                                    # bottom plate
        cuts = [self.cyl(x, y, M2_CLEAR / 2, zb0 - 1, -COVER_Z + 1) for x, y in BOT_SCREWS]
        if top:
            solids.append(self.box((xb, -wo, zt0), (xp, wo, zt1)))                             # top plate
            cuts += [self.cyl(x, y, M2_CLEAR / 2, COVER_Z - 1, zt1 + 1) for x, y in TOP_SCREWS]
        return solids, cuts

    def clevis(self):
        """Arm discs on the horn and the idler disc, with screw holes. Returns (solids, cuts)."""
        solids = [self.cyl(*SHAFT, ARM_R, HORN_FACE, HORN_FACE + ARM_T),
                  self.cyl(*SHAFT, ARM_R, IDLER_FACE - ARM_T, IDLER_FACE)]
        cuts = []
        for z0, z1 in ((HORN_FACE - 1, HORN_FACE + ARM_T + 1), (IDLER_FACE - ARM_T - 1, IDLER_FACE + 1)):
            cuts += [self.cyl(x, y, HORN_HOLE / 2, z0, z1) for x, y in HORN_HOLES]
            cuts.append(self.cyl(*SHAFT, RELIEF_D / 2, z0, z1))
        return solids, cuts

    def horn_cuts(self):
        """Only the horn-side holes (for a plate screwed to the horn)."""
        z0, z1 = HORN_FACE - 1, HORN_FACE + 10
        return [self.cyl(x, y, HORN_HOLE / 2, z0, z1) for x, y in HORN_HOLES] + [self.cyl(*SHAFT, RELIEF_D / 2, z0, z1)]


def chamfer_y(x, z, y0, y1, leg):
    """Cut a 45 deg chamfer with legs ``leg`` on an edge along y at (x, z)."""
    c, h = np.sqrt(0.5), leg / np.sqrt(2)
    q = box((-h, y0, -h), (h, y1, h), R=np.array([[c, 0, c], [0, 1, 0], [-c, 0, c]]))
    q["c"] = [x, (y0 + y1) / 2, z]
    return q


def chamfer_x(y, z, x0, x1, leg):
    """Cut a 45 deg chamfer with legs ``leg`` on an edge along x at (y, z)."""
    c, h = np.sqrt(0.5), leg / np.sqrt(2)
    q = box((x0, -h, -h), (x1, h, h), R=np.array([[1, 0, 0], [0, c, -c], [0, c, c]]))
    q["c"] = [(x0 + x1) / 2, y, z]
    return q


def part(name, link, *groups, cuts=(), clip=()):
    """(union of solids - cuts), intersected with the union of ``clip`` if given."""
    solids, cc = [], list(cuts)
    for s, c in groups:
        solids += s
        cc += c
    return dict(name=name, link=link, solids=solids, cuts=cc, clip=list(clip))


def left_leg():
    Y, R, P, K, A = (Servo("left_" + j) for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"))
    parts = []

    # hip yaw link, two pieces so the roll servo can go in: a bracket (cradle for the roll servo, body hanging
    # down, + walls) and a cap on the yaw horn, held to the walls by 4 M2 screws with sunk heads
    yaw_horn_z = Y.p(*SHAFT, HORN_FACE)[2]
    cap_z0 = yaw_horn_z - 6.0
    rx0, rx1 = sorted([R.p(*SHAFT, -COVER_Z)[0], R.p(*SHAFT, COVER_Z)[0]])
    ry = R.p(*SHAFT, 0)[1]
    wi = CASE_Y + CLR                                  # inner face of the walls
    wy0, wy1 = ry - wi - YAW_WALL, ry + wi + YAW_WALL
    wall_z0 = R.p(CRADLE_FRONT, 0, 0)[2]
    walls = [box((rx0, wy0, wall_z0), (rx1, ry - wi, cap_z0)), box((rx0, ry + wi, wall_z0), (rx1, wy1, cap_z0))]
    cap = [box((rx0, wy0, cap_z0), (rx1, wy1, yaw_horn_z))]
    ys = Y.p(*SHAFT, 0)
    # the two yaw links turn next to each other: on the inner side stay inside YAW_CLIP_R of the yaw axis
    clip = [cyl(ys, (0, 0, 1), YAW_CLIP_R, -200, 200), box((ys[0] - 60, ys[1], 0), (ys[0] + 60, ys[1] + 60, 300))]
    # 45 deg chamfer on the bottom outer edge: the ankle comes up here in a deep sit
    bottom = R.p(-CASE_X - CLR - WALL, 0, 0)[2]
    c45 = np.sqrt(0.5)
    chamfer = [box((-25, -9, -9), (25, 9, 9), R=np.array([[1, 0, 0], [0, c45, -c45], [0, c45, c45]]))]
    chamfer[0]["c"] = [ys[0], ry + wi + WALL, bottom]
    spots = [(ys[0] + dx, y) for dx in (-9.0, 9.0) for y in ((wy0 + ry - wi) / 2, (wy1 + ry + wi) / 2)]
    Z = (0, 0, 1)
    cap_cuts = [cyl((x, y, 0), Z, M2_CLEAR / 2, cap_z0 - 1, yaw_horn_z + 1) for x, y in spots]
    cap_cuts += [cyl((x, y, 0), Z, 2.1, yaw_horn_z - 1.8, yaw_horn_z + 1) for x, y in spots]
    pilots = [cyl((x, y, 0), Z, PILOT / 2, cap_z0 - 9, cap_z0 + 1) for x, y in spots]
    parts.append(part("left_hip_yaw_bracket", "left_hip_yaw_link", R.cradle(), (walls, []),
                      cuts=R.keepout() + R.rotor_keepout() + chamfer + pilots, clip=clip))
    parts.append(part("left_hip_yaw_cap", "left_hip_yaw_link", (cap, []),
                      cuts=Y.horn_cuts() + Y.keepout() + cap_cuts, clip=clip))

    # hip cross: clevis on the roll servo + clevis on the pitch servo, joined in front of the pitch servo
    rs, ps = R.p(*SHAFT, 0), P.p(*SHAFT, 0)
    r_horn_x, r_idl_x = R.p(*SHAFT, HORN_FACE)[0], R.p(*SHAFT, IDLER_FACE)[0]
    p_horn_y, p_idl_y = P.p(*SHAFT, HORN_FACE)[1], P.p(*SHAFT, IDLER_FACE)[1]
    zc = ps[2]
    bridge_x0 = ps[0] + 20.0                      # stays clear of the pitch servo's front corners over its range
    bridge_x1 = r_horn_x + ARM_T
    s = [box((r_horn_x, rs[1], rs[2] - 10), (r_horn_x + ARM_T, p_horn_y, rs[2] + 10)),          # roll horn arm -> +y
         box((r_idl_x - ARM_T, rs[1] + 8, rs[2] - 4), (r_idl_x, p_horn_y, rs[2] + 9)),          # roll idler arm -> +y, above the plugs
         box((r_idl_x - ARM_T, p_horn_y - ARM_T, zc - ARM_R), (bridge_x1, p_horn_y, zc + ARM_R)),  # pitch horn plate
         box((ps[0], p_idl_y, zc - ARM_R), (bridge_x1, p_idl_y + ARM_T, zc + ARM_R)),           # pitch idler arm
         box((bridge_x0, p_horn_y - ARM_T, zc - ARM_R), (bridge_x1, p_idl_y + ARM_T, zc + ARM_R))]  # front bridge
    # upper-back corner chamfer: rises toward the hip yaw servo when the hip rolls out
    cc = [chamfer_y(r_idl_x - ARM_T, zc + ARM_R, rs[1], p_horn_y + 1, 9.0)]
    parts.append(part("left_hip_cross", "left_hip_cross", R.clevis(), P.clevis(), (s, []),
                      cuts=R.keepout() + P.keepout() + cc))

    # thigh: cradles for the pitch and knee servos + a web between them in the case layer
    y0, y1 = P.p(*SHAFT, COVER_Z)[1], P.p(*SHAFT, -COVER_Z)[1]
    y0, y1 = min(y0, y1) + CLR, max(y0, y1) - CLR
    web = [box((-42.0, y0, 113.0), (-28.0, y1, 149.0))]
    # upper-back edge chamfer on the inner (horn-side) layer: same reason as the hip cross
    p_back = P.p(-CASE_X - CLR - WALL, 0, 0)[0]
    p_top = P.p(*SHAFT, 0)[2] + CASE_Y + CLR + WALL
    cc = [chamfer_y(p_back, p_top, P.p(*SHAFT, HORN_FACE + 2)[1], y0, 10.0)]
    parts.append(part("left_thigh", "left_thigh", P.cradle(), K.cradle(), (web, []),
                      cuts=P.keepout() + K.keepout() + P.rotor_keepout() + K.rotor_keepout() + cc))

    # shin: clevis on the knee servo, arms down to a cradle for the ankle servo (body pointing up)
    ks, as_ = K.p(*SHAFT, 0), A.p(*SHAFT, 0)
    k_horn_y, k_idl_y = K.p(*SHAFT, HORN_FACE)[1], K.p(*SHAFT, IDLER_FACE)[1]
    a_horn_y = A.p(*SHAFT, HORN_FACE)[1]
    a_top = A.p(-CASE_X - CLR - WALL, 0, 0)[2]          # top of the ankle cradle's back wall
    hx = 10.0
    lo_y, hi_y = sorted([k_horn_y, k_idl_y])
    s = [box((ks[0] - hx, lo_y - ARM_T, a_top - 3), (ks[0] + hx, lo_y, ks[2])),                 # knee arm (-y side)
         box((ks[0] - hx, hi_y, as_[2] + 14), (ks[0] + hx, hi_y + ARM_T, ks[2])),                # knee arm (+y side)
         box((ks[0] - hx, a_horn_y - 1.4, as_[2] + 14), (ks[0] + hx, hi_y + ARM_T, a_top))]       # block to the cradle
    # round the top of the ankle cradle about the knee axis so it clears the thigh at both ends of the knee range
    kax = K.T[:3, 2]
    clip = [cyl(ks, kax, SHIN_TOP_R, -60, 60), box((ks[0] - 60, ks[1] - 60, 0), (ks[0] + 60, ks[1] + 60, ks[2] - SHIN_TOP_R + 2))]
    parts.append(part("left_shin", "left_shin", K.clevis(), A.cradle(), (s, []),
                      cuts=K.keepout() + A.keepout() + A.rotor_keepout(), clip=clip))

    # foot: clevis on the ankle servo + sole plate (footprint and sole height of the baseline foot, scaled)
    a_idl_y = A.p(*SHAFT, IDLER_FACE)[1]
    plate = (np.array([-67.3, 38.0, 17.5]), np.array([2.9, 91.5, 23.5]))
    s = [box((as_[0] - hx, a_horn_y, plate[1][2]), (as_[0] + hx, a_horn_y + ARM_T, as_[2])),
         box((as_[0] - hx, a_idl_y - ARM_T, plate[1][2]), (as_[0] + hx, a_idl_y, as_[2])),
         box(*plate)]
    parts.append(part("left_foot", "left_foot", A.clevis(), (s, []), cuts=A.keepout()))
    return parts


def battery_tray(back_x, bridge_hw, bridge_z0, bridge_z1, deck_top):
    """Tray for the 2S pack, hung on the trunk's back face: a front plate screwed into the solid bridge between the
    hip yaw servos (4 x M2 self-tapping), a lip hooked over the deck, an open-top pocket for the two cells and a
    floor with two slots for a hook-and-loop strap over the cells. In the taller layouts, bosses on the back carry
    the controller board. Returns (solids, cuts, pilot cuts for the trunk).

    The space behind the trunk is squeezed from two sides: below about z 160 the thighs come in at combined hip
    yaw and roll, and above the deck the head comes down when the neck tips back with the head pitched back. The
    flat layout (one layer, 8 mm below the deck) sits in the gap between them; see the README for what still
    collides."""
    X = (1, 0, 0)
    f0, f1 = back_x - TRAY_PLATE, back_x                       # front plate
    px1 = f0                                                    # pocket, front
    cell = CELL_D + 2 * CELL_CLR
    nx = 1                                                      # cells side by side in x
    if TRAY_LAYOUT == "upright":                                # cells standing, side by side in y
        hw, depth = cell, POCKET_L - 2.0                        # the cells stand 2 mm proud, to grip them
        board = BOARD_HOLES[::-1]
    elif TRAY_LAYOUT == "flat":                                 # cells lying across (y), side by side in x
        hw, depth, nx = POCKET_L / 2, cell - 1.0, 2             # they stand 1 mm proud
        board = BOARD_HOLES
    else:                                                       # cells lying across (y), stacked in z
        hw, depth = POCKET_L / 2, 2 * cell - 1.0                # the top one stands 1 mm proud
        board = BOARD_HOLES
    px0 = px1 - nx * cell                                       # pocket, back
    bx0 = px0 - TRAY_WALL                                       # back wall
    ow = hw + TRAY_WALL
    pz1 = deck_top + TRAY_DZ                                    # pocket top, level with the deck unless raised
    pz0 = pz1 - depth
    fz0 = pz0 - TRAY_WALL
    solids = [box((bx0, -ow, fz0), (f1, ow, pz1)),                                  # block, pocket cut below
              box((f0, -15.0, deck_top), (back_x + 7.0, 15.0, deck_top + LIP_T)),     # lip over the deck
              box((f0, -15.0, pz1), (f1, 15.0, deck_top + LIP_T))]                  # plate up to the lip
    cuts = [box((px0, -hw, pz0), (px1, hw, pz1 + 1))]                               # pocket
    # strap slots through the floor, just behind the front plate
    cuts += [box((px1 - 3.0, sy * 3.0, fz0 - 1), (px1 - 0.5, sy * (hw - 3.0), pz0 + 1)) for sy in (1, -1)]
    # 45 deg chamfers on the front outer vertical edges: the trunk's matching chamfers leave room for a thigh corner
    c45, h = np.sqrt(0.5), 4.0 / np.sqrt(2)                      # 4 mm legs: keeps the pocket corner walled
    for sy in (1, -1):
        q = box((-h, -h, fz0 - 1), (h, h, pz1 + 1), R=np.array([[c45, -c45, 0], [c45, c45, 0], [0, 0, 1]]))
        q["c"] = [f1, sy * ow, (fz0 + pz1) / 2]
        cuts.append(q)
    if TRAY_BOT_CHAMFER:
        cuts += [chamfer_x(sy * ow, fz0, bx0 - BOSS[1] - 1, f1 + 1, TRAY_BOT_CHAMFER) for sy in (1, -1)]
    if TRAY_TOP_CHAMFER:
        cuts.append(chamfer_y(bx0, pz1, -ow - 1, ow + 1, TRAY_TOP_CHAMFER))
    # screws into the solid bridge: y inside its half-width, z inside its height and reachable from the pocket
    zs = (max(bridge_z0 + 6.0, pz0 + 3.0), min(bridge_z1 - 6.0, pz1 - 3.0))
    spots = [(sy * (bridge_hw - 5.5), z) for sy in (1, -1) for z in zs]
    pilots = []
    for y, z in spots:
        cuts.append(cyl((0, y, z), X, M2_CLEAR / 2, f0 - 1, f1 + 1))
        cuts.append(cyl((0, y, z), X, M2_HEAD[0] / 2, f0 - 1, f0 + M2_HEAD[1]))       # head counterbore
        pilots.append(cyl((0, y, z), X, PILOT / 2, f1 - 1, f1 + 10))
    # controller board bosses on the back wall, with M2 self-tapping pilots. The flat tray's back wall is too low
    # for the board: it goes on a lid over the cells instead, once the board is chosen.
    if TRAY_LAYOUT == "flat":
        return solids, cuts, pilots
    zc = (fz0 + pz1) / 2
    for sy in (1, -1):
        for sz in (1, -1):
            y, z = sy * board[0] / 2, zc + sz * board[1] / 2
            solids.append(cyl((0, y, z), X, BOSS[0] / 2, bx0 - BOSS[1], bx0 + 0.5))
            cuts.append(cyl((0, y, z), X, PILOT / 2, bx0 - BOSS[1] - 1, bx0 + 0.5))
    return solids, cuts, pilots


def body():
    YL, YR, N = Servo("left_hip_yaw"), Servo("right_hip_yaw"), Servo("neck_pitch")
    HP, HY, HR = Servo("head_pitch"), Servo("head_yaw"), Servo("head_roll")
    parts = []

    # trunk: cradles for both hip yaw servos, a bridge between them, a deck over their idler side and a clevis
    # for the neck pitch servo standing on the deck
    back_x, front_x = YL.p(-CASE_X - CLR - WALL, 0, 0)[0], YL.p(CRADLE_FRONT, 0, 0)[0]
    z0, z1 = sorted([YL.p(0, 0, -COVER_Z)[2], YL.p(0, 0, COVER_Z)[2]])
    yl = YL.p(*SHAFT, 0)[1]
    inner, outer = yl - CASE_Y - CLR, yl + CASE_Y + CLR + WALL
    deck_z0 = YL.p(*SHAFT, IDLER_FACE)[2] + 0.4
    deck_z1 = deck_z0 + ARM_T
    ns = N.p(*SHAFT, 0)
    s = [box((back_x, -inner - 0.5, z0), (front_x, inner + 0.5, z1)),                                # bridge
         box((back_x, -outer, deck_z0), (YL.p(CASE_X, 0, 0)[0] - 2.0, outer, deck_z1))]              # deck
    for ya, yb in ((N.p(*SHAFT, HORN_FACE)[1], N.p(*SHAFT, HORN_FACE + ARM_T)[1]),
                   (N.p(*SHAFT, IDLER_FACE - ARM_T)[1], N.p(*SHAFT, IDLER_FACE)[1])):
        s.append(box((8.0, ya, deck_z0), (ns[0], yb, ns[2] - 2.0)))     # neck arms, kept low: the neck swings over them
    cuts = YL.keepout() + YR.keepout() + N.keepout()
    cuts += YL.rotor_keepout(r=24.5, idler=False) + YR.rotor_keepout(r=24.5, idler=False)
    # no horn-side plates on these cradles: the thighs come up there when the hips roll
    # ... and no outer side walls either, for the same reason
    lim = yl + CASE_Y + CLR
    clip = [box((-200, -lim, 0), (200, lim, 400))]
    # 45 deg chamfers on the back outer vertical edges: a thigh corner reaches there in fall recovery
    c45, h = np.sqrt(0.5), 8.0 / np.sqrt(2)
    for sy in (1, -1):
        q = box((-h, -h, z0 - 1), (h, h, deck_z1 + 1), R=np.array([[c45, -c45, 0], [c45, c45, 0], [0, 0, 1]]))
        q["c"] = [back_x, sy * lim, (z0 + deck_z1) / 2]
        cuts.append(q)
    tray, tray_cuts, trunk_pilots = battery_tray(back_x, inner, z0, z1, deck_z1)
    parts.append(part("trunk", "trunk", YL.cradle(top=False), YR.cradle(top=False), N.clevis(), (s, []),
                      cuts=cuts + trunk_pilots, clip=clip))
    parts.append(part("battery_tray", "trunk", (tray, []), cuts=tray_cuts))

    # neck: cradles for the neck pitch servo (body forward-up) and the head pitch servo (body down), joined by a web
    yc = COVER_Z - CLR
    web = [box((38.0, -yc, 216.0), (52.0, yc, 230.0))]
    parts.append(part("neck", "neck", N.cradle(), HP.cradle(), (web, []),
                      cuts=N.keepout() + HP.keepout() + N.rotor_keepout() + HP.rotor_keepout()))

    # neck top: clevis on the head pitch servo, arms up to a plate screwed to the head yaw horn
    hp, hyf = HP.p(*SHAFT, 0), HY.p(*SHAFT, HORN_FACE)
    hx = 10.0
    y_h, y_i = HP.p(*SHAFT, HORN_FACE)[1], HP.p(*SHAFT, IDLER_FACE)[1]
    s = [box((hp[0] - hx, y_h, hp[2]), (hp[0] + hx, y_h + ARM_T, hyf[2])),
         box((hp[0] - hx, y_i - ARM_T, hp[2]), (hp[0] + hx, y_i, hyf[2])),
         box((hp[0] - hx, y_i - ARM_T, hyf[2] - ARM_T), (hp[0] + hx, y_h + ARM_T, hyf[2]))]
    parts.append(part("neck_top", "neck_top", HP.clevis(), (s, []),
                      cuts=HY.horn_cuts() + HP.keepout() + HY.keepout()))

    # head yaw link: cradle for the head yaw servo + a plate on the head roll horn (screw it on before the
    # yaw servo goes in: the servo covers the screw heads)
    rf = HR.p(*SHAFT, HORN_FACE)
    wall_x = HY.p(0, CASE_Y + CLR, 0)[0]                 # inner face of the cradle side wall facing the roll servo
    wall_x = min(wall_x, HY.p(0, -CASE_Y - CLR, 0)[0])
    y_back = HY.p(-CASE_X - CLR - WALL, 0, 0)[1]
    zc = rf[2]
    s = [box((rf[0], y_back, zc - ARM_R), (rf[0] + ARM_T, ARM_R, zc + ARM_R)),                        # horn plate
         box((rf[0] + ARM_T, y_back, zc - ARM_R), (wall_x - WALL, -9.0, zc + ARM_R)),                 # spacer
         box((wall_x - WALL, y_back, HY.p(0, 0, COVER_Z)[2]), (wall_x, 5.0, HY.p(0, 0, -COVER_Z)[2]))]  # long side wall
    parts.append(part("head_yaw_link", "head_yaw_link", HY.cradle(), (s, []),
                      cuts=HR.horn_cuts() + HY.keepout() + HR.keepout() + HY.rotor_keepout(r=25.5, idler=False)))

    # head: cradle for the head roll servo (no horn-side plate: the head yaw link's plate turns there).
    # The shells and the beak are not designed yet.
    parts.append(part("head", "head", HR.cradle(top=False), cuts=HR.keepout() + HR.rotor_keepout(idler=False)))
    return parts


def mirror(prim):
    M = np.diag([1.0, -1.0, 1.0])
    q = dict(prim)
    q["c"] = (M @ np.array(prim["c"])).tolist()
    if prim["t"] == "box":
        q["R"] = (M @ np.array(prim["R"]) @ M).tolist()
    else:
        q["a"] = (M @ np.array(prim["a"])).tolist()
    return q


def main():
    left = left_leg()
    right = [dict(name=p["name"].replace("left", "right"), link=p["link"].replace("left", "right"),
                  solids=[mirror(x) for x in p["solids"]], cuts=[mirror(x) for x in p["cuts"]],
                  clip=[mirror(x) for x in p["clip"]]) for p in left]
    parts = body() + left + right
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    (out / "parts.json").write_text(json.dumps(dict(version="v0", parts=parts), indent=1))
    (HERE / "stl").mkdir(exist_ok=True)
    (HERE / "stl" / "manifest.json").write_text(json.dumps({p["name"]: p["link"] for p in parts}, indent=1))
    print(f"{len(parts)} parts -> {out / 'parts.json'}")


if __name__ == "__main__":
    main()
