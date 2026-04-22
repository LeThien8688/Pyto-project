import math
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


def fmt(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    s = s.rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return s


@dataclass
class DoubleArcInput:
    Dx: float
    C1: float
    R2: float
    Hx: float
    R1: float | None = None
    L: float | None = None
    step_x: float = 2.0
    W: float = 0.4

    def resolved(self) -> "DoubleArcInput":
        r1 = self.Dx / 2.0 if self.R1 is None else self.R1
        l_val = self.C1 + (self.Dx / 2.0) if self.L is None else self.L
        return DoubleArcInput(
            Dx=self.Dx,
            C1=self.C1,
            R2=self.R2,
            Hx=self.Hx,
            R1=r1,
            L=l_val,
            step_x=self.step_x,
            W=self.W,
        )


@dataclass
class TailInput:
    R: float
    r: float
    L: float
    cutting_mount: float = 1.0
    alpha_deg: float | None = None
    W: float = 0.4
    U: float = 0.4


def build_double_arc_geometry(inp: DoubleArcInput) -> dict:
    inp = inp.resolved()
    if inp.Dx <= 0 or inp.C1 <= 0 or inp.R1 <= 0 or inp.R2 <= 0 or inp.Hx < 0:
        raise ValueError("Input không hợp lệ")
    if inp.Hx >= inp.Dx:
        raise ValueError("Hx phải nhỏ hơn Dx")
    if inp.L < 0:
        raise ValueError("L phải >= 0")

    ratio = ((inp.Hx / 2.0) + inp.R2) / (inp.R1 + inp.R2)
    ratio = max(-1.0, min(1.0, ratio))

    beta_deg = round(math.degrees(math.acos(ratio)), 0)
    beta_rad = math.radians(beta_deg)

    Bx = round(2.0 * inp.R1 * math.cos(beta_rad), 3)
    Bz = round(inp.C1 - inp.R1 * math.sin(beta_rad), 3)

    Az = round(Bz - inp.R2 * math.sin(beta_rad), 3)
    Ax = inp.Hx

    Cz = inp.C1
    Cx = 2.0 * inp.R1

    return {
        "beta_deg": beta_deg,
        "A": (Az, Ax),
        "B": (Bz, Bx),
        "C": (Cz, Cx),
        "O": (0.0, inp.Hx),
    }


def build_rough_cuts(inp: DoubleArcInput, geo: dict) -> dict:
    inp = inp.resolved()
    _, Bx = geo["B"]
    Cz, _ = geo["C"]
    Az, _ = geo["A"]

    r1_x, r1_z = [], []
    r2_x, r2_z = [], []

    solan_cat = int((inp.Dx - inp.Hx) / inp.step_x)

    for i in range(2, solan_cat + 2):
        x = round(inp.Dx - inp.step_x * (i - 1), 3)

        if x >= Bx:
            ratio = (x / 2.0) / inp.R1
            ratio = max(-1.0, min(1.0, ratio))
            beta_r1 = round(math.degrees(math.acos(ratio)), 0)

            r_z = round(inp.R1 * math.sin(math.radians(beta_r1)), 3)
            z = round(-((Cz - r_z) - inp.W), 3)

            r1_x.append(x)
            r1_z.append(z)

        elif Bx > x > inp.Hx:
            dx_r2 = (inp.Hx / 2.0) + inp.R2
            x_r2 = dx_r2 - (x / 2.0)

            ratio = x_r2 / inp.R2
            ratio = max(-1.0, min(1.0, ratio))
            beta_r2 = round(math.degrees(math.acos(ratio)), 0)

            r_z = round(inp.R2 * math.sin(math.radians(beta_r2)), 3)
            z = round(-(Az + r_z - inp.W), 3)

            r2_x.append(x)
            r2_z.append(z)

    return {"R1_X": r1_x, "R1_Z": r1_z, "R2_X": r2_x, "R2_Z": r2_z}


def transform_z_with_L(inp: DoubleArcInput, geo: dict, rough: dict) -> dict:
    inp = inp.resolved()
    def z_geo(z: float) -> float:
        return round(inp.L - z, 3)

    def z_rough(z: float) -> float:
        return round(inp.L + z, 3)

    geo_new = {
        "A": (z_geo(geo["A"][0]), geo["A"][1]),
        "B": (z_geo(geo["B"][0]), geo["B"][1]),
        "C": (z_geo(geo["C"][0]), geo["C"][1]),
        "O": (z_geo(geo["O"][0]), geo["O"][1]),
        "beta_deg": geo["beta_deg"],
    }

    rough_new = {
        "R1_X": list(rough["R1_X"]),
        "R1_Z": [z_rough(z) for z in rough["R1_Z"]],
        "R2_X": list(rough["R2_X"]),
        "R2_Z": [z_rough(z) for z in rough["R2_Z"]],
    }

    return {"geometry": geo_new, "rough": rough_new}


def coarse_cuts_one(inp: TailInput) -> dict:
    if inp.R <= 0 or inp.r < 0:
        raise ValueError("R và r phải > 0")
    if inp.r >= inp.R:
        raise ValueError("Cần R > r để có tiếp xúc trong")
    if inp.cutting_mount <= 0:
        raise ValueError("cutting_mount phải > 0")

    alpha_deg = inp.alpha_deg
    if alpha_deg is None:
        d = inp.R - inp.r
        c = (inp.L - inp.r) / d
        if not -1.0 <= c <= 1.0:
            raise ValueError("Không thể tiếp xúc: (L - r) ngoài miền.")
        alpha_deg = math.degrees(math.acos(c))

    a = math.radians(alpha_deg)
    d = inp.R - inp.r

    Cx, Cy = d * math.cos(a), d * math.sin(a)
    A = (inp.R * math.cos(a), inp.R * math.sin(a))
    B = (inp.L, Cy)
    C = (Cx, Cy)
    E = (0.0, inp.R)

    geom_base = {
        "alpha_deg": alpha_deg,
        "A": A,
        "B": B,
        "C": C,
        "E": E,
        "touch_err": abs((inp.L - Cx) - inp.r),
    }

    A_ = (A[0] - inp.L, A[1])
    B_ = (B[0] - inp.L, B[1])
    C_ = (C[0] - inp.L, C[1])
    O_ = (-inp.L, 0.0)
    E_ = (-inp.L, inp.R)

    geom = {"alpha_deg": alpha_deg, "A": A_, "B": B_, "C": C_, "O": O_, "E": E_}

    step_r = inp.cutting_mount / 2.0   # vì output X là đường kính

    x_top = inp.R - step_r
    x_bot = B_[1] + inp.U

    if x_bot > x_top:
      return {"geom_base": geom_base, "geom": geom, "x_levels": [], "z_edge": []}

    span = x_top - x_bot
    n_steps = int(span // step_r) + 1

    x_all = [x_top - i * step_r for i in range(n_steps)]
    if x_all[-1] > x_bot + 1e-9:
      x_all.append(x_bot)

    x_switch = A_[1]
    x_levels = []
    z_edge = []

    for x in x_all:
        if x >= x_switch - 1e-12:
            val = max(0.0, inp.R * inp.R - x * x)
            z = O_[0] + math.sqrt(val) + inp.W
        else:
            val = max(0.0, inp.r * inp.r - (x - C_[1]) ** 2)
            z = C_[0] + math.sqrt(val) + inp.W

        if z > 0:
            continue

        x_levels.append(x)
        z_edge.append(z)

    return {"geom_base": geom_base, "geom": geom, "x_levels": x_levels, "z_edge": z_edge}


def sample_arc(center_z, center_y, radius, theta1_deg, theta2_deg, n=120):
    th = np.radians(np.linspace(theta1_deg, theta2_deg, n))
    z = center_z + radius * np.cos(th)
    y = center_y + radius * np.sin(th)
    return z, y


def build_tail_cut_info_with_L(inp: DoubleArcInput,
                            geo: dict,
                            tail_R: float,
                            tail_r: float,
                            tail_L: float,
                            tail_cutting_mount: float = 1.0,
                            tail_alpha_deg: float | None = None,
                            tail_W: float = 0.4,
                            tail_U: float = 0.4) -> dict:
    inp = inp.resolved()
    tail = coarse_cuts_one(
        TailInput(
            R=tail_R,
            r=tail_r,
            L=tail_L,
            cutting_mount=tail_cutting_mount,
            alpha_deg=tail_alpha_deg,
            W=tail_W,
            U=tail_U,
        )
    )

    Cz, Cx = geo["C"]
    c_plot_y = (Cx + inp.Hx) / 2.0
    alpha_tail = tail["geom_base"]["alpha_deg"]
    a_tail = math.radians(alpha_tail)
    d_tail = tail_R - tail_r

    O0 = (0.0, 0.0)
    C0 = (d_tail * math.cos(a_tail), d_tail * math.sin(a_tail))
    A0 = (tail_R * math.cos(a_tail), tail_R * math.sin(a_tail))
    E0 = (0.0, tail_R)

    def profile_plot_point_from_local(pt):
        z0, x0 = pt
        plot_z = -z0 - Cz
        plot_y = x0 - E0[1] + c_plot_y
        return plot_z, plot_y

    def point_to_print(pt):
        plot_z, plot_y = profile_plot_point_from_local(pt)
        z_print = round(inp.L + plot_z, 3)
        x_print = round(2.0 * (plot_y - (inp.Hx / 2.0)), 3)
        return z_print, x_print

    left_wall_plot = -inp.L

    d_print = point_to_print(A0)
    e_print = point_to_print(C0)

    rough_tail = []
    rough_tail_plot = []
    for x_half in tail["x_levels"]:
        val = max(0.0, tail_R * tail_R - x_half * x_half)
        offset_from_left_wall = round(tail_R - math.sqrt(val) - tail_W, 3)
        z_end_print = round(inp.L - offset_from_left_wall, 3)

        y_plot = x_half - E0[1] + c_plot_y
        z_end_plot = -z_end_print

        rough_tail.append({
            "X": round(2.0 * x_half, 3),
            "Z_start": 0.0,
            "Z_end": z_end_print,
            "Z_offset": offset_from_left_wall,
        })
        rough_tail_plot.append({
            "y": round(y_plot, 3),
            "z_start": left_wall_plot,
            "z_end": z_end_plot,
        })

    return {
        "tail": tail,
        "D": d_print,
        "E": e_print,
        "rough_tail": rough_tail,
        "rough_tail_plot": rough_tail_plot,
        "left_wall_plot": left_wall_plot,
    }


def print_result_with_L(inp: DoubleArcInput, geo: dict, rough: dict,
                        tail_R: float, tail_r: float, tail_L: float,
                        tail_cutting_mount: float = 1.0,
                        tail_alpha_deg: float | None = None,
                        tail_W: float = 0.4, tail_U: float = 0.4) -> None:
    inp = inp.resolved()
    new_data = transform_z_with_L(inp, geo, rough)
    geo_new = new_data["geometry"]
    rough_new = new_data["rough"]
    tail_info = build_tail_cut_info_with_L(
        inp, geo,
        tail_R=tail_R,
        tail_r=tail_r,
        tail_L=tail_L,
        tail_cutting_mount=tail_cutting_mount,
        tail_alpha_deg=tail_alpha_deg,
        tail_W=tail_W,
        tail_U=tail_U,
    )

    print("========== INPUT ==========")
    print(f"Dx   = {fmt(inp.Dx)}")
    print(f"C1   = {fmt(inp.C1)}")
    print(f"R1   = Dx/2 = {fmt(inp.R1)}")
    print(f"R2   = {fmt(inp.R2)}")
    print(f"Hx   = {fmt(inp.Hx)}")
    print(f"L    = C1 + Dx/2 = {fmt(inp.L)}")
    print(f"step = {fmt(inp.step_x)}")
    print(f"W    = {fmt(inp.W)}")
    print()
    print("========== PRINT THEO L ==========")
    print(f"A = (Z={fmt(geo_new['A'][0])}, X={fmt(geo_new['A'][1])})")
    print(f"B = (Z={fmt(geo_new['B'][0])}, X={fmt(geo_new['B'][1])})")
    print(f"C = (Z={fmt(geo_new['C'][0])}, X={fmt(geo_new['C'][1])})")
    print(f"O = (Z={fmt(geo_new['O'][0])}, X={fmt(geo_new['O'][1])})")
    print(f"D = (Z={fmt(tail_info['D'][0])}, X={fmt(tail_info['D'][1])})")
    print(f"E = (Z={fmt(tail_info['E'][0])}, X={fmt(tail_info['E'][1])})")

    for x, z in zip(rough_new["R1_X"], rough_new["R1_Z"]):
        print(f"X{fmt(x)}  Z{fmt(z)}")
    for x, z in zip(rough_new["R2_X"], rough_new["R2_Z"]):
        print(f"X{fmt(x)}  Z{fmt(z)}")

    print("========== ROUGH CUT - TAIL THEO L ==========")
    for row in tail_info["rough_tail"]:
        print(f"X{fmt(row['X'])}  Z{fmt(row['Z_start'])} -> Z{fmt(row['Z_end'])}")


def plot_full_profile(inp: DoubleArcInput, geo: dict, rough: dict,
                      tail_R: float, tail_r: float, tail_L: float,
                      tail_cutting_mount: float = 1.0,
                      tail_alpha_deg: float | None = None,
                      tail_W: float = 0.4, tail_U: float = 0.4) -> None:
    inp = inp.resolved()
    Az, Ax = geo["A"]
    Bz, Bx = geo["B"]
    Cz, Cx = geo["C"]
    Oz, Ox = geo["O"]
    beta_deg = geo["beta_deg"]

    A_plot = (-Az, (Ax + inp.Hx) / 2.0)
    B_plot = (-Bz, (Bx + inp.Hx) / 2.0)
    C_plot = (-Cz, (Cx + inp.Hx) / 2.0)
    O_plot = (Oz, Ox)

    center_r2 = (-Az, Ax + inp.R2)
    center_r1 = (-Cz, inp.Hx / 2.0)

    z_r2, y_r2 = sample_arc(center_r2[0], center_r2[1], inp.R2, 270 - beta_deg, 270, 100)
    z_r1, y_r1 = sample_arc(center_r1[0], center_r1[1], inp.R1, 90, 90 - beta_deg, 140)

    tail = coarse_cuts_one(
        TailInput(
            R=tail_R,
            r=tail_r,
            L=tail_L,
            cutting_mount=tail_cutting_mount,
            alpha_deg=tail_alpha_deg,
            W=tail_W,
            U=tail_U,
        )
    )

    alpha_tail = tail["geom_base"]["alpha_deg"]
    a_tail = math.radians(alpha_tail)
    d_tail = tail_R - tail_r

    O0 = (0.0, 0.0)
    O0 = (0.0, 0.0)
    C0 = (d_tail * math.cos(a_tail), d_tail * math.sin(a_tail))
    A0 = (tail_R * math.cos(a_tail), tail_R * math.sin(a_tail))
    E0 = (0.0, tail_R)

    def map_tail_point(pt):
        z0, x0 = pt
        return (-z0 + C_plot[0], x0 - E0[1] + C_plot[1])

    A_tail_plot = map_tail_point(A0)
    C_tail_plot = map_tail_point(C0)

    theta_big = np.radians(np.linspace(90, alpha_tail, 180))
    z_big = tail_R * np.cos(theta_big)
    x_big = tail_R * np.sin(theta_big)
    z_big_plot = [-z + C_plot[0] for z in z_big]
    y_big_plot = [x - E0[1] + C_plot[1] for x in x_big]

    theta_small = np.radians(np.linspace(alpha_tail, 0, 140))
    z_small = C0[0] + tail_r * np.cos(theta_small)
    x_small = C0[1] + tail_r * np.sin(theta_small)
    z_small_plot = [-z + C_plot[0] for z in z_small]
    y_small_plot = [x - E0[1] + C_plot[1] for x in x_small]

    tail_info = build_tail_cut_info_with_L(
        inp, geo,
        tail_R=tail_R,
        tail_r=tail_r,
        tail_L=tail_L,
        tail_cutting_mount=tail_cutting_mount,
        tail_alpha_deg=tail_alpha_deg,
        tail_W=tail_W,
        tail_U=tail_U,
    )
    left_wall_z = tail_info["left_wall_plot"]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot([O_plot[0], A_plot[0]], [O_plot[1], A_plot[1]], linewidth=2.6, label="Đường OA")
    ax.plot(z_r2, y_r2, linewidth=2.6, label="Cung R2")
    ax.plot(z_r1, y_r1, linewidth=2.8, label="Cung R1")
    ax.plot(z_big_plot, y_big_plot, linewidth=2.8, label="Tail R")
    ax.plot(z_small_plot, y_small_plot, linewidth=2.6, label="Tail r")

    for x, z in zip(rough["R1_X"], rough["R1_Z"]):
        y = (x + inp.Hx) / 2.0
        ax.plot([z, 0], [y, y], "--", linewidth=1.0, alpha=0.8)
        ax.scatter(z, y, s=16)

    for x, z in zip(rough["R2_X"], rough["R2_Z"]):
        y = (x + inp.Hx) / 2.0
        ax.plot([z, 0], [y, y], "--", linewidth=1.0, alpha=0.8)
        ax.scatter(z, y, s=16)

    for row in tail_info["rough_tail_plot"]:
        ax.plot([row["z_start"], row["z_end"]], [row["y"], row["y"]], "--", linewidth=1.0, alpha=0.8)
        ax.scatter(row["z_end"], row["y"], s=16)

    points = {"O": O_plot, "A": A_plot, "B": B_plot, "C": C_plot}
    for label, (z, y) in points.items():
        ax.scatter(z, y, s=36)
        ax.annotate(label, (z, y), xytext=(6, 6), textcoords="offset points", fontsize=11)

    ax.scatter(A_tail_plot[0], A_tail_plot[1], s=28)
    ax.annotate("D", A_tail_plot, xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax.scatter(C_tail_plot[0], C_tail_plot[1], s=28)
    ax.annotate("E", C_tail_plot, xytext=(6, 6), textcoords="offset points", fontsize=10)

    all_z = (
        [O_plot[0], A_plot[0], B_plot[0], C_plot[0], A_tail_plot[0], C_tail_plot[0], left_wall_z]
        + rough["R1_Z"] + rough["R2_Z"] + list(z_r1) + list(z_r2)
        + list(z_big_plot) + list(z_small_plot)
        + [row["z_end"] for row in tail_info["rough_tail_plot"]]
    )
    all_y = (
        [O_plot[1], A_plot[1], B_plot[1], C_plot[1], A_tail_plot[1], C_tail_plot[1]]
        + [(x + inp.Hx) / 2.0 for x in rough["R1_X"]]
        + [(x + inp.Hx) / 2.0 for x in rough["R2_X"]]
        + list(y_r1) + list(y_r2) + list(y_big_plot) + list(y_small_plot)
        + [row["y"] for row in tail_info["rough_tail_plot"]]
    )

    z_min, z_max = min(all_z), max(all_z)
    y_min, y_max = min(all_y), max(all_y)
    pad_z = max(2.0, (z_max - z_min) * 0.08)
    pad_y = max(1.2, (y_max - y_min) * 0.12)

    ax.set_xlim(z_min - pad_z, z_max + pad_z)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Trục Z", fontsize=12)
    ax.set_ylabel("Trục X/2", fontsize=12)
    ax.set_title("Mô phỏng profile ghép tail trực tiếp", fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=10)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    inp = DoubleArcInput(
        Dx=40,
        C1=36,
        R2=7,
        Hx=15.5,
        R1=None,
        L=None,
        step_x=2.0,
        W=0.4,
    ).resolved()

    geo = build_double_arc_geometry(inp)
    rough = build_rough_cuts(inp, geo)

    tail_R = inp.Dx / 2.0
    tail_r = 0.0
    tail_L = 20
    tail_cutting_mount = 2.0
    tail_W = 0.4
    tail_U = 0.4

    print_result_with_L(
        inp, geo, rough,
        tail_R=tail_R,
        tail_r=tail_r,
        tail_L=tail_L,
        tail_cutting_mount=tail_cutting_mount,
        tail_alpha_deg=None,
        tail_W=tail_W,
        tail_U=tail_U,
    )

    plot_full_profile(
        inp, geo, rough,
        tail_R=tail_R,
        tail_r=tail_r,
        tail_L=tail_L,
        tail_cutting_mount=tail_cutting_mount,
        tail_alpha_deg=None,
        tail_W=tail_W,
        tail_U=tail_U,
    )
