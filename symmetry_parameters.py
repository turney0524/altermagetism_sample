"""One-orbital square-lattice altermagnet: mirror, C4, and C2 constraints.

Basis (A up, B up, A down, B down), with A=(0,1/2), B=(1/2,0).
No SOC; all energies are arbitrary model units. Run from any directory.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
DELTA = 0.65
MU = -1.90
N = 301
COLORS = ("#1764ab", "#cf3b3b")


@dataclass(frozen=True)
class Model:
    name: str
    tx_a: float
    ty_a: float
    tx_b: float
    ty_b: float
    t_plus: float
    t_minus: float


MODELS = (
    Model("Mirror only", 0.55, 0.19, 0.19, 0.55, 0.36, 0.20),
    Model("C4 (original)", 0.55, 0.19, 0.19, 0.55, 0.30, 0.30),
    Model("C2 exchange", 0.55, 0.19, 0.55, 0.19, 0.36, 0.20),
    Model("C2 origin only", 0.55, 0.19, 0.40, 0.10, 0.36, 0.20),
)


def terms(kx, ky, p):
    """Return common, sublattice-odd and A-B hopping dispersions."""
    ea = -2 * p.tx_a * np.cos(kx) - 2 * p.ty_a * np.cos(ky)
    eb = -2 * p.tx_b * np.cos(kx) - 2 * p.ty_b * np.cos(ky)
    e0 = (ea + eb) / 2
    d = (ea - eb) / 2
    g = (-2 * p.t_plus * np.cos((kx + ky) / 2)
         -2 * p.t_minus * np.cos((kx - ky) / 2))
    return e0, d, g


def hamiltonian(kx, ky, p):
    e0, d, g = terms(kx, ky, p)
    h = np.zeros((4, 4), dtype=float)
    h[:2, :2] = [[e0 + d - DELTA, g], [g, e0 - d + DELTA]]
    h[2:, 2:] = [[e0 + d + DELTA, g], [g, e0 - d - DELTA]]
    return h


def energies(kx, ky, p, spin):
    e0, d, g = terms(kx, ky, p)
    radius = np.sqrt(g * g + (d - spin * DELTA) ** 2)
    return e0 - radius, e0 + radius


def path():
    points = [(0, np.pi), (0, 0), (np.pi, 0), (np.pi, np.pi), (0, 0)]
    pieces = []
    for start, end in zip(points[:-1], points[1:]):
        t = np.linspace(0, 1, 120, endpoint=False)[:, None]
        pieces.append(np.array(start)[None, :] * (1 - t) + np.array(end)[None, :] * t)
    k = np.vstack([*pieces, np.array(points[-1])[None, :]])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(k, axis=0), axis=1))]
    return k, distance, distance[[0, 120, 240, 360, -1]]


def check_symmetries():
    tau_x = np.array([[0., 1.], [1., 0.]])
    zero = np.zeros((2, 2))
    u = np.block([[zero, tau_x], [tau_x, zero]])
    samples = ((0.37, 1.21), (1.7, -0.66), (2.11, 0.42))
    transforms = {
        "Mxy": lambda x, y: (y, x),
        "C4z": lambda x, y: (-y, x),
        "C2center": lambda x, y: (-x, -y),
    }
    print("Matrix covariance errors for [spin flip || spatial operation]:")
    for p in MODELS:
        values = {}
        for label, transform in transforms.items():
            values[label] = max(
                np.max(np.abs(u @ hamiltonian(x, y, p) @ u.T
                              - hamiltonian(*transform(x, y), p)))
                for x, y in samples
            )
        plain_c2 = max(np.max(np.abs(hamiltonian(x, y, p) - hamiltonian(-x, -y, p)))
                       for x, y in samples)
        print(f"{p.name:15s}: " + ", ".join(f"{k}={v:.6g}" for k, v in values.items())
              + f", plain C2(origin)={plain_c2:.6g}")
    assert all(np.max(np.abs(u @ hamiltonian(x, y, MODELS[0]) @ u.T
                             - hamiltonian(y, x, MODELS[0]))) < 1e-12 for x, y in samples)
    assert all(np.max(np.abs(u @ hamiltonian(x, y, MODELS[1]) @ u.T
                             - hamiltonian(-y, x, MODELS[1]))) < 1e-12 for x, y in samples)
    assert all(np.max(np.abs(u @ hamiltonian(x, y, MODELS[2]) @ u.T
                             - hamiltonian(-x, -y, MODELS[2]))) < 1e-12 for x, y in samples)
    assert all(np.max(np.abs(hamiltonian(x, y, MODELS[3])
                             - hamiltonian(-x, -y, MODELS[3]))) < 1e-12 for x, y in samples)


def make_comparison():
    k, distance, ticks = path()
    q = np.linspace(-np.pi, np.pi, N)
    kx, ky = np.meshgrid(q, q)
    fig, axes = plt.subplots(2, len(MODELS), figsize=(18.2, 7.7), constrained_layout=True)
    fig.suptitle("Which symmetry constrains which hopping?  One orbital / no SOC", fontsize=15)
    for col, p in enumerate(MODELS):
        ax = axes[0, col]
        for spin, color, label in [(1, COLORS[0], "spin up"), (-1, COLORS[1], "spin down")]:
            lower, upper = energies(k[:, 0], k[:, 1], p, spin)
            ax.plot(distance, lower, color=color, lw=1.35, label=label)
            ax.plot(distance, upper, color=color, lw=1.35)
        ax.axhline(MU, color="0.3", ls="--", lw=0.85)
        for tick in ticks[1:-1]:
            ax.axvline(tick, color="0.85", lw=0.7)
        ax.set_xticks(ticks, ["Y", r"$\Gamma$", "X", "M", r"$\Gamma$"])
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_ylim(-2.65, 2.65)
        ax.set_title(p.name)
        if col == 0:
            ax.set_ylabel("Energy")
            ax.legend(frameon=False, fontsize=8, loc="upper right")

        up_lo = energies(kx, ky, p, 1)[0]
        dn_lo = energies(kx, ky, p, -1)[0]
        splitting = up_lo - dn_lo
        ax = axes[1, col]
        im = ax.imshow(splitting, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
                       vmin=-1.4, vmax=1.4, cmap="RdBu_r", interpolation="nearest")
        if np.max(np.abs(splitting)) > 1e-10:
            ax.contour(kx, ky, splitting, levels=[0], colors="black", linewidths=0.75)
        ax.set_title(r"$E_{\uparrow,-}-E_{\downarrow,-}$")
        ax.set_aspect("equal")
        ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        ax.set_xlabel(r"$k_x$")
        if col == 0:
            ax.set_ylabel(r"$k_y$")
        print(f"{p.name:15s}: max |spin splitting|={np.max(np.abs(splitting)):.6f}; "
              f"X={energies(np.pi, 0, p, 1)[0]-energies(np.pi, 0, p, -1)[0]:+.6f}; "
              f"Y={energies(0, np.pi, p, 1)[0]-energies(0, np.pi, p, -1)[0]:+.6f}")
    fig.colorbar(im, ax=axes[1, :], orientation="horizontal", shrink=0.72,
                 label="Lower-band spin splitting (model energy units)")
    dest = OUT / "symmetry_band_comparison.png"
    fig.savefig(dest, dpi=190)
    plt.close(fig)
    print(f"Saved {dest}")


def make_hopping_diagram():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(17.5, 4.5), constrained_layout=True)
    fig.suptitle("Same two-site square lattice; different spatial operations", fontsize=15)
    a = np.array([0., 0.5])
    b = np.array([0.5, 0.])
    for index, (ax, p) in enumerate(zip(axes, MODELS)):
        for gx in range(-1, 2):
            for gy in range(-1, 2):
                shift = np.array([gx, gy])
                ax.scatter(*(a + shift), s=75, c=COLORS[0], zorder=3)
                ax.scatter(*(b + shift), s=75, c=COLORS[1], zorder=3)
        for direction, val in [(np.array([1., 0.]), p.tx_a), (np.array([0., 1.]), p.ty_a)]:
            start = a
            end = a + direction
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#1764ab", lw=3)
            ax.text(*(0.5 * (start + end) + np.array([0.03, 0.04])), f"{val:.2f}", fontsize=9)
        for direction, val in [(np.array([1., 0.]), p.tx_b), (np.array([0., 1.]), p.ty_b)]:
            start = b
            end = b + direction
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#cf3b3b", lw=3)
            ax.text(*(0.5 * (start + end) + np.array([0.03, -0.11])), f"{val:.2f}", fontsize=9)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#9b59b6", lw=2.5)
        ax.plot([a[0], b[0] - 1], [a[1], b[1]], color="#2b9a74", lw=2.5)
        ax.text(0.11, 0.31, f"{p.t_minus:.2f}", fontsize=9, color="#9b59b6")
        ax.text(-0.34, 0.27, f"{p.t_plus:.2f}", fontsize=9, color="#2b9a74")
        ax.text(a[0] - 0.10, a[1] + 0.10, "A↑", color=COLORS[0], weight="bold")
        ax.text(b[0] + 0.05, b[1] - 0.07, "B↓", color=COLORS[1], weight="bold")
        if index == 0:
            ax.plot([-0.45, 1.1], [-0.45, 1.1], color="0.45", ls="--", lw=1)
            ax.text(0.63, 0.82, r"$M_{xy}$", color="0.35", fontsize=10)
        else:
            center = np.array([0.25, 0.25]) if index == 2 else np.array([0., 0.])
            ax.scatter(*center, marker="*", s=120, color="black", zorder=5)
            ax.text(*(center + np.array([0.07, -0.17])),
                    "$C_4$ center" if index == 1 else "$C_2$ center", fontsize=9)
        ax.set_xlim(-0.7, 1.1)
        ax.set_ylim(-0.45, 1.15)
        ax.set_aspect("equal")
        ax.set_title(p.name)
        ax.grid(alpha=0.15)
        ax.set_xticks([])
        ax.set_yticks([])
    dest = OUT / "symmetry_hopping_layout.png"
    fig.savefig(dest, dpi=190)
    plt.close(fig)
    print(f"Saved {dest}")


def make_fermi_surfaces():
    """Plot lower-band constant-energy contours at the fixed illustrative mu."""
    q = np.linspace(-np.pi, np.pi, 501)
    kx, ky = np.meshgrid(q, q)
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 9.2), constrained_layout=True)
    fig.suptitle(r"Spin-resolved lower-band contours at $\mu=-1.90$", fontsize=15)
    for ax, p in zip(axes.flat, MODELS):
        up = energies(kx, ky, p, 1)[0]
        down = energies(kx, ky, p, -1)[0]
        degenerate = np.max(np.abs(up - down)) < 1e-12
        if degenerate:
            ax.contour(kx, ky, up, levels=[MU], colors=["#303030"],
                       linewidths=2.0, linestyles="solid")
            ax.plot([], [], color="#303030", lw=2, label="up = down")
        else:
            ax.contour(kx, ky, up, levels=[MU], colors=[COLORS[0]],
                       linewidths=1.7, linestyles="solid")
            ax.contour(kx, ky, down, levels=[MU], colors=[COLORS[1]],
                       linewidths=1.7, linestyles="solid")
            ax.plot([], [], color=COLORS[0], label="spin up")
            ax.plot([], [], color=COLORS[1], label="spin down")
        ax.axline((0, 0), slope=1, color="0.82", lw=0.75, ls=":")
        ax.axline((0, 0), slope=-1, color="0.82", lw=0.75, ls=":")
        ax.set_title(p.name)
        ax.set_aspect("equal")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    dest = OUT / "symmetry_fermi_surfaces.png"
    fig.savefig(dest, dpi=220)
    plt.close(fig)
    print(f"Saved {dest}")


if __name__ == "__main__":
    check_symmetries()
    make_comparison()
    make_hopping_diagram()
    make_fermi_surfaces()
