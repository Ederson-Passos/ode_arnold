"""
Scientific Visualization: Small Oscillations (Simple Harmonic Oscillator) in V. I. Arnold's Formalism
=====================================================================================================
Based on Vladimir I. Arnold's "Ordinary Differential Equations" (Chapter 1, Section 1.1.15).
Styled in a 1940s–1950s archival scientific monograph palette adhering to Edward Tufte's
principles of analytical design (high data-ink ratio, despined axes, transparent 3D panes,
and zero chartjunk).

The autonomous dynamical system for small harmonic oscillations (normalized frequency omega_0 = 1) is:
    dx_1 / dt =  x_2
    dx_2 / dt = -x_1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


FloatArray = NDArray[np.float64]

# =============================================================================
# ARCHIVAL 1940s–1950s PAPER & TUFTE INK PALETTE
# =============================================================================
ARCHIVAL_PAPER_BG: str = "#F5EFE0"
INK_PRIMARY: str = "#191715"
INK_SECONDARY: str = "#47423B"
INK_MUTED: str = "#787064"
RULE_HAIRLINE: str = "#C8BFA8"

INK_COBALT: str = "#1B4965"     # Prussian / Cobalt Blue (R = 1 & Phase Projection)
INK_VERMILION: str = "#9E2A2B"  # Carmine / Vermilion Red (R = 2 & 3D Helix gamma(t))
INK_FOREST: str = "#2D5A27"     # Olive / Forest Green (R = 3)
INK_OCHRE: str = "#8C5A16"      # Warm Sepia Ochre (Invariant Cylinder Manifold)


# =============================================================================
# 1. DATA STRUCTURES & CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class TrajectorySolution:
    """Encapsulates a numerically integrated trajectory of the dynamical system."""

    t: FloatArray
    x1: FloatArray
    x2: FloatArray
    initial_state: Tuple[float, float]
    energy: float

    @property
    def radius(self) -> float:
        """Phase-space orbital radius R = sqrt(x_1(0)^2 + x_2(0)^2)."""
        return float(np.hypot(self.initial_state[0], self.initial_state[1]))


@dataclass(frozen=True)
class VectorFieldGrid:
    """Encapsulates the 2D phase-space mesh and phase velocity components."""

    X1: FloatArray
    X2: FloatArray
    U: FloatArray
    V: FloatArray
    magnitude: FloatArray


@dataclass(frozen=True)
class OscillatorConfig:
    """Configuration parameters for numerical integration and visualization."""

    omega: float = 1.0
    phase_radii: Tuple[float, ...] = (1.0, 2.0, 3.0)
    phase_t_span: Tuple[float, float] = (0.0, 2.0 * np.pi)
    extended_initial_state: Tuple[float, float] = (2.0, 0.0)
    extended_t_span: Tuple[float, float] = (0.0, 4.0 * np.pi)
    grid_bounds: Tuple[float, float] = (-3.6, 3.6)
    quiver_density: int = 17
    stream_density: int = 120
    num_eval_points_2d: int = 500
    num_eval_points_3d: int = 800
    rtol: float = 1e-9
    atol: float = 1e-11
    solver_method: str = "DOP853"
    orbit_colors: Tuple[str, ...] = field(
        default_factory=lambda: (INK_COBALT, INK_VERMILION, INK_FOREST)
    )


# =============================================================================
# 2. MATHEMATICAL MODEL & NUMERICAL INTEGRATION LOGIC
# =============================================================================
class HarmonicOscillatorModel:
    """
    Mathematical model and numerical integrator for the autonomous harmonic oscillator:
        dx_1 / dt =  x_2
        dx_2 / dt = -omega^2 * x_1
    where omega = 1.0 yields Arnold's canonical system dx_1/dt = x_2, dx_2/dt = -x_1.
    """

    def __init__(self, config: OscillatorConfig | None = None) -> None:
        self.config: OscillatorConfig = config or OscillatorConfig()

    def vector_field(self, t: float, state: Sequence[float] | FloatArray) -> FloatArray:
        """
        Evaluates the autonomous phase velocity vector field V(x_1, x_2) = (x_2, -omega^2 * x_1).
        """
        x1, x2 = state[0], state[1]
        dx1_dt: float = float(x2)
        dx2_dt: float = float(-(self.config.omega**2) * x1)
        return np.array([dx1_dt, dx2_dt], dtype=np.float64)

    def mechanical_energy(self, x1: float | FloatArray, x2: float | FloatArray) -> float | FloatArray:
        """
        Computes the first integral (total mechanical energy) E(x_1, x_2) = 0.5 * (omega^2 * x_1^2 + x_2^2).
        """
        return 0.5 * ((self.config.omega**2) * np.square(x1) + np.square(x2))

    def compute_vector_field_grid(
        self,
        bounds: Tuple[float, float] | None = None,
        resolution: int | None = None,
    ) -> VectorFieldGrid:
        """
        Generates a 2D meshgrid on R^2 and computes the phase velocity vector field components.
        """
        b_min, b_max = bounds or self.config.grid_bounds
        res: int = resolution or self.config.quiver_density

        x1_lin: FloatArray = np.linspace(b_min, b_max, res, dtype=np.float64)
        x2_lin: FloatArray = np.linspace(b_min, b_max, res, dtype=np.float64)
        X1, X2 = np.meshgrid(x1_lin, x2_lin)

        U: FloatArray = X2.copy()
        V: FloatArray = -(self.config.omega**2) * X1
        magnitude: FloatArray = np.hypot(U, V)

        return VectorFieldGrid(X1=X1, X2=X2, U=U, V=V, magnitude=magnitude)

    def solve_trajectory(
        self,
        initial_state: Tuple[float, float],
        t_span: Tuple[float, float],
        num_points: int = 500,
    ) -> TrajectorySolution:
        """
        Solves the Initial Value Problem (IVP) for a given initial condition using SciPy's solve_ivp.
        """
        t_eval: FloatArray = np.linspace(t_span[0], t_span[1], num_points, dtype=np.float64)
        y0: FloatArray = np.array(initial_state, dtype=np.float64)

        sol = solve_ivp(
            fun=self.vector_field,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=self.config.solver_method,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )

        if not sol.success:
            raise RuntimeError(f"Numerical IVP integration failed: {sol.message}")

        x1_sol: FloatArray = sol.y[0].astype(np.float64)
        x2_sol: FloatArray = sol.y[1].astype(np.float64)
        energy_val: float = float(self.mechanical_energy(initial_state[0], initial_state[1]))

        return TrajectorySolution(
            t=sol.t.astype(np.float64),
            x1=x1_sol,
            x2=x2_sol,
            initial_state=initial_state,
            energy=energy_val,
        )

    def solve_concentric_phase_curves(self) -> Tuple[TrajectorySolution, ...]:
        """
        Solves the IVP for all configured phase radii R in config.phase_radii.
        """
        solutions: list[TrajectorySolution] = []
        for radius in self.config.phase_radii:
            sol = self.solve_trajectory(
                initial_state=(radius, 0.0),
                t_span=self.config.phase_t_span,
                num_points=self.config.num_eval_points_2d,
            )
            solutions.append(sol)
        return tuple(solutions)

    def solve_extended_helix(self) -> TrajectorySolution:
        """
        Solves the IVP for the 3D extended phase space integral curve over t in [0, 4*pi].
        """
        return self.solve_trajectory(
            initial_state=self.config.extended_initial_state,
            t_span=self.config.extended_t_span,
            num_points=self.config.num_eval_points_3d,
        )


# =============================================================================
# 3. TUFTE MINIMALIST & 1940s–1950s ARCHIVAL PLOTTING LOGIC
# =============================================================================
def plot_phase_space_2d(
    ax: Axes,
    quiver_grid: VectorFieldGrid,
    stream_grid: VectorFieldGrid,
    phase_curves: Sequence[TrajectorySolution],
    orbit_colors: Sequence[str],
    bounds: Tuple[float, float],
) -> None:
    """
    Renders Fig. 1a: 2D Phase Space (x_1 vs x_2) adhering to Edward Tufte's high data-ink ratio.
    Uses fine sepia streamlines, monochrome drafting-ink velocity vectors (magnitude encoded
    directly by vector length without redundant colorbar chartjunk), concentric phase orbits
    with direct inline labels, and despined range-frame axes.
    """
    ax.set_facecolor(ARCHIVAL_PAPER_BG)

    # 1. Ultra-fine background flow streamlines in diluted sepia ink
    ax.streamplot(
        stream_grid.X1,
        stream_grid.X2,
        stream_grid.U,
        stream_grid.V,
        color="#D7CEBC",
        linewidth=0.6,
        density=1.0,
        arrowsize=0.65,
        zorder=1,
    )

    # 2. Monochrome drafting-ink velocity vectors (length encodes ||V|| = sqrt(x_1^2 + x_2^2))
    ax.quiver(
        quiver_grid.X1,
        quiver_grid.X2,
        quiver_grid.U,
        quiver_grid.V,
        color=INK_SECONDARY,
        alpha=0.52,
        angles="xy",
        scale_units="xy",
        scale=8.2,
        width=0.0032,
        headwidth=3.4,
        headlength=4.2,
        zorder=2,
    )

    # 3. Concentric circular phase curves from solve_ivp + Tufte direct labeling
    for idx, sol in enumerate(phase_curves):
        color = orbit_colors[idx % len(orbit_colors)]
        r_val = sol.radius
        ax.plot(
            sol.x1,
            sol.x2,
            color=color,
            linewidth=2.1,
            label=rf"Orbit $R = {r_val:.0f}$ ($E = {sol.energy:.1f}$)",
            zorder=4,
        )
        # Initial condition node (R, 0)
        ax.scatter(
            [sol.initial_state[0]],
            [sol.initial_state[1]],
            color=color,
            edgecolor=ARCHIVAL_PAPER_BG,
            s=46,
            linewidth=1.1,
            zorder=5,
        )
        # Clockwise flow direction arrowheads at theta = pi/4 and 5*pi/4
        for angle_idx in (len(sol.t) // 8, (5 * len(sol.t)) // 8):
            ax.annotate(
                "",
                xy=(sol.x1[angle_idx + 3], sol.x2[angle_idx + 3]),
                xytext=(sol.x1[angle_idx], sol.x2[angle_idx]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.7, mutation_scale=12),
                zorder=5,
            )
        # Direct Tufte label along the upper vertical axis (x_1 = 0, x_2 = +R)
        ax.text(
            0.08,
            r_val + 0.11,
            rf"$R={r_val:.0f}$",
            color=color,
            fontsize=9.5,
            bbox=dict(boxstyle="square,pad=0.1", facecolor=ARCHIVAL_PAPER_BG, edgecolor="none", alpha=0.85),
            zorder=6,
        )

    # 4. Lyapunov-stable center equilibrium (0, 0) as a classic hollow technical fixed-point ring
    ax.scatter(
        [0.0],
        [0.0],
        facecolor=ARCHIVAL_PAPER_BG,
        edgecolor=INK_PRIMARY,
        marker="o",
        s=90,
        linewidth=1.8,
        zorder=6,
        label=r"Equilibrium $(0, 0)$",
    )

    # 5. Tufte Despined Axes, Hairline Origin Crosshairs, and Minimalist Grid
    ax.axhline(0.0, color=INK_MUTED, linestyle="-", linewidth=0.7, alpha=0.55, zorder=1)
    ax.axvline(0.0, color=INK_MUTED, linestyle="-", linewidth=0.7, alpha=0.55, zorder=1)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect("equal", adjustable="box")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_SECONDARY)
    ax.spines["bottom"].set_color(INK_SECONDARY)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)

    ax.tick_params(direction="out", length=4, width=0.8, colors=INK_SECONDARY, labelsize=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, color=RULE_HAIRLINE, alpha=0.65)

    ax.set_title(
        r"$\bf{Fig.\ 1a.\ \ Phase\ Plane\ \mathbb{R}^2\ (x_1,\ x_2)}$" + "\n"
        r"Velocity Field $V(x_1, x_2) = (x_2, -x_1)$ & Concentric Level Curves $x_1^2 + x_2^2 = R^2$",
        fontsize=11.5,
        color=INK_PRIMARY,
        pad=14,
    )
    ax.set_xlabel(r"Position Coordinate $x_1$", fontsize=11, color=INK_PRIMARY, labelpad=8)
    ax.set_ylabel(r"Velocity Coordinate $x_2 = \dot{x}_1$", fontsize=11, color=INK_PRIMARY, labelpad=8)
    ax.legend(
        loc="lower left",
        fontsize=8.8,
        frameon=True,
        facecolor=ARCHIVAL_PAPER_BG,
        framealpha=0.88,
        edgecolor="none",
        labelcolor=INK_PRIMARY,
    )


def plot_extended_phase_space_3d(
    ax: Axes,
    helix_sol: TrajectorySolution,
    bounds: Tuple[float, float],
) -> None:
    """
    Renders Fig. 1b: 3D Extended Phase Space R x R^2 (t, x_1, x_2) with transparent
    archival panes, sepia invariant energy cylinder, Prussian blue t = 0 shadow projection,
    and Carmine/Vermilion cylindrical helix gamma(t) over t in [0, 4*pi].
    """
    ax.set_facecolor(ARCHIVAL_PAPER_BG)

    # Remove 3D gray box chartjunk: make panes transparent with subtle hairline edges
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(RULE_HAIRLINE)
        axis._axinfo["grid"]["color"] = (0.78, 0.75, 0.66, 0.45)
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["linewidth"] = 0.65

    t_min, t_max = float(helix_sol.t[0]), float(helix_sol.t[-1])
    radius: float = helix_sol.radius

    # 1. Subtle sepia invariant energy cylinder x_1^2 + x_2^2 = R^2 across t in [0, 4*pi]
    t_cyl: FloatArray = np.linspace(t_min, t_max, 50, dtype=np.float64)
    theta_cyl: FloatArray = np.linspace(0.0, 2.0 * np.pi, 60, dtype=np.float64)
    T_mesh, Theta_mesh = np.meshgrid(t_cyl, theta_cyl)
    X1_cyl: FloatArray = radius * np.cos(Theta_mesh)
    X2_cyl: FloatArray = -radius * np.sin(Theta_mesh)

    ax.plot_surface(
        T_mesh,
        X1_cyl,
        X2_cyl,
        color=INK_OCHRE,
        alpha=0.09,
        edgecolor="none",
        antialiased=True,
    )

    # 2. Equilibrium worldline (t, 0, 0) along the central axis of the cylinder
    ax.plot(
        [t_min, t_max],
        [0.0, 0.0],
        [0.0, 0.0],
        color=INK_SECONDARY,
        linestyle="-.",
        linewidth=1.3,
        alpha=0.8,
        label=r"Equilibrium Axis $(t, 0, 0)$",
    )

    # 3. Orthogonal phase-plane shadow on the wall t = 0 in Prussian Cobalt Ink
    ax.plot(
        np.zeros_like(helix_sol.t),
        helix_sol.x1,
        helix_sol.x2,
        color=INK_COBALT,
        linestyle="--",
        linewidth=1.7,
        alpha=0.85,
        label=r"Phase Projection ($t = 0$)",
    )

    # 4. Orthogonal tie-lines from key period points on the 3D helix to the t = 0 plane
    key_t_vals: FloatArray = np.array([0.0, np.pi, 2.0 * np.pi, 3.0 * np.pi, 4.0 * np.pi])
    key_x1_vals: FloatArray = radius * np.cos(key_t_vals)
    key_x2_vals: FloatArray = -radius * np.sin(key_t_vals)

    for tk, x1k, x2k in zip(key_t_vals, key_x1_vals, key_x2_vals):
        ax.plot(
            [float(tk), 0.0],
            [float(x1k), float(x1k)],
            [float(x2k), float(x2k)],
            color=INK_MUTED,
            linestyle=":",
            linewidth=0.95,
            alpha=0.7,
        )

    # 5. Primary 3D Integral Curve gamma(t) = (t, x_1(t), x_2(t)) in Vermilion Ink
    ax.plot(
        helix_sol.t,
        helix_sol.x1,
        helix_sol.x2,
        color=INK_VERMILION,
        linewidth=2.7,
        label=rf"Integral Curve $\gamma(t)$ from $[{helix_sol.initial_state[0]:.0f}, {helix_sol.initial_state[1]:.0f}]$",
    )

    # 6. Key temporal markers along the helix (t = 0, pi, 2*pi, 3*pi, 4*pi)
    ax.scatter(
        key_t_vals,
        key_x1_vals,
        key_x2_vals,
        color=ARCHIVAL_PAPER_BG,
        edgecolor=INK_PRIMARY,
        s=44,
        linewidth=1.5,
        depthshade=False,
        zorder=6,
        label=r"Half-Period Nodes ($t = k\pi$)",
    )

    # 7. Rigorous LaTeX axes labels, pi-ticks, and 3D camera perspective
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(bounds)
    ax.set_zlim(bounds)

    ax.set_xticks(key_t_vals)
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"], fontsize=10, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9.5)

    ax.set_xlabel(r"Time $t$", fontsize=11, color=INK_PRIMARY, labelpad=10)
    ax.set_ylabel(r"Position $x_1$", fontsize=11, color=INK_PRIMARY, labelpad=10)
    ax.set_zlabel(r"Velocity $x_2$", fontsize=11, color=INK_PRIMARY, labelpad=8)

    ax.set_title(
        r"$\bf{Fig.\ 1b.\ \ Extended\ Phase\ Space\ \mathbb{R} \times \mathbb{R}^2\ (t,\ x_1,\ x_2)}$" + "\n"
        r"Cylindrical Helix $\gamma(t) = (t,\ R\cos t,\ -R\sin t)$ for $t \in [0,\ 4\pi]$ ($R = 2$)",
        fontsize=11.5,
        color=INK_PRIMARY,
        y=1.03,
        pad=0,
    )

    ax.view_init(elev=24, azim=-58)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.94),
        fontsize=8.8,
        frameon=True,
        facecolor=ARCHIVAL_PAPER_BG,
        framealpha=0.88,
        edgecolor="none",
        labelcolor=INK_PRIMARY,
    )


def build_matplotlib_figure(config: OscillatorConfig | None = None) -> Figure:
    """
    Orchestrates numerical integration and constructs the archival Tufte dual-subplot figure.
    """
    cfg: OscillatorConfig = config or OscillatorConfig()
    model = HarmonicOscillatorModel(cfg)

    quiver_grid: VectorFieldGrid = model.compute_vector_field_grid(
        bounds=cfg.grid_bounds, resolution=cfg.quiver_density
    )
    stream_grid: VectorFieldGrid = model.compute_vector_field_grid(
        bounds=cfg.grid_bounds, resolution=cfg.stream_density
    )
    phase_curves: Tuple[TrajectorySolution, ...] = model.solve_concentric_phase_curves()
    helix_sol: TrajectorySolution = model.solve_extended_helix()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.edgecolor": INK_SECONDARY,
            "axes.linewidth": 0.85,
            "text.color": INK_PRIMARY,
        }
    )
    fig: Figure = plt.figure(figsize=(17.5, 8.6), dpi=300, facecolor=ARCHIVAL_PAPER_BG)
    fig.suptitle(
        r"$\bf{Dynamical\ Systems\ and\ Ordinary\ Differential\ Equations\ (V.\ I.\ Arnold\ —\ Section\ 1.1.15)}$"
        + "\n"
        + r"Simple Harmonic Oscillator (Small Oscillations): $\dot{x}_1 = x_2,\; \dot{x}_2 = -x_1$"
        + r"   $\bullet$   First Integral: $E(x_1, x_2) = \frac{1}{2}(x_1^2 + x_2^2) = \mathrm{const.}$",
        fontsize=13.5,
        color=INK_PRIMARY,
        y=0.97,
    )

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.16)
    ax_2d: Axes = fig.add_subplot(gs[0, 0])
    ax_3d: Axes = fig.add_subplot(gs[0, 1], projection="3d")

    plot_phase_space_2d(
        ax=ax_2d,
        quiver_grid=quiver_grid,
        stream_grid=stream_grid,
        phase_curves=phase_curves,
        orbit_colors=cfg.orbit_colors,
        bounds=cfg.grid_bounds,
    )
    plot_extended_phase_space_3d(
        ax=ax_3d,
        helix_sol=helix_sol,
        bounds=cfg.grid_bounds,
    )

    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.08, top=0.81, wspace=0.16)
    return fig


# =============================================================================
# 4. MAIN EXECUTION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys
    import webbrowser

    script_dir = Path(__file__).resolve().parent
    pictures_dir = script_dir.parent / "pictures"
    pictures_dir.mkdir(parents=True, exist_ok=True)

    output_path = pictures_dir / "small_oscillations_arnold.png"
    figure = build_matplotlib_figure()
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"[1/2] Saved publication-ready figure to: '{output_path}'")

    interactive_html_path = script_dir / "small_oscillations_arnold_interactive.html"
    if interactive_html_path.exists() and "--no-browser" not in sys.argv:
        print(f"[2/2] Launching interactive 60 FPS animation in browser: '{interactive_html_path}'")
        webbrowser.open(interactive_html_path.as_uri())
