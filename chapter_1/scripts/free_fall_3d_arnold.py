"""
Scientific 3D Visualization: Free Fall in V. I. Arnold's Formalism (Archival Tufte Edition)
===========================================================================================
Based on Vladimir I. Arnold's "Ordinary Differential Equations" (Chapter 1, Section 1.1.14).
Styled in a 1940s–1950s archival scientific monograph palette adhering to Edward Tufte's
principles of analytical design (high data-ink ratio, transparent 3D panes, muted surfaces,
and zero chartjunk).

This script constructs a side-by-side dual-panel 3D visualization:
- Fig. 1a: The Extended Phase Space R x R^2 with axes (t, x_1, x_2).
  Features the invariant energy manifold (parabolic cylinder),
  the three-dimensional integral curve gamma(t), the extended vector field V_tilde(t, x_1, x_2),
  the phase plane projection on the lateral wall t = 0 with the autonomous vector field V(x_1, x_2),
  and orthogonal projection drop lines connecting the 3D trajectory to its shadow.

- Fig. 1b: The First Integral Relief (Energy Surface) with axes (x_1, x_2, E).
  Features the energy surface E(x_1, x_2) = (1/2) m x_2^2 + m g x_1, the horizontal cutting plane
  at E = E_0, the 3D intersection level curve, and the projection onto the base plane
  demonstrating geometrically that phase curves are level curves of mechanical energy.

Outputs:
1. Interactive HTML 3D visualization via Plotly: 'scripts/free_fall_arnold_3d.html'
2. High-resolution publication-quality image via Matplotlib: 'pictures/free_fall_arnold_3d.png'
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from plotly.subplots import make_subplots


# =============================================================================
# ARCHIVAL 1940s–1950s PAPER & TUFTE INK PALETTE
# =============================================================================
ARCHIVAL_PAPER_BG: str = "#F5EFE0"
INK_PRIMARY: str = "#191715"
INK_SECONDARY: str = "#47423B"
INK_MUTED: str = "#787064"
RULE_HAIRLINE: str = "#C8BFA8"

INK_COBALT: str = "#1B4965"     # Prussian / Cobalt Blue (Phase Plane Projections)
INK_VERMILION: str = "#9E2A2B"  # Carmine / Vermilion Red (3D Integral & Level Curves)
INK_FOREST: str = "#2D5A27"     # Olive / Forest Green (Autonomous Phase Vector Field V)
INK_OCHRE: str = "#8C5A16"      # Warm Sepia Ochre (Extended Field V_tilde & Cutting Plane)

ARCHIVAL_CMAP = LinearSegmentedColormap.from_list(
    "archival_sepia", ["#E6DEC8", "#C8B998", "#8C5A16", "#1B4965"]
)


# =============================================================================
# 1. PHYSICAL PARAMETERS AND INITIAL CONDITIONS
# =============================================================================
g = 9.8        # Gravitational acceleration (m/s^2)
m = 1.0        # Particle mass (kg)
x0 = 0.0       # Initial position x_1(0) (m)
v0 = 14.0      # Initial velocity x_2(0) (m/s)
t_min, t_max = 0.0, 3.0  # Time interval t in [0, 3] s

# Constant mechanical energy (First Integral)
E0 = 0.5 * m * (v0 ** 2) + m * g * x0  # 98.0 Joules

print("=" * 70)
print("FREE FALL DYNAMICAL SYSTEM (V. I. ARNOLD)")
print("=" * 70)
print(f"Mass (m):                       {m:.1f} kg")
print(f"Gravity (g):                    {g:.2f} m/s^2")
print(f"Initial position x_1(0):        {x0:.2f} m")
print(f"Initial velocity x_2(0):        {v0:.2f} m/s")
print(f"Mechanical Energy (E_0):        {E0:.2f} J")
print(f"Time interval:                  t in [{t_min}, {t_max}] s")
print(f"Theoretical max height (x_2=0): {E0 / (m * g):.2f} m")
print(f"Time to apex:                   {v0 / g:.4f} s")
print("=" * 70)


# =============================================================================
# 2. MATHEMATICAL MESHES AND DATA
# =============================================================================
# Analytical trajectory / Integral curve gamma(t)
t_curve = np.linspace(t_min, t_max, 250)
x1_curve = -0.5 * g * (t_curve ** 2) + v0 * t_curve + x0
x2_curve = -g * t_curve + v0

# Invariant Energy Manifold in Extended Space (Parabolic Cylinder)
# x_1(t, x_2) = E0/(mg) - x_2^2/(2g)
t_surf = np.linspace(t_min, t_max, 40)
x2_surf = np.linspace(-16.0, 16.0, 50)
T_mesh, X2_mesh = np.meshgrid(t_surf, x2_surf)
X1_mesh = (E0 / (m * g)) - (X2_mesh ** 2) / (2.0 * g)

# Orthogonal projection points connecting 3D curve to t = 0 wall (dotted tie lines)
t_proj_pts = np.array([0.0, 0.5, 1.0, 1.4286, 2.0, 2.5, 3.0])
x1_proj_pts = -0.5 * g * (t_proj_pts ** 2) + v0 * t_proj_pts + x0
x2_proj_pts = -g * t_proj_pts + v0

# Extended 3D vector field V_tilde = (1, x_2, -g) on the cylinder surface
t_vec_grid = np.linspace(0.4, 2.6, 4)
x2_vec_grid = np.linspace(-12.0, 12.0, 5)
T_vec, X2_vec = np.meshgrid(t_vec_grid, x2_vec_grid)
T_vec = T_vec.flatten()
X2_vec = X2_vec.flatten()
X1_vec = (E0 / (m * g)) - (X2_vec ** 2) / (2.0 * g)

U_t = np.ones_like(T_vec)
U_x1 = X2_vec
U_x2 = -g * np.ones_like(T_vec)
# Normalization for visually balanced arrow rendering
norm_3d = np.sqrt(U_t**2 + U_x1**2 + U_x2**2)
scale_3d = 0.5
U_t_n = (U_t / norm_3d) * scale_3d
U_x1_n = (U_x1 / norm_3d) * scale_3d * 2.5
U_x2_n = (U_x2 / norm_3d) * scale_3d * 2.5

# 2D grid for autonomous vector field V = (x_2, -g) on the t = 0 wall
x1_grid_2d = np.linspace(-4.0, 10.0, 5)
x2_grid_2d = np.linspace(-15.0, 15.0, 7)
X1_g2d, X2_g2d = np.meshgrid(x1_grid_2d, x2_grid_2d)
X1_g2d = X1_g2d.flatten()
X2_g2d = X2_g2d.flatten()
T_g2d = np.zeros_like(X1_g2d)

Vx1 = X2_g2d
Vx2 = -g * np.ones_like(X1_g2d)
norm_2d = np.sqrt(Vx1**2 + Vx2**2)
norm_2d[norm_2d == 0] = 1.0
scale_2d = 1.6
Vx1_n = (Vx1 / norm_2d) * scale_2d
Vx2_n = (Vx2 / norm_2d) * scale_2d

# Mechanical Energy Surface E(x_1, x_2) = (1/2) m x_2^2 + m g x_1 for Panel 2
x1_e_grid = np.linspace(-6.0, 12.0, 45)
x2_e_grid = np.linspace(-18.0, 18.0, 45)
X1_E, X2_E = np.meshgrid(x1_e_grid, x2_e_grid)
Energy_Surface = 0.5 * m * (X2_E ** 2) + m * g * X1_E

# Constant Energy Plane E = E0
Plane_E0 = np.full_like(Energy_Surface, E0)

# Intersection 3D level curve at E = E0
x2_iso = np.linspace(-16.0, 16.0, 200)
x1_iso = (E0 / (m * g)) - (x2_iso ** 2) / (2.0 * g)
E_iso = np.full_like(x2_iso, E0)

# Projection onto base plane E = E_base (using E_base = 0 J for clear vertical separation)
E_base = 0.0
E_proj_base = np.full_like(x2_iso, E_base)

# Sample indices for vertical drop lines connecting 3D level curve to base projection
sample_indices = np.linspace(15, len(x2_iso) - 15, 8, dtype=int)

# Autonomous phase vector field on base plane (E = E_base)
x1_eb_grid = np.linspace(-5.0, 10.0, 5)
x2_eb_grid = np.linspace(-15.0, 15.0, 7)
X1_EB, X2_EB = np.meshgrid(x1_eb_grid, x2_eb_grid)
X1_EB = X1_EB.flatten()
X2_EB = X2_EB.flatten()
E_EB = np.full_like(X1_EB, E_base)
EB_Vx1 = X2_EB
EB_Vx2 = -g * np.ones_like(X1_EB)
norm_eb = np.sqrt(EB_Vx1**2 + EB_Vx2**2)
norm_eb[norm_eb == 0] = 1.0
EB_Vx1_n = (EB_Vx1 / norm_eb) * scale_2d
EB_Vx2_n = (EB_Vx2 / norm_eb) * scale_2d


# =============================================================================
# 3. CONSTRUCTION OF INTERACTIVE PLOTLY VISUALIZATION (ARCHIVAL TUFTE STYLE)
# =============================================================================
def build_plotly_figure():
    """Creates 3D interactive dual-panel visualization in 1940s–50s archival Tufte style."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            "<b>Fig. 1a. Extended Phase Space ℝ × ℝ² (t, x₁, x₂)</b><br>"
            "<sup><i>Parabolic Cylinder Manifold, Integral Curve γ(t) & Projection onto t = 0</i></sup>",
            "<b>Fig. 1b. First Integral Relief (x₁, x₂, E)</b><br>"
            "<sup><i>Mechanical Energy Surface E(x₁, x₂), Cutting Plane E = E₀ & Base Level Curve</i></sup>"
        ),
        horizontal_spacing=0.03
    )

    # Style subplot title annotations in classic monograph serif
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(
            family="EB Garamond, Palatino Linotype, Georgia, serif",
            size=15,
            color=INK_PRIMARY
        )

    # -------------------------------------------------------------------------
    # PANEL 1: EXTENDED PHASE SPACE (t, x_1, x_2)
    # -------------------------------------------------------------------------

    # 1.1 Constant Energy Invariant Surface (Parabolic Cylinder in Muted Sepia Wash)
    fig.add_trace(
        go.Surface(
            x=T_mesh, y=X1_mesh, z=X2_mesh,
            colorscale=[
                [0.0, 'rgba(140, 90, 22, 0.14)'],
                [1.0, 'rgba(27, 73, 101, 0.18)']
            ],
            showscale=False,
            name="Invariant Manifold x₁ = E₀/mg − x₂²/(2g)",
            legendgroup="p1_surf",
            showlegend=True,
            hoverinfo="text",
            hovertext="Invariant Manifold E₀ = 98 J (Parabolic Cylinder)"
        ),
        row=1, col=1
    )

    # 1.2 Projection on lateral wall t = 0 (Phase Parabola in Prussian Cobalt Ink)
    fig.add_trace(
        go.Scatter3d(
            x=np.zeros_like(x1_curve), y=x1_curve, z=x2_curve,
            mode='lines',
            line=dict(color=INK_COBALT, width=6),
            name="Phase Projection (t = 0)",
            legendgroup="p1_proj",
            showlegend=True,
            hoverinfo="text",
            hovertext="Projected Phase Orbit at t = 0: x₁ = 10 − x₂²/19.6"
        ),
        row=1, col=1
    )

    # 1.3 3D Integral Curve gamma(t) in Carmine Vermilion Ink
    fig.add_trace(
        go.Scatter3d(
            x=t_curve, y=x1_curve, z=x2_curve,
            mode='lines',
            line=dict(color=INK_VERMILION, width=7),
            name="Integral Curve γ(t) = (t, x₁(t), x₂(t))",
            legendgroup="p1_gamma",
            showlegend=True,
            hoverinfo="text",
            hovertext=[
                f"t = {t:.2f} s | x₁ = {x1:.2f} m | x₂ = {x2:.2f} m/s"
                for t, x1, x2 in zip(t_curve, x1_curve, x2_curve)
            ]
        ),
        row=1, col=1
    )

    # Key trajectory nodes (Classic Hollow Technical Rings)
    key_t = np.array([0.0, v0 / g, 2 * v0 / g, 3.0])
    key_x1 = -0.5 * g * (key_t ** 2) + v0 * key_t + x0
    key_x2 = -g * key_t + v0
    fig.add_trace(
        go.Scatter3d(
            x=key_t, y=key_x1, z=key_x2,
            mode='markers+text',
            marker=dict(
                size=5.5,
                color=ARCHIVAL_PAPER_BG,
                line=dict(color=INK_PRIMARY, width=2)
            ),
            text=[" t=0", " Apex (x₂=0)", " Return", " t=3s"],
            textfont=dict(family="EB Garamond, Georgia, serif", size=12, color=INK_PRIMARY),
            textposition="top right",
            name="Characteristic Nodes on γ(t)",
            legendgroup="p1_pts",
            showlegend=True,
            hoverinfo="text",
            hovertext=[
                "Start: t=0 s, x₁=0 m, x₂=+14 m/s",
                "Apex: t=1.43 s, x₁=10 m, x₂=0 m/s",
                "Return to base: t=2.86 s, x₁=0 m, x₂=−14 m/s",
                "End: t=3.0 s, x₁=−2.1 m, x₂=−15.4 m/s"
            ]
        ),
        row=1, col=1
    )

    # 1.4 Orthogonal projection drop lines (gamma(t) -> wall t = 0)
    tie_x, tie_y, tie_z = [], [], []
    for tk, x1k, x2k in zip(t_proj_pts, x1_proj_pts, x2_proj_pts):
        tie_x.extend([tk, 0.0, None])
        tie_y.extend([x1k, x1k, None])
        tie_z.extend([x2k, x2k, None])

    fig.add_trace(
        go.Scatter3d(
            x=tie_x, y=tie_y, z=tie_z,
            mode='lines',
            line=dict(color='rgba(120, 112, 100, 0.72)', width=2, dash='dot'),
            name="Orthogonal Tie-Lines (t → 0)",
            legendgroup="p1_tie",
            showlegend=True,
            hoverinfo="none"
        ),
        row=1, col=1
    )

    # 1.5 Extended 3D vector field V_tilde = (1, x_2, -g) on surface in Warm Ochre Ink
    fig.add_trace(
        go.Cone(
            x=T_vec, y=X1_vec, z=X2_vec,
            u=U_t_n, v=U_x1_n, w=U_x2_n,
            sizemode="absolute",
            sizeref=1.15,
            anchor="tail",
            colorscale=[[0, INK_OCHRE], [1, INK_OCHRE]],
            showscale=False,
            name="Extended Field Ṽ = (1, x₂, −g)",
            legendgroup="p1_cone",
            showlegend=True,
            hoverinfo="text",
            hovertext="Extended Tangent Vector Ṽ(t, x₁, x₂) = (1, x₂, −g)"
        ),
        row=1, col=1
    )

    shaft_x, shaft_y, shaft_z = [], [], []
    for tx, x1x, x2x, ut, ux1, ux2 in zip(T_vec, X1_vec, X2_vec, U_t_n, U_x1_n, U_x2_n):
        shaft_x.extend([tx, tx + ut, None])
        shaft_y.extend([x1x, x1x + ux1, None])
        shaft_z.extend([x2x, x2x + ux2, None])
    fig.add_trace(
        go.Scatter3d(
            x=shaft_x, y=shaft_y, z=shaft_z,
            mode='lines',
            line=dict(color=INK_OCHRE, width=2.2),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=1
    )

    # 1.6 2D Autonomous vector field V = (x_2, -g) on wall t = 0 in Forest Ink
    fig.add_trace(
        go.Cone(
            x=T_g2d, y=X1_g2d, z=X2_g2d,
            u=np.zeros_like(Vx1_n), v=Vx1_n, w=Vx2_n,
            sizemode="absolute",
            sizeref=0.82,
            anchor="tail",
            colorscale=[[0, INK_FOREST], [1, INK_FOREST]],
            showscale=False,
            name="Autonomous Field V(x₁, x₂) at t = 0",
            legendgroup="p1_field2d",
            showlegend=True,
            hoverinfo="text",
            hovertext=[f"V = (x₂={x2:.1f}, −g={-g:.1f})" for x2 in X2_g2d]
        ),
        row=1, col=1
    )

    shaft2d_x, shaft2d_y, shaft2d_z = [], [], []
    for tx, x1x, x2x, vx, vz in zip(T_g2d, X1_g2d, X2_g2d, Vx1_n, Vx2_n):
        shaft2d_x.extend([tx, tx, None])
        shaft2d_y.extend([x1x, x1x + vx, None])
        shaft2d_z.extend([x2x, x2x + vz, None])
    fig.add_trace(
        go.Scatter3d(
            x=shaft2d_x, y=shaft2d_y, z=shaft2d_z,
            mode='lines',
            line=dict(color=INK_FOREST, width=1.8),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=1
    )

    # -------------------------------------------------------------------------
    # PANEL 2: FIRST INTEGRAL RELIEF (x_1, x_2, E)
    # -------------------------------------------------------------------------

    # 2.1 Mechanical Energy Surface E(x_1, x_2) = (1/2) m x_2^2 + m g x_1
    fig.add_trace(
        go.Surface(
            x=X1_E, y=X2_E, z=Energy_Surface,
            colorscale=[
                [0.0, 'rgba(120, 112, 100, 0.16)'],
                [0.5, 'rgba(140, 90, 22, 0.22)'],
                [1.0, 'rgba(27, 73, 101, 0.28)']
            ],
            showscale=False,
            name="Energy Surface E(x₁, x₂)",
            legendgroup="p2_energysurf",
            showlegend=True,
            hoverinfo="text",
            hovertext="First Integral Relief: E = ½ m x₂² + m g x₁"
        ),
        row=1, col=2
    )

    # 2.2 Constant Energy Horizontal Cutting Plane E = E0
    fig.add_trace(
        go.Surface(
            x=X1_E, y=X2_E, z=Plane_E0,
            colorscale=[
                [0, 'rgba(140, 90, 22, 0.16)'],
                [1, 'rgba(140, 90, 22, 0.16)']
            ],
            showscale=False,
            name="Cutting Plane E = E₀ = 98 J",
            legendgroup="p2_plane",
            showlegend=True,
            hoverinfo="text",
            hovertext="Total Mechanical Energy Plane E₀ = 98.0 J"
        ),
        row=1, col=2
    )

    # 2.3 3D Intersection Level Curve at E = E0 in Vermilion Ink
    fig.add_trace(
        go.Scatter3d(
            x=x1_iso, y=x2_iso, z=E_iso,
            mode='lines',
            line=dict(color=INK_VERMILION, width=7),
            name="3D Level Curve: E(x₁, x₂) = E₀",
            legendgroup="p2_level3d",
            showlegend=True,
            hoverinfo="text",
            hovertext="3D Intersection: Orbit on Level E = 98 J"
        ),
        row=1, col=2
    )

    # 2.4 Projection of Level Curve onto Base Plane E = 0 in Prussian Cobalt Ink
    fig.add_trace(
        go.Scatter3d(
            x=x1_iso, y=x2_iso, z=E_proj_base,
            mode='lines',
            line=dict(color=INK_COBALT, width=6),
            name="Base Projection E = 0 (Phase Orbit)",
            legendgroup="p2_levelbase",
            showlegend=True,
            hoverinfo="text",
            hovertext="Base Phase Plane Orbit: x₁ = E₀/mg − x₂²/(2g)"
        ),
        row=1, col=2
    )

    # 2.5 Vertical drop lines connecting 3D level curve to base projection
    vert_x, vert_y, vert_z = [], [], []
    for idx in sample_indices:
        x1_val = x1_iso[idx]
        x2_val = x2_iso[idx]
        vert_x.extend([x1_val, x1_val, None])
        vert_y.extend([x2_val, x2_val, None])
        vert_z.extend([E0, E_base, None])

    fig.add_trace(
        go.Scatter3d(
            x=vert_x, y=vert_y, z=vert_z,
            mode='lines',
            line=dict(color='rgba(120, 112, 100, 0.72)', width=2, dash='dot'),
            name="Vertical Tie-Lines (E₀ → Base)",
            legendgroup="p2_vert_tie",
            showlegend=True,
            hoverinfo="none"
        ),
        row=1, col=2
    )

    # 2.6 Autonomous Vector Field on Base Plane E = 0 in Forest Ink
    fig.add_trace(
        go.Cone(
            x=X1_EB, y=X2_EB, z=E_EB,
            u=EB_Vx1_n, v=EB_Vx2_n, w=np.zeros_like(EB_Vx1_n),
            sizemode="absolute",
            sizeref=0.82,
            anchor="tail",
            colorscale=[[0, INK_FOREST], [1, INK_FOREST]],
            showscale=False,
            name="Base Phase Vector Field",
            legendgroup="p2_basefield",
            showlegend=True,
            hoverinfo="text",
            hovertext=[f"V(base) = (x₂={x2:.1f}, −g={-g:.1f})" for x2 in X2_EB]
        ),
        row=1, col=2
    )

    shaft_eb_x, shaft_eb_y, shaft_eb_z = [], [], []
    for x1_val, x2_val, e_val, vx, vy in zip(X1_EB, X2_EB, E_EB, EB_Vx1_n, EB_Vx2_n):
        shaft_eb_x.extend([x1_val, x1_val + vx, None])
        shaft_eb_y.extend([x2_val, x2_val + vy, None])
        shaft_eb_z.extend([e_val, e_val, None])
    fig.add_trace(
        go.Scatter3d(
            x=shaft_eb_x, y=shaft_eb_y, z=shaft_eb_z,
            mode='lines',
            line=dict(color=INK_FOREST, width=1.8),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=2
    )

    # -------------------------------------------------------------------------
    # TUFTE MINIMALIST CAMERA & ARCHIVAL PAPER LAYOUT
    # -------------------------------------------------------------------------
    camera_scene1 = dict(
        eye=dict(x=-1.75, y=-1.85, z=1.25),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0.0, z=-0.1)
    )

    camera_scene2 = dict(
        eye=dict(x=1.75, y=-1.75, z=1.35),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.0, y=0.0, z=-0.1)
    )

    axis_common = dict(
        showbackground=False,
        gridcolor="rgba(120, 112, 100, 0.24)",
        zerolinecolor="rgba(25, 23, 21, 0.45)",
        tickfont=dict(family="IBM Plex Mono, monospace", size=10, color=INK_SECONDARY),
        title_font=dict(family="EB Garamond, Georgia, serif", size=13, color=INK_PRIMARY)
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Dynamical Systems and Ordinary Differential Equations (V. I. Arnold — Section 1.1.14)</b><br>"
                "<sup><i>Free Fall Problem: dx₁/dt = x₂ , dx₂/dt = −g  •  First Integral Conservation E(x₁, x₂) = ½mx₂² + mgx₁ = 98.0 J</i></sup>"
            ),
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=INK_PRIMARY, family="EB Garamond, Palatino Linotype, Georgia, serif")
        ),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title='t [Time (s)]', **axis_common),
            yaxis=dict(title='x₁ [Position / Height (m)]', **axis_common),
            zaxis=dict(title='x₂ [Velocity (m/s)]', **axis_common),
            camera=camera_scene1,
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1.0, z=1.0)
        ),
        scene2=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title='x₁ [Position / Height (m)]', **axis_common),
            yaxis=dict(title='x₂ [Velocity (m/s)]', **axis_common),
            zaxis=dict(title='E [Mechanical Energy (J)]', **axis_common),
            camera=camera_scene2,
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=1.0, z=1.0)
        ),
        legend=dict(
            x=1.01, y=0.5,
            xanchor='left', yanchor='middle',
            bgcolor='rgba(245, 239, 224, 0.85)',
            borderwidth=0,
            font=dict(size=11.5, color=INK_PRIMARY, family="EB Garamond, Georgia, serif")
        ),
        margin=dict(l=20, r=20, t=105, b=30),
        height=750,
        width=1500,
        paper_bgcolor=ARCHIVAL_PAPER_BG,
        plot_bgcolor=ARCHIVAL_PAPER_BG
    )

    return fig


def save_archival_plotly_html(fig, html_filepath: Path) -> None:
    """Exports the Plotly figure to a self-contained HTML file with 1940s–50s archival paper styling."""
    raw_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    archival_css = f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      body {{
        background-color: {ARCHIVAL_PAPER_BG};
        background-image: radial-gradient(circle at 50% 35%, #faf6eb 0%, #f5efe0 70%, #ede4cf 100%);
        color: {INK_PRIMARY};
        font-family: 'EB Garamond', Georgia, serif;
        margin: 0;
        padding: 16px 24px;
        display: flex;
        justify-content: center;
      }}
    </style>
    </head>
    """
    styled_html = raw_html.replace("</head>", archival_css, 1)
    Path(html_filepath).write_text(styled_html, encoding="utf-8")


# =============================================================================
# 4. CONSTRUCTION OF MATPLOTLIB VISUALIZATION (ARCHIVAL TUFTE EDITION)
# =============================================================================
def build_matplotlib_figure():
    """Generates publication-quality 3D dual-panel visualization in 1940s–50s archival Tufte style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.edgecolor": INK_SECONDARY,
            "axes.linewidth": 0.85,
            "text.color": INK_PRIMARY,
        }
    )

    fig = plt.figure(figsize=(19, 9.0), dpi=300, facecolor=ARCHIVAL_PAPER_BG)
    fig.suptitle(
        r"$\bf{Dynamical\ Systems\ and\ Ordinary\ Differential\ Equations\ (V.\ I.\ Arnold\ —\ Section\ 1.1.14)}$"
        + "\n"
        + r"Free Fall Problem: $\dot{x}_1 = x_2,\; \dot{x}_2 = -g$ ($g = 9.8\ \mathrm{m/s^2},\; m = 1.0\ \mathrm{kg}$)   $\bullet$   "
        + r"First Integral: $E(x_1, x_2) = \frac{1}{2}mx_2^2 + mgx_1 = 98.0\ \mathrm{J}$",
        fontsize=14,
        color=INK_PRIMARY,
        y=0.97,
    )

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.14)

    # -------------------------------------------------------------------------
    # PANEL 1: EXTENDED PHASE SPACE (Fig. 1a)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_facecolor(ARCHIVAL_PAPER_BG)

    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(RULE_HAIRLINE)
        axis._axinfo["grid"]["color"] = (0.78, 0.75, 0.66, 0.45)
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["linewidth"] = 0.65

    ax1.set_title(
        r"$\bf{Fig.\ 1a.\ \ Extended\ Phase\ Space\ \mathbb{R} \times \mathbb{R}^2\ (t,\ x_1,\ x_2)}$" + "\n"
        r"Parabolic Cylinder $x_1 = \frac{E_0}{mg} - \frac{1}{2g}x_2^2$, Integral Curve $\gamma(t)$ & Projection ($t=0$)",
        fontsize=11.5,
        color=INK_PRIMARY,
        y=1.02,
        pad=0,
    )

    # 2D Invariant Surface (Parabolic Cylinder in Muted Sepia Wash)
    ax1.plot_surface(
        T_mesh, X1_mesh, X2_mesh,
        color=INK_OCHRE, alpha=0.11, edgecolor='none', antialiased=True
    )

    # 3D Integral Curve gamma(t) in Carmine Vermilion Ink
    ax1.plot(
        t_curve, x1_curve, x2_curve,
        color=INK_VERMILION, linewidth=2.8, label=r'Integral Curve $\gamma(t) = (t, x_1(t), x_2(t))$'
    )

    # Projection onto lateral wall t = 0 (Phase Parabola in Prussian Cobalt Ink)
    ax1.plot(
        np.zeros_like(x1_curve), x1_curve, x2_curve,
        color=INK_COBALT, linewidth=2.0, linestyle='--', label=r'Phase Projection ($t=0$)'
    )

    # Orthogonal projection drop lines connecting gamma(t) to shadow at t = 0
    for tk, x1k, x2k in zip(t_proj_pts, x1_proj_pts, x2_proj_pts):
        ax1.plot([tk, 0.0], [x1k, x1k], [x2k, x2k], color=INK_MUTED, linestyle=':', linewidth=1.0, alpha=0.75)

    # Key characteristic nodes (Classic Hollow Technical Fixed-Point Rings)
    node_t = [0.0, v0 / g, t_max]
    node_x1 = [x0, E0 / (m * g), x1_curve[-1]]
    node_x2 = [v0, 0.0, x2_curve[-1]]
    ax1.scatter(
        node_t, node_x1, node_x2,
        color=ARCHIVAL_PAPER_BG, edgecolor=INK_PRIMARY, s=48, linewidth=1.6, depthshade=False, zorder=6,
        label=r'Characteristic Nodes ($t=0$, Apex, $t=3$)'
    )

    # Extended vector field V_tilde on cylinder surface in Ochre Ink
    ax1.quiver(
        T_vec, X1_vec, X2_vec,
        U_t_n, U_x1_n, U_x2_n,
        color=INK_OCHRE, length=1.0, normalize=False, arrow_length_ratio=0.32, linewidth=1.0, alpha=0.8,
        label=r'Extended Field $\tilde{V} = (1, x_2, -g)$'
    )

    # 2D Autonomous vector field on wall t = 0 in Forest Ink
    ax1.quiver(
        T_g2d, X1_g2d, X2_g2d,
        np.zeros_like(Vx1_n), Vx1_n, Vx2_n,
        color=INK_FOREST, length=1.0, normalize=False, arrow_length_ratio=0.32, linewidth=0.95, alpha=0.8,
        label=r'Phase Field $V = (x_2, -g)$ at $t=0$'
    )

    ax1.set_xlabel(r'$t$ [Time (s)]', labelpad=10, fontsize=10.5, color=INK_PRIMARY)
    ax1.set_ylabel(r'$x_1$ [Position (m)]', labelpad=10, fontsize=10.5, color=INK_PRIMARY)
    ax1.set_zlabel(r'$x_2$ [Velocity (m/s)]', labelpad=10, fontsize=10.5, color=INK_PRIMARY)
    ax1.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax1.view_init(elev=24, azim=-125)
    ax1.legend(
        loc='upper left',
        bbox_to_anchor=(-0.02, 0.94),
        fontsize=8.5,
        frameon=True,
        facecolor=ARCHIVAL_PAPER_BG,
        framealpha=0.88,
        edgecolor='none',
        labelcolor=INK_PRIMARY,
    )

    # -------------------------------------------------------------------------
    # PANEL 2: FIRST INTEGRAL RELIEF (Fig. 1b)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_facecolor(ARCHIVAL_PAPER_BG)

    for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(RULE_HAIRLINE)
        axis._axinfo["grid"]["color"] = (0.78, 0.75, 0.66, 0.45)
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["linewidth"] = 0.65

    ax2.set_title(
        r"$\bf{Fig.\ 1b.\ \ First\ Integral\ Relief\ (x_1,\ x_2,\ E)}$" + "\n"
        r"Energy Surface $E(x_1, x_2) = \frac{1}{2}mx_2^2 + mgx_1$, Cutting Plane $E=E_0$ & Base Orbit",
        fontsize=11.5,
        color=INK_PRIMARY,
        y=1.02,
        pad=0,
    )

    # Mechanical Energy Surface E(x_1, x_2) in Subtle Archival Sepia-Cobalt Wash
    ax2.plot_surface(
        X1_E, X2_E, Energy_Surface,
        cmap=ARCHIVAL_CMAP, alpha=0.22, edgecolor='none', antialiased=True
    )

    # Constant Energy Horizontal Plane E = E0 in Subtle Warm Ochre
    ax2.plot_surface(
        X1_E, X2_E, Plane_E0,
        color=INK_OCHRE, alpha=0.12, edgecolor='none'
    )

    # 3D Intersection Level Curve at E = E0 in Carmine Vermilion Ink
    ax2.plot(
        x1_iso, x2_iso, E_iso,
        color=INK_VERMILION, linewidth=2.8, label=r'Level Curve $E(x_1, x_2) = E_0$'
    )

    # Projection onto Base Plane E = E_base (Phase Orbit in Prussian Cobalt Ink)
    ax2.plot(
        x1_iso, x2_iso, E_proj_base,
        color=INK_COBALT, linewidth=2.0, linestyle='--', label=r'Base Projection ($E=0$): Phase Orbit'
    )

    # Vertical drop lines connecting 3D level curve to base projection
    for idx in sample_indices:
        x1_val = x1_iso[idx]
        x2_val = x2_iso[idx]
        ax2.plot([x1_val, x1_val], [x2_val, x2_val], [E0, E_base], color=INK_MUTED, linestyle=':', linewidth=1.0, alpha=0.75)

    # Autonomous Phase Vector Field on Base Plane in Forest Ink
    ax2.quiver(
        X1_EB, X2_EB, E_EB,
        EB_Vx1_n, EB_Vx2_n, np.zeros_like(EB_Vx1_n),
        color=INK_FOREST, length=1.0, normalize=False, arrow_length_ratio=0.32, linewidth=0.95, alpha=0.8,
        label=r'Field $V = (x_2, -g)$ at Base'
    )

    ax2.set_xlabel(r'$x_1$ [Position (m)]', labelpad=10, fontsize=10.5, color=INK_PRIMARY)
    ax2.set_ylabel(r'$x_2$ [Velocity (m/s)]', labelpad=10, fontsize=10.5, color=INK_PRIMARY)
    ax2.set_zlabel(r'$E$ [Energy (J)]', labelpad=10, fontsize=10.5, color=INK_PRIMARY)
    ax2.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax2.view_init(elev=28, azim=-55)
    ax2.legend(
        loc='upper right',
        bbox_to_anchor=(1.02, 0.94),
        fontsize=8.5,
        frameon=True,
        facecolor=ARCHIVAL_PAPER_BG,
        framealpha=0.88,
        edgecolor='none',
        labelcolor=INK_PRIMARY,
    )

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.82, wspace=0.14)
    return fig


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    pictures_dir = script_dir.parent / "pictures"
    pictures_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Generating interactive 3D visualization via Plotly...")
    plotly_fig = build_plotly_figure()
    html_filepath = script_dir / "free_fall_arnold_3d.html"
    save_archival_plotly_html(plotly_fig, html_filepath)
    print(f" -> Success! Interactive HTML file saved as: '{html_filepath}'")

    print("\n[2/2] Generating high-resolution 3D visualization via Matplotlib...")
    mpl_fig = build_matplotlib_figure()
    png_filepath = pictures_dir / "free_fall_arnold_3d.png"
    mpl_fig.savefig(str(png_filepath), dpi=300, bbox_inches="tight", facecolor=mpl_fig.get_facecolor())
    print(f" -> Success! High-resolution image saved as: '{png_filepath}'")

    print("\nProcess successfully completed!")
