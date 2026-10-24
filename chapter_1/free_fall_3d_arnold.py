"""
Scientific 3D Visualization: Free Fall in V. I. Arnold's Formalism
==================================================================
Based on Vladimir I. Arnold's "Ordinary Differential Equations" (Chapter 1, Section 1.1.14).

This script constructs a side-by-side dual-panel 3D visualization:
- Panel 1: The Extended Phase Space R x R^2 with axes (t, x_1, x_2).
  Features the invariant energy manifold (parabolic cylinder),
  the three-dimensional integral curve gamma(t), the extended vector field V_tilde(t, x_1, x_2),
  the phase plane projection on the lateral wall t = 0 with the autonomous vector field V(x_1, x_2),
  and orthogonal projection drop lines connecting the 3D trajectory to its shadow.

- Panel 2: The First Integral Relief (Energy Surface) with axes (x_1, x_2, E).
  Features the energy surface E(x_1, x_2) = (1/2) m x_2^2 + m g x_1, the horizontal cutting plane
  at E = E_0, the 3D intersection level curve, and the projection onto the base plane
  demonstrating geometrically that phase curves are level curves of mechanical energy.

Outputs:
1. Interactive HTML 3D visualization via Plotly: 'free_fall_arnold_3d.html'
2. High-resolution publication-quality image via Matplotlib: 'free_fall_arnold_3d.png'
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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

# Orthogonal projection points connecting 3D curve to t = 0 wall (dashed tie lines)
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
# 3. CONSTRUCTION OF INTERACTIVE PLOTLY VISUALIZATION
# =============================================================================
def build_plotly_figure():
    """Creates high-fidelity 3D interactive dual-panel visualization using Plotly."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            "<b>Panel 1: Extended Phase Space ℝ × ℝ² (t, x₁, x₂)</b><br>"
            "<sup>Invariant Manifold, Integral Curve and Parabolic Projection onto t = 0</sup>",
            "<b>Panel 2: First Integral Relief (x₁, x₂, E)</b><br>"
            "<sup>Mechanical Energy Surface, Section at E = E₀ and Base Level Curve</sup>"
        ),
        horizontal_spacing=0.03
    )

    # -------------------------------------------------------------------------
    # PANEL 1: EXTENDED PHASE SPACE (t, x_1, x_2)
    # Axes: X = t (Time), Y = x_1 (Position/Height), Z = x_2 (Velocity)
    # -------------------------------------------------------------------------

    # 1.1 Constant Energy Invariant Surface (Parabolic Cylinder)
    fig.add_trace(
        go.Surface(
            x=T_mesh, y=X1_mesh, z=X2_mesh,
            colorscale=[[0.0, 'rgba(30,144,255,0.45)'], [1.0, 'rgba(0,206,209,0.55)']],
            showscale=False,
            name="Invariant Surface x₁ = E₀/mg - x₂²/(2g)",
            legendgroup="p1_surf",
            showlegend=True,
            hoverinfo="text",
            hovertext="Invariant Manifold E₀ = 98 J (Parabolic Cylinder)"
        ),
        row=1, col=1
    )

    # 1.2 Projection on lateral wall t = 0 (Phase Parabola)
    fig.add_trace(
        go.Scatter3d(
            x=np.zeros_like(x1_curve), y=x1_curve, z=x2_curve,
            mode='lines',
            line=dict(color='#8B008B', width=8),
            name="Phase Plane Projection (t = 0)",
            legendgroup="p1_proj",
            showlegend=True,
            hoverinfo="text",
            hovertext="Projected Phase Orbit at t = 0: x₁ = 10 - x₂²/19.6"
        ),
        row=1, col=1
    )

    # 1.3 3D Integral Curve gamma(t)
    fig.add_trace(
        go.Scatter3d(
            x=t_curve, y=x1_curve, z=x2_curve,
            mode='lines',
            line=dict(color='#FF2200', width=10),
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

    # Key trajectory markers
    key_t = np.array([0.0, v0 / g, 2 * v0 / g, 3.0])
    key_x1 = -0.5 * g * (key_t ** 2) + v0 * key_t + x0
    key_x2 = -g * key_t + v0
    fig.add_trace(
        go.Scatter3d(
            x=key_t, y=key_x1, z=key_x2,
            mode='markers+text',
            marker=dict(size=6, color=['#00FF00', '#FFFF00', '#FFA500', '#FF0000']),
            text=[" t=0", " Apex (x₂=0)", " Return", " t=3s"],
            textposition="top right",
            name="Key Points on γ(t)",
            legendgroup="p1_pts",
            showlegend=True,
            hoverinfo="text",
            hovertext=[
                "Start: t=0 s, x₁=0 m, x₂=+14 m/s",
                "Apex: t=1.43 s, x₁=10 m, x₂=0 m/s",
                "Return to base: t=2.86 s, x₁=0 m, x₂=-14 m/s",
                "End: t=3.0 s, x₁=-2.1 m, x₂=-15.4 m/s"
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
            line=dict(color='rgba(120, 120, 120, 0.85)', width=3, dash='dash'),
            name="Orthogonal Projection Lines (t → 0)",
            legendgroup="p1_tie",
            showlegend=True,
            hoverinfo="none"
        ),
        row=1, col=1
    )

    # 1.5 Extended 3D vector field V_tilde = (1, x_2, -g) on surface
    fig.add_trace(
        go.Cone(
            x=T_vec, y=X1_vec, z=X2_vec,
            u=U_t_n, v=U_x1_n, w=U_x2_n,
            sizemode="absolute",
            sizeref=1.3,
            anchor="tail",
            colorscale=[[0, '#FF8C00'], [1, '#FF4500']],
            showscale=False,
            name="Extended Field Ṽ = (1, x₂, -g)",
            legendgroup="p1_cone",
            showlegend=True,
            hoverinfo="text",
            hovertext="Extended Tangent Vector Ṽ(t, x₁, x₂) = (1, x₂, -g)"
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
            line=dict(color='#FF4500', width=3),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=1
    )

    # 1.6 2D Autonomous vector field V = (x_2, -g) on wall t = 0
    fig.add_trace(
        go.Cone(
            x=T_g2d, y=X1_g2d, z=X2_g2d,
            u=np.zeros_like(Vx1_n), v=Vx1_n, w=Vx2_n,
            sizemode="absolute",
            sizeref=0.9,
            anchor="tail",
            colorscale=[[0, '#2E8B57'], [1, '#006400']],
            showscale=False,
            name="Autonomous Field V(x₁, x₂) at t = 0",
            legendgroup="p1_field2d",
            showlegend=True,
            hoverinfo="text",
            hovertext=[f"V = (x₂={x2:.1f}, -g={-g:.1f})" for x2 in X2_g2d]
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
            line=dict(color='#006400', width=2),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=1
    )

    # -------------------------------------------------------------------------
    # PANEL 2: FIRST INTEGRAL RELIEF (x_1, x_2, E)
    # Axes: X = x_1 (Position/Height), Y = x_2 (Velocity), Z = E (Mechanical Energy)
    # -------------------------------------------------------------------------

    # 2.1 Mechanical Energy Surface E(x_1, x_2) = (1/2) m x_2^2 + m g x_1
    fig.add_trace(
        go.Surface(
            x=X1_E, y=X2_E, z=Energy_Surface,
            colorscale='Viridis',
            opacity=0.72,
            showscale=False,
            name="Energy Surface E(x₁, x₂)",
            legendgroup="p2_energysurf",
            showlegend=True,
            hoverinfo="text",
            hovertext="First Integral Relief: E = 0.5·m·x₂² + m·g·x₁"
        ),
        row=1, col=2
    )

    # 2.2 Constant Energy Horizontal Cutting Plane E = E0
    fig.add_trace(
        go.Surface(
            x=X1_E, y=X2_E, z=Plane_E0,
            colorscale=[[0, 'rgba(255, 215, 0, 0.45)'], [1, 'rgba(255, 215, 0, 0.45)']],
            showscale=False,
            name="Plane E = E₀ = 98 J",
            legendgroup="p2_plane",
            showlegend=True,
            hoverinfo="text",
            hovertext="Total Mechanical Energy Plane E₀ = 98.0 J"
        ),
        row=1, col=2
    )

    # 2.3 3D Intersection Level Curve at E = E0
    fig.add_trace(
        go.Scatter3d(
            x=x1_iso, y=x2_iso, z=E_iso,
            mode='lines',
            line=dict(color='#FF1493', width=8),
            name="3D Level Curve: E(x₁, x₂) = E₀",
            legendgroup="p2_level3d",
            showlegend=True,
            hoverinfo="text",
            hovertext="3D Intersection: Orbit on Level E = 98 J"
        ),
        row=1, col=2
    )

    # 2.4 Projection of Level Curve onto Base Plane E = 0 (Phase Orbit)
    fig.add_trace(
        go.Scatter3d(
            x=x1_iso, y=x2_iso, z=E_proj_base,
            mode='lines',
            line=dict(color='#8B008B', width=8),
            name="Base Plane Projection E = 0 (Phase Orbit)",
            legendgroup="p2_levelbase",
            showlegend=True,
            hoverinfo="text",
            hovertext="Base Phase Plane Orbit: x₁ = E₀/mg - x₂²/(2g)"
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
            line=dict(color='rgba(100, 100, 100, 0.75)', width=3, dash='dash'),
            name="Vertical Projection (E₀ → Base)",
            legendgroup="p2_vert_tie",
            showlegend=True,
            hoverinfo="none"
        ),
        row=1, col=2
    )

    # 2.6 Autonomous Vector Field on Base Plane E = 0
    fig.add_trace(
        go.Cone(
            x=X1_EB, y=X2_EB, z=E_EB,
            u=EB_Vx1_n, v=EB_Vx2_n, w=np.zeros_like(EB_Vx1_n),
            sizemode="absolute",
            sizeref=0.9,
            anchor="tail",
            colorscale=[[0, '#2E8B57'], [1, '#006400']],
            showscale=False,
            name="Base Phase Vector Field",
            legendgroup="p2_basefield",
            showlegend=True,
            hoverinfo="text",
            hovertext=[f"V(base) = (x₂={x2:.1f}, -g={-g:.1f})" for x2 in X2_EB]
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
            line=dict(color='#006400', width=2),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=2
    )

    # -------------------------------------------------------------------------
    # CAMERA AND LAYOUT CONFIGURATIONS
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

    fig.update_layout(
        title=dict(
            text=(
                "<b>Classical Mechanics and Dynamical Systems (V. I. Arnold): Free Fall Problem</b><br>"
                "<sup>Mechanical Energy Conservation (E₀ = 98 J), Extended Phase Space and Invariant Manifolds</sup>"
            ),
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#1A202C', family='Arial, sans-serif')
        ),
        scene=dict(
            xaxis=dict(title='<b>t [Time (s)]</b>', backgroundcolor='#F7FAFC', gridcolor='#CBD5E0', showbackground=True),
            yaxis=dict(title='<b>x₁ [Position / Height (m)]</b>', backgroundcolor='#F7FAFC', gridcolor='#CBD5E0', showbackground=True),
            zaxis=dict(title='<b>x₂ [Velocity (m/s)]</b>', backgroundcolor='#EDF2F7', gridcolor='#CBD5E0', showbackground=True),
            camera=camera_scene1,
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1.0, z=1.0)
        ),
        scene2=dict(
            xaxis=dict(title='<b>x₁ [Position / Height (m)]</b>', backgroundcolor='#F7FAFC', gridcolor='#CBD5E0', showbackground=True),
            yaxis=dict(title='<b>x₂ [Velocity (m/s)]</b>', backgroundcolor='#F7FAFC', gridcolor='#CBD5E0', showbackground=True),
            zaxis=dict(title='<b>E [Mechanical Energy (J)]</b>', backgroundcolor='#EDF2F7', gridcolor='#CBD5E0', showbackground=True),
            camera=camera_scene2,
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=1.0, z=1.0)
        ),
        legend=dict(
            x=1.02, y=0.5,
            xanchor='left', yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#CBD5E0',
            borderwidth=1,
            font=dict(size=10, family='Arial, sans-serif')
        ),
        margin=dict(l=20, r=20, t=100, b=30),
        height=750,
        width=1500,
        paper_bgcolor='#FFFFFF'
    )

    return fig


# =============================================================================
# 4. CONSTRUCTION OF MATPLOTLIB VISUALIZATION (STATIC / PUBLICATION)
# =============================================================================
def build_matplotlib_figure():
    """Generates publication-quality 3D dual-panel visualization using Matplotlib."""
    fig = plt.figure(figsize=(20, 9.5))
    plt.suptitle(
        r"$\bf{Dynamical\ Systems\ and\ Ordinary\ Differential\ Equations\ (V.\ I.\ Arnold)}$" + "\n"
        r"Free Fall Problem: $\dot{x}_1 = x_2,\; \dot{x}_2 = -g$ ($g = 9.8\ \mathrm{m/s^2},\; m = 1.0\ \mathrm{kg},\; E_0 = 98.0\ \mathrm{J}$)",
        fontsize=15, y=0.98
    )

    # -------------------------------------------------------------------------
    # PANEL 1: EXTENDED PHASE SPACE
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title(
        r"$\bf{Panel\ 1:\ Extended\ Phase\ Space\ \mathbb{R} \times \mathbb{R}^2}$" + "\n"
        r"Parabolic Cylinder $x_1 = \frac{E_0}{mg} - \frac{1}{2g}x_2^2$, Trajectory $\gamma(t)$ and Projection onto $t=0$",
        fontsize=11, pad=10
    )

    # 2D Invariant Surface (Parabolic Cylinder)
    surf1 = ax1.plot_surface(
        T_mesh, X1_mesh, X2_mesh,
        cmap=cm.coolwarm, alpha=0.38, edgecolor='none', antialiased=True
    )

    # 3D Integral Curve gamma(t)
    ax1.plot(
        t_curve, x1_curve, x2_curve,
        color='red', linewidth=3.5, label=r'Integral Curve $\gamma(t) = (t, x_1(t), x_2(t))$'
    )

    # Projection onto lateral wall t = 0 (Phase Parabola)
    ax1.plot(
        np.zeros_like(x1_curve), x1_curve, x2_curve,
        color='purple', linewidth=3.0, linestyle='-', label=r'Phase Projection onto $t=0$'
    )

    # Orthogonal projection drop lines connecting gamma(t) to shadow at t = 0
    for tk, x1k, x2k in zip(t_proj_pts, x1_proj_pts, x2_proj_pts):
        ax1.plot([tk, 0.0], [x1k, x1k], [x2k, x2k], color='gray', linestyle='--', linewidth=1.2)

    # Key singular points
    ax1.scatter([0.0], [x0], [v0], color='green', s=60, zorder=5, label=r'Start $t=0$')
    ax1.scatter([v0 / g], [E0 / (m * g)], [0.0], color='gold', s=60, zorder=5, label=r'Apex $x_2=0$')
    ax1.scatter([t_max], [x1_curve[-1]], [x2_curve[-1]], color='darkred', s=60, zorder=5, label=r'End $t=3$')

    # Extended vector field V_tilde on cylinder surface
    ax1.quiver(
        T_vec, X1_vec, X2_vec,
        U_t_n, U_x1_n, U_x2_n,
        color='darkorange', length=1.0, normalize=False, arrow_length_ratio=0.35, linewidth=1.2,
        label=r'Extended Field $\tilde{V} = (1, x_2, -g)$'
    )

    # 2D Autonomous vector field on wall t = 0
    ax1.quiver(
        T_g2d, X1_g2d, X2_g2d,
        np.zeros_like(Vx1_n), Vx1_n, Vx2_n,
        color='darkgreen', length=1.0, normalize=False, arrow_length_ratio=0.35, linewidth=1.1,
        label=r'Phase Field $V = (x_2, -g)$ at $t=0$'
    )

    ax1.set_xlabel(r'$t$ [Time (s)]', labelpad=10, fontsize=10)
    ax1.set_ylabel(r'$x_1$ [Position (m)]', labelpad=10, fontsize=10)
    ax1.set_zlabel(r'$x_2$ [Velocity (m/s)]', labelpad=10, fontsize=10)
    ax1.view_init(elev=24, azim=-125)
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.05, 0.95), fontsize=8, framealpha=0.85)

    # -------------------------------------------------------------------------
    # PANEL 2: FIRST INTEGRAL RELIEF (ENERGY SURFACE)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title(
        r"$\bf{Panel\ 2:\ First\ Integral\ Relief\ (x_1, x_2, E)}$" + "\n"
        r"Energy Surface $E(x_1, x_2) = \frac{1}{2}mx_2^2 + mgx_1$, Section $E=E_0$ and Base Level Curve",
        fontsize=11, pad=10
    )

    # Mechanical Energy Surface E(x_1, x_2)
    surf2 = ax2.plot_surface(
        X1_E, X2_E, Energy_Surface,
        cmap=cm.viridis, alpha=0.55, edgecolor='none', antialiased=True
    )

    # Constant Energy Horizontal Plane E = E0
    plane2 = ax2.plot_surface(
        X1_E, X2_E, Plane_E0,
        color='gold', alpha=0.35, edgecolor='none'
    )

    # 3D Intersection Level Curve at E = E0
    ax2.plot(
        x1_iso, x2_iso, E_iso,
        color='magenta', linewidth=3.5, label=r'Level Curve $E(x_1, x_2) = E_0$'
    )

    # Projection onto Base Plane E = E_base (Phase Orbit)
    ax2.plot(
        x1_iso, x2_iso, E_proj_base,
        color='purple', linewidth=3.0, linestyle='-', label=r'Base Projection ($E=0$): Phase Orbit'
    )

    # Vertical drop lines connecting 3D level curve to base projection
    for idx in sample_indices:
        x1_val = x1_iso[idx]
        x2_val = x2_iso[idx]
        ax2.plot([x1_val, x1_val], [x2_val, x2_val], [E0, E_base], color='gray', linestyle='--', linewidth=1.2)

    # Autonomous Phase Vector Field on Base Plane
    ax2.quiver(
        X1_EB, X2_EB, E_EB,
        EB_Vx1_n, EB_Vx2_n, np.zeros_like(EB_Vx1_n),
        color='darkgreen', length=1.0, normalize=False, arrow_length_ratio=0.35, linewidth=1.1,
        label=r'Field $V = (x_2, -g)$ at Base'
    )

    ax2.set_xlabel(r'$x_1$ [Position (m)]', labelpad=10, fontsize=10)
    ax2.set_ylabel(r'$x_2$ [Velocity (m/s)]', labelpad=10, fontsize=10)
    ax2.set_zlabel(r'$E$ [Energy (J)]', labelpad=10, fontsize=10)
    ax2.view_init(elev=28, azim=-55)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 0.95), fontsize=8, framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n[1/2] Generating interactive 3D visualization via Plotly...")
    plotly_fig = build_plotly_figure()
    html_filename = "free_fall_arnold_3d.html"
    plotly_fig.write_html(html_filename, include_plotlyjs="cdn")
    print(f" -> Success! Interactive HTML file saved as: '{html_filename}'")

    print("\n[2/2] Generating high-resolution 3D visualization via Matplotlib...")
    mpl_fig = build_matplotlib_figure()
    png_filename = "free_fall_arnold_3d.png"
    mpl_fig.savefig(png_filename, dpi=300, bbox_inches='tight')
    print(f" -> Success! High-resolution image saved as: '{png_filename}'")

    print("\nProcess successfully completed!")
