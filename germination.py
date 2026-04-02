################################################################################
#
# Bacteria Germination with Chemotaxis (2D Version - Booster Model)
#
# Authors: Mohammad Mousavi
# Email: sm2652@cornell.edu
# date: 03/20/2025
#
# Updated: 02/05/2026
# Changes:
#   1. Nondimensionalization using intrinsic length scale sqrt(D/r).
#   2. Glucose Initial Condition uniform everywhere.
#   3. "Booster Model" Implementation:
#      - Glucose fuels base growth (100% of rmax).
#      - Germinant fuels a "boost" (extra 10% of rmax).
#      - Consumption terms separated to satisfy mass balance.
#
################################################################################
from __future__ import division
from dolfin import *
from mshr import *
import ufl
import os
import numpy as np
import matplotlib.pyplot as plt

#################################Problem Setup################################################
# SIMULATION TIME
# Note: Time scale is still 1/rmax, so T=2.0 is still ~1.1 hours.
T = 2           
num_steps = 200  # Kept high for stability
dt = T / num_steps  

# OUTPUT STEPS
# Using linspace to get 6 evenly spaced unique integer steps
selected_steps = np.unique(np.linspace(1, num_steps, 6).astype(int)).tolist()

#################################Material Property & SCALING##################################
# 1. Dimensional Parameters (Physical Units)
Dn_phys   = 2.0e-6    # [cm^2/s] Bacterial diffusion
Dg_phys   = 6.7e-6    # [cm^2/s] Glucose diffusion
Dc_phys   = 9.1e-6    # [cm^2/s] Germinant diffusion
chi0_phys = 1.0e-5    # [cm^2/s] Chemotactic coefficient

rmax_phys_hr = 1.8    # [1/hr] Max growth rate (per hour)
qmax_phys_hr = 2.0    # [1/hr] Max germination rate (per hour)

Kg_phys   = 4.0e-6    # [g/mL] Glucose half-saturation
Kd_phys   = 4.0e-6    # [g/mL] Receptor sensitivity
Kc_phys   = 1.0e-4    # [g/mL] Germinant half-saturation (Used for uptake affinity)
cmin_phys = 1.0e-4    # [g/mL] Germinant threshold

g0_phys   = 2.0e-3    # [g/mL] Initial glucose scale
c0_phys   = 5.0e-3    # [g/mL] Initial germinant scale

# Physical Geometry
R_dish_phys = 4.8     # [cm] Physical Dish Radius
R0_spot_phys = 0.96   # [cm] Physical Spot Radius (was 0.2*4.8 in previous code)

# 2. NEW SCALING FACTORS (Advisor Recommendation)
rmax_phys = rmax_phys_hr / 3600.0  # [1/s]

# Length Scale: L = sqrt(D_n / r_max)
# This is the characteristic reaction-diffusion length
L_scale   = np.sqrt(Dn_phys / rmax_phys) # [cm]

T_scale   = 1.0 / rmax_phys        # [s] Time Scale
Diff_scale = (L_scale**2) * rmax_phys # [cm^2/s] = D_n_phys

# 3. Non-dimensional Geometry
# Since L_scale is small (~0.06 cm), the Dimensionless Radius will be large (~75)
R = R_dish_phys / L_scale
R0 = R0_spot_phys / L_scale

# 4. Non-dimensional Parameters (Calculated)
# Dn* will be exactly 1.0 by definition of the scaling
Dn   = Constant(Dn_phys / Diff_scale)
Dg   = Constant(Dg_phys / Diff_scale)
Dc   = Constant(Dc_phys / Diff_scale)
chi0 = Constant(chi0_phys / Diff_scale)

rmax = Constant(1.0)  
qmax = Constant(qmax_phys_hr / rmax_phys_hr) 

# Normalized Saturation Constants
Kg   = Constant(Kg_phys / g0_phys)
Kd   = Constant(Kd_phys / g0_phys)
Kc   = Constant(Kc_phys / c0_phys)
cmin = Constant(cmin_phys / c0_phys)

# Concentration Ratio (Lambda) needed for stoichiometric scaling in Eq 4
# Because u_n is scaled by g0 but u_c is scaled by c0.
Lambda_ratio = Constant(g0_phys / c0_phys)

Y    = Constant(1.0)  # Yield on Glucose
Yc   = Constant(1.0)  # Yield on Germinant

# Initial Conditions (ND)
glucose_IC   = 1.0
germinant_IC = 1.0

# Smoothing Parameters
k_IC = 10.0 
transition_k = 200.0 

#################################Logging & Saving#############################################
T_phys_hours = T / rmax_phys_hr

# Create Folder Name
simulation_name = (f"Run_BoosterModel_T{T}_Steps{num_steps}")
savedir = simulation_name + "/"
os.makedirs(savedir, exist_ok=True)

# Create Information String
info_str = f"""
================================================================
SIMULATION PARAMETERS & NEW SCALING
================================================================

1. SCALING CHANGE
------------------
Method: Intrinsic Reaction-Diffusion Length Scale
Length Scale (L = sqrt(Dn/rmax)) = {L_scale:.4f} [cm]
Time Scale (1/rmax)              = {T_scale:.2f} [s]
Diffusive Scale (D_n)            = {Diff_scale:.2e} [cm^2/s]

2. DIMENSIONLESS GEOMETRY
-------------------------
Physical Radius      = {R_dish_phys} [cm]
Dimensionless Radius = {R:.4f}

Physical Inner Radius (R0) = {R0_spot_phys} [cm]
Dimensionless R0           = {R0:.4f}

3. NON-DIMENSIONAL PARAMETERS
-----------------------------
Dn   = {float(Dn):.6f} (Should be 1.0)
Dg   = {float(Dg):.6f}
Dc   = {float(Dc):.6f}
chi0 = {float(chi0):.6f}
Lambda (g0/c0) = {float(Lambda_ratio):.6f}

4. KINETICS UPDATES (Booster Model)
-----------------------------------
Growth = Base (Glucose) * [1 + 0.1 * Boost (Alanine)]
Glucose Consumption: Fuels ONLY the Base term.
Germinant Consumption: Fuels ONLY the 0.1 Boost term.
   - Scaled by rmax (not qmax).
   - Scaled by Lambda (stoichiometry).

================================================================
"""
print(info_str)
with open(savedir + "simulation_parameters.txt", "w") as f:
    f.write(info_str)

#######################################Mesh###################################################
# --- MESH GENERATION ---
# Since R is now large (~75), we need to ensure the mesh resolution is adequate.
# Resolution 50 on Radius 75 gives element size ~ 1.5.
# This corresponds to physical size 1.5 * 0.06 ~ 0.09 cm, which is fine.
coarse_resolution = 40
domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, coarse_resolution)

# Refine the mesh LOCALLY near the center
# We need to scale the refinement zone by the new Dimensionless scale
# Physical Zone ~ 2.0 cm -> Dimensionless ~ 32.0
fine_zone_radius = 2.0 / L_scale 
refinement_levels = 2 

print(f"Initial coarse cells: {mesh.num_cells()}")

for i in range(refinement_levels):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    
    for cell in cells(mesh):
        p = cell.midpoint()
        if (p.x()**2 + p.y()**2) < fine_zone_radius**2:
            cell_markers[cell] = True
            
    mesh = refine(mesh, cell_markers)
    print(f"Refinement level {i+1}: {mesh.num_cells()} cells")

#################################Function Space###############################################
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
element = MixedElement([P1, P1, P1, P1])
V = FunctionSpace(mesh, element)

# trial functions
u = Function(V) 
u_n, u_s, u_g, u_c = split(u)

# test functions
v = TestFunction(V)
v_n, v_s, v_g, v_c = split(v)

u_prev = Function(V) 
u_n_prev, u_s_prev, u_g_prev, u_c_prev = split(u_prev)

du = TrialFunction(V)
####################################Equations#################################################
u_n_safe = ufl.max_value(u_n, 0.0)
u_s_safe = ufl.max_value(u_s, 0.0)
u_g_safe = ufl.max_value(u_g, 0.0)
u_c_safe = ufl.max_value(u_c, 0.0)

def germination_factor_smooth(c):
    switch = 0.5 * (1.0 + ufl.tanh(transition_k * (c - cmin)))
    rate = (c - cmin) / (c - 2*cmin + Kc)
    return switch * rate

def germination_factor_exact(c):
    # 1. Define the raw rate equation from the PDF (Eq 1)
    numerator = c - cmin
    denominator = c - 2*cmin + Kc
    
    raw_rate = numerator / denominator

    # 2. Implement the EXACT "max(0, ...)" logic using UFL conditional
    # Logic: If c < cmin, force output to 0.0. Otherwise, use the calculated rate.
    # This removes the "tanh" smoothing and restores the sharp physical cutoff.
    return ufl.conditional(ufl.lt(c, cmin), 0.0, raw_rate)

# --- BOOSTER MODEL IMPLEMENTATION ---

# 1. Base Monod Term (Depends on Glucose)
# This drives 100% of the normal growth
monod_glucose = u_g_safe / (u_g_safe + Kg)
base_growth = rmax * u_n_safe * monod_glucose

# 2. Booster Term (Depends on Germinant)
# Adds a 10% bonus if germinant is saturated
# Multiplicative factor: [1 + 0.1 * (c / (c + Kc))]
monod_germinant = u_c_safe / (u_c_safe + Kc)
boost_factor = 1.0 + (0.1 * monod_germinant)

# 3. Total Growth Term
growth_term = base_growth * boost_factor

# Germination term (Spores -> Cells)
germination_term = qmax * u_s_safe * germination_factor_smooth(u_c_safe)

# Chemotaxis Term
chi = chi0 * Kd / (Kd + u_g_safe)**2
chemotaxis_term = chi * u_n_safe * dot(grad(u_g), grad(v_n))

# --- CONSUMPTION TERMS ---

# Glucose consumption: Fuels the BASE growth only (The "1" in the boost factor)
glucose_consumption = (1.0 / Y) * base_growth

# Germinant consumption: Fuels the BOOST growth only (The "0.1" part)
# Rate = (0.1 / Yc) * Lambda * Base_Growth * Monod_Germinant
# This corresponds to: 0.1 * Lambda * rmax * n * (g/g+Kg) * (c/c+Kc)
germinant_consumption = (0.1 / Yc) * Lambda_ratio * base_growth * monod_germinant

# Equations
n_equation = ((u_n - u_n_prev) / dt) * v_n * dx \
            + Dn * dot(grad(u_n), grad(v_n)) * dx \
            - chemotaxis_term * dx \
            - growth_term * v_n * dx \
            - germination_term * v_n * dx

s_equation = ((u_s - u_s_prev) / dt) * v_s * dx \
            + germination_term * v_s * dx

g_equation = ((u_g - u_g_prev) / dt) * v_g * dx + Dg * dot(grad(u_g), grad(v_g)) * dx \
            + glucose_consumption * v_g * dx

c_equation = ((u_c - u_c_prev) / dt) * v_c * dx + Dc * dot(grad(u_c), grad(v_c)) * dx \
            + germinant_consumption * v_c * dx

F = n_equation + s_equation + g_equation + c_equation
J = derivative(F, u, du)
####################################B.C.###################################################
class BC(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]**2 + x[1]**2 <= R0**2  
BC = BC()
bc_c = DirichletBC(V.sub(3), Constant(1.0), BC)
bcs = [] 
#################################Visualization##############################################
xdmf_file = XDMFFile(f"{savedir}/bacterial_growth_simulation.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False
####################################I.C.###################################################
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.0 # n
        values[1] = 0.5 # s
        r = sqrt(x[0]**2 + x[1]**2) 
        
        # --- MODIFIED GLUCOSE I.C. ---
        # Glucose is now 1.0 EVERYWHERE (no step function)
        values[2] = glucose_IC 
        
        # Germinant remains localized
        values[3] = germinant_IC / (1.0 + np.exp(k_IC * (r - R0)))

    def value_shape(self):
        return (4,)

initial_conditions = InitialConditions(degree=1)
u.interpolate(initial_conditions)
u_prev.assign(u)
####################################Solver##########################################
solver_u_parameters   = {"nonlinear_solver": "snes",
                         "symmetric": True,
                         "snes_solver": {"linear_solver": "mumps",
                                         "method" : "newtonls",  
                                         "line_search": "bt",
                                         "maximum_iterations": 100,
                                         "absolute_tolerance": 1e-6,
                                         "relative_tolerance": 1e-6,
                                         "report": True,
                                         "error_on_nonconvergence": False}}
problem_u = NonlinearVariationalProblem(F, u, bcs, J=J)
solver_u  = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)

# Extract individual functions for visualization
V_n = V.sub(0).collapse()
V_s = V.sub(1).collapse()
V_g = V.sub(2).collapse()
V_c = V.sub(3).collapse()

n_viz = Function(V_n, name="Bacterial_Cells")
s_viz = Function(V_s, name="Spores")
g_viz = Function(V_g, name="Glucose")
c_viz = Function(V_c, name="Germinant")

# Save initial state
n_viz.assign(project(u_n, V_n))
s_viz.assign(project(u_s, V_s))
g_viz.assign(project(u_g, V_g))
c_viz.assign(project(u_c, V_c))

xdmf_file.write(n_viz, 0.0)
xdmf_file.write(s_viz, 0.0)
xdmf_file.write(g_viz, 0.0)
xdmf_file.write(c_viz, 0.0)

# Prepare to store profiles
saved_data = {
    "Bacterial_Cells": [],
    "Spores": [],
    "Glucose": [],
    "Germinant": []
}
saved_steps = []

def extract_radial_data(f):
    coords = mesh.coordinates()
    r_vals = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    values = f.compute_vertex_values(mesh)
    return r_vals, values

# Time-stepping loop
t = 0.0
GREEN_BOLD = "\033[1;32m"
RESET_COLOR = "\033[0m"

for i in range(num_steps):
    t += dt
    # Print Step number in Bold Green
    print(f"{GREEN_BOLD}Step {i+1}/{num_steps} (t = {t:.4f}){RESET_COLOR}")

    solver_u.solve()
    u_prev.assign(u)

    n_viz.assign(project(u_n, V_n))
    s_viz.assign(project(u_s, V_s))
    g_viz.assign(project(u_g, V_g))
    c_viz.assign(project(u_c, V_c))

    xdmf_file.write(n_viz, t)
    xdmf_file.write(s_viz, t)
    xdmf_file.write(g_viz, t)
    xdmf_file.write(c_viz, t)

    # Save profiles for selected steps
    current_step = i + 1
    if current_step in selected_steps:
        saved_steps.append(current_step)
        
        r_n, v_n = extract_radial_data(n_viz)
        r_s, v_s = extract_radial_data(s_viz)
        r_g, v_g = extract_radial_data(g_viz)
        r_c, v_c = extract_radial_data(c_viz)
        
        saved_data["Bacterial_Cells"].append((r_n, v_n))
        saved_data["Spores"].append((r_s, v_s))
        saved_data["Glucose"].append((r_g, v_g))
        saved_data["Germinant"].append((r_c, v_c))

# Variable names and labels
variable_names = ["Bacterial_Cells", "Spores", "Glucose", "Germinant"]
plot_labels = ["Bacterial Cells", "Spores", "Glucose", "Germinant"]

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (var, label) in enumerate(zip(variable_names, plot_labels)):
    ax = axes[idx]
    
    for k, step_num in enumerate(saved_steps):
        r, v = saved_data[var][k]
        
        sorted_indices = np.argsort(r)
        r_sorted = r[sorted_indices]
        v_sorted = v[sorted_indices]

        # Calculate time for label
        step_time = step_num * dt
        ax.plot(r_sorted, v_sorted, linewidth=2,
                label=f"Step {step_num} (t={step_time:.2f})")

    ax.set_title(label)
    ax.set_xlabel("Radius (r) [ND]")
    ax.set_ylabel(label)
    ax.set_xlim([0, R])
    ax.grid(True)
    ax.legend()

fig.suptitle("Radial Profiles (Booster Model L=sqrt(D/r))", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(savedir + '/plots.pdf', dpi=300)
plt.show()