import os
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *

set_log_level(LogLevel.ERROR)

# ---------------------------------------------------------
# 1. PARAMETERS
# ---------------------------------------------------------
E = 210.0e3        # Young's modulus [MPa]
nu = 0.3           # Poisson's ratio
Gc = 2.7           # Fracture toughness [N/mm]
l = 0.02           # Length scale [mm] 
alpha_T = 56.25    # Fatigue threshold [N/mm^2]

tol = 0.001
max_iter = 100

lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))

# Loading parameters
num_cycles = 50
steps_per_cycle = 8
u_max = 4.0e-3
time_array = np.linspace(0, num_cycles, num_cycles * steps_per_cycle)

# Meshing parameters
Resolution = 25
num_refinements = 2 
# Create dynamic folder name based on properties
folder_name = f"results_E{E}_Gc{Gc}_l{l}_aT{alpha_T}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Fatigue Degradation Function (Asymptotic)
def f_alpha(a):
    return conditional(le(a, alpha_T), 1.0, (2.0 * alpha_T / (a + alpha_T))**2)

# ---------------------------------------------------------
# 2. MESH & FUNCTION SPACES (Locally Refined)
# ---------------------------------------------------------
# Step 1: Start with a coarse background mesh
# 25x25 gives an initial element size of ~0.04 mm
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), Resolution, Resolution, "crossed")

# Step 2: Define the horizontal strip where the crack will propagate
# The crack is at y=0.5. We will refine a strip from y=0.4 to y=0.6.
class RefinementZone(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.4 and x[1] <= 0.6

refinement_zone = RefinementZone()

# Step 3: Iteratively refine only the cells inside the strip
# To resolve l = 0.02, we need h <= 0.004. 
# Starting at h=0.04, four bisections (refinements) drops h down to ~0.0025 in the strip.
for _ in range(num_refinements):
    # Create a boolean marker for the cells
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    
    # Mark cells inside the strip as True
    refinement_zone.mark(cell_markers, True)
    
    # Refine the mesh based on the markers
    mesh = refine(mesh, cell_markers)

print(f"\033[1;32mMesh generated: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices\033[0m")

# Function spaces remain exactly the same
V_u = VectorFunctionSpace(mesh, "CG", 1)  
V_d = FunctionSpace(mesh, "CG", 1)        
V_local = FunctionSpace(mesh, "DG", 0)

u = Function(V_u, name="Displacement")
du = TrialFunction(V_u)
v_u = TestFunction(V_u)

d = Function(V_d, name="Damage")
dd = TrialFunction(V_d)
v_d = TestFunction(V_d)

H_n = Function(V_local, name="H_prev_step")
H_hist = Function(V_local, name="Max_Energy")
alpha_bar = Function(V_local, name="Fatigue_Odometer")
alpha_bar_n = Function(V_local, name="Odometer_prev_step")
alpha_prev = Function(V_local, name="Previous_Alpha")
d_iter_prev = Function(V_d)

# ---------------------------------------------------------
# 3. BOUNDARY CONDITIONS & MEASURES
# ---------------------------------------------------------
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0) and on_boundary

class Crack(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0.5 and near(x[1], 0.5, 0.01)

# Mark boundaries for force integration
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top_boundary = Top()
top_boundary.mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
n = FacetNormal(mesh)

bc_notch = DirichletBC(V_d, Constant(1.0), Crack())
bc_notch.apply(d.vector())

u_top = Expression("u_disp", u_disp=0.0, degree=1)
bc_u_bottom = DirichletBC(V_u, Constant((0.0, 0.0)), Bottom())
bc_u_top = DirichletBC(V_u.sub(1), u_top, Top())
bcs_u = [bc_u_bottom, bc_u_top]

# ---------------------------------------------------------
# 4. ENERGIES & WEAK FORMS
# ---------------------------------------------------------
def epsilon(u):
    return sym(grad(u))

def psi_plus(u):
    eps = epsilon(u)
    return 0.5 * lmbda * tr(eps)**2 + mu * inner(eps, eps)

# Displacement
sigma = (1.0 - d)**2 * (lmbda * tr(epsilon(u)) * Identity(2) + 2.0 * mu * epsilon(u))
F_u = inner(sigma, epsilon(v_u)) * dx
J_u = derivative(F_u, u, du)

problem_u = NonlinearVariationalProblem(F_u, u, bcs_u, J_u)
solver_u = NonlinearVariationalSolver(problem_u)

# Phase-Field
F_d = ( -2.0 * (1.0 - d) * H_hist * v_d + \
        Gc * l * f_alpha(alpha_bar) * inner(grad(d), grad(v_d)) + \
        (Gc / l) * f_alpha(alpha_bar) * d * v_d ) * dx
J_d = derivative(F_d, d, dd)

d_min = interpolate(Constant(0.0), V_d)
bc_d = [bc_notch] 
problem_d = NonlinearVariationalProblem(F_d, d, bc_d, J_d)
problem_d.set_bounds(d_min, interpolate(Constant(1.0), V_d))

solver_d = NonlinearVariationalSolver(problem_d)
solver_d.parameters["nonlinear_solver"] = "snes"
solver_d.parameters["snes_solver"]["method"] = "vinewtonssls" 
solver_d.parameters["snes_solver"]["absolute_tolerance"] = 1e-5
solver_d.parameters["snes_solver"]["relative_tolerance"] = 1e-5
solver_d.parameters["snes_solver"]["maximum_iterations"] = 200
solver_d.parameters["snes_solver"]["error_on_nonconvergence"] = False 

# ---------------------------------------------------------
# 5. EXPORT SETUP
# ---------------------------------------------------------
file_results = XDMFFile(f"{folder_name}/fatigue_results.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

V_vis = FunctionSpace(mesh, "CG", 1)
alpha_vis = Function(V_vis, name="Fatigue_Odometer")

# Data tracking lists for plotting
disp_list = []
force_list = []
time_list = []
alpha_max_list = []
max_force_cycle_list = []
cycle_indices = []
current_cycle_max_force = 0.0
# ---------------------------------------------------------
# 6. CYCLIC SOLVER LOOP
# ---------------------------------------------------------
print("\033[1;36m==================================================\033[0m")
print("\033[1;36m       PHASE-FIELD FATIGUE SIMULATION SETUP       \033[0m")
print("\033[1;36m==================================================\033[0m")
print(f"\033[1;33mMaterial Properties:\033[0m")
print(f"  Young's Modulus (E) : {E} MPa")
print(f"  Poisson's Ratio (nu): {nu}")
print(f"  Fracture Toughness  : {Gc} N/mm")
print(f"  Length Scale (l)    : {l} mm")
print(f"  Fatigue Threshold   : {alpha_T} N/mm^2")
print(f"\n\033[1;33mSimulation Parameters:\033[0m")
print(f"  Total Cycles        : {num_cycles}")
print(f"  Steps per Cycle     : {steps_per_cycle}")
print(f"  Max Displacement    : {u_max} mm")
print("\033[1;36m==================================================\033[0m\n")

for step, t in enumerate(time_array):
    cycle_fraction = t % 1.0
    if cycle_fraction <= 0.5:
        current_disp = u_max * (cycle_fraction / 0.5)
    else:
        current_disp = u_max * (1.0 - (cycle_fraction - 0.5) / 0.5)
        
    u_top.u_disp = current_disp
    
    err = 1.0
    iteration = 0
    
    # INNER STAGGERED LOOP
    # INNER STAGGERED LOOP
    while err > tol and iteration < max_iter:
        d_iter_prev.vector()[:] = d.vector()[:]
        
        solver_u.solve()
        
        psi_proj = project(psi_plus(u), V_local)
        H_hist.vector()[:] = np.maximum(H_n.vector()[:], psi_proj.vector()[:])
        
        diff = psi_proj.vector()[:] - alpha_prev.vector()[:]
        increment = np.where(diff > 0, diff, 0.0) 
        alpha_bar.vector()[:] = alpha_bar_n.vector()[:] + increment
        
        solver_d.solve()
        
        diff_d = d.vector()[:] - d_iter_prev.vector()[:]
        err = np.linalg.norm(diff_d, ord=np.inf)
        
        # --- NEW PRINT STATEMENT FOR INNER LOOP ITERATIONS ---
        print(f"    -> Staggered Iter {iteration + 1}: max \u0394d = {err:.3e}")
        
        iteration += 1
        
    # TIME STEP CONVERGED. UPDATE GLOBAL HISTORIES
    d_min.vector()[:] = d.vector()[:]
    alpha_prev.vector()[:] = psi_proj.vector()[:]
    alpha_bar_n.vector()[:] = alpha_bar.vector()[:]
    H_n.vector()[:] = H_hist.vector()[:]
    
    # Calculate Total Force via integration of traction over top boundary
    T_y = dot(sigma, n)[1]
    total_force = assemble(T_y * ds(1))

    current_cycle_max_force = max(current_cycle_max_force, total_force)
    
    # If this step completes a full cycle (e.g., step 8, 16, 24...), save the max and reset
    if step > 0 and step % steps_per_cycle == 0:
        max_force_cycle_list.append(current_cycle_max_force)
        cycle_indices.append(int(t))
        current_cycle_max_force = 0.0 # Reset for the next cycle
    
    # Store data for plotting
    disp_list.append(current_disp)
    force_list.append(total_force)
    time_list.append(t)
    alpha_max_list.append(alpha_bar.vector().max())
    
    # Write to XDMF
    file_results.write(u, t)
    file_results.write(d, t)
    alpha_vis.assign(project(alpha_bar, V_vis))
    file_results.write(alpha_vis, t)
    
    print(f"\033[1;32m[Step {step:03d} | Time: {t:.2f}]\033[0m Disp: {current_disp:.5f}, \033[1;35mIters: {iteration}\033[0m, Force: {total_force:.3f}, Max a_bar: \033[1;31m{alpha_max_list[-1]:.2f}\033[0m")

print(f"\n\033[1;32mSimulation Complete. Generating plots in '{folder_name}'...\033[0m")

# ---------------------------------------------------------
# 7. DATA EXPORT & PLOTTING
# ---------------------------------------------------------
print(f"\n\033[1;32mSaving plot data to text files in '{folder_name}'...\033[0m")

# Export 1: Force vs Displacement
np.savetxt(f"{folder_name}/data_Force_vs_Disp.txt", 
           np.column_stack((disp_list, force_list)), 
           header="Displacement_[mm] Total_Force_[N/mm]", 
           comments='', fmt='%.6e')

# Export 2: Alpha bar vs Time (Cycles)
np.savetxt(f"{folder_name}/data_Alpha_vs_Cycles.txt", 
           np.column_stack((time_list, alpha_max_list)), 
           header="Time_[Cycles] Max_Alpha_bar", 
           comments='', fmt='%.6e')

# Export 3: Max Force vs Cycle Number
np.savetxt(f"{folder_name}/data_Max_Force_vs_Cycles.txt", 
           np.column_stack((cycle_indices, max_force_cycle_list)), 
           header="Cycle_Number Max_Force_[N/mm]", 
           comments='', fmt='%.6e')


print("\033[1;32mGenerating plots...\033[0m")

# Plot 1: Force-Displacement Curve
plt.figure(figsize=(8, 6))
plt.plot(disp_list, force_list, '-k', linewidth=1.5)
plt.xlabel(r'Displacement $\bar{u}$ [mm]', fontsize=14)
plt.ylabel('Total Force F [N/mm]', fontsize=14)
plt.title('Cyclic Force-Displacement Curve', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{folder_name}/Force_vs_Displacement.png", dpi=300)

# Plot 2: Fatigue Odometer Accumulation
plt.figure(figsize=(8, 6))
plt.plot(time_list, alpha_max_list, '-b', linewidth=2)
plt.axhline(y=alpha_T, color='r', linestyle='--', label=r'Threshold $\alpha_T$')
plt.xlabel('Cycles (N)', fontsize=14)
plt.ylabel(r'Max Cumulated History $\bar{\alpha}$', fontsize=14)
plt.title('Fatigue Accumulation over Cycles', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{folder_name}/Alpha_bar_vs_Cycles.png", dpi=300)

# Plot 3: Max Force vs. Cycle Number
plt.figure(figsize=(8, 6))
plt.plot(cycle_indices, max_force_cycle_list, '-ro', linewidth=2, markersize=6)
plt.xlabel('Cycle Number (N)', fontsize=14)
plt.ylabel('Peak Force $F_{max}$ [N/mm]', fontsize=14)
plt.title('Maximum Reaction Force Degradation per Cycle', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

# --- THE X-AXIS FIX ---
# Calculate a step size to only show ~10 ticks, ensuring it never goes below a step of 1.
tick_step = max(1, len(cycle_indices) // 10)
plt.xticks(cycle_indices[::tick_step]) 

plt.tight_layout()
plt.savefig(f"{folder_name}/Max_Force_vs_Cycles.png", dpi=300)

print("\033[1;32mPlots successfully saved.\033[0m")