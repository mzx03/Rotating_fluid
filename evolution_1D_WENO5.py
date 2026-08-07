import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# =========================== #
# Parameters and EOS
# =========================== #

def rho2h(rho):    
    gamma = 4.0 #4/3
    K = 1.0
    rho = np.maximum(rho, 0)
    return K * gamma/(gamma-1) * rho**(gamma-1)

def h2rho(h):    
    gamma = 4.0
    K = 1.0
    h = np.maximum(h, 0)
    return (h * (gamma-1) / (K * gamma))**(1/(gamma-1))

def rho2P(rho):
    gamma = 4.0
    K = 1.0
    P = K*rho**gamma
    return P    

def P2rho(P):
    gamma = 4.0
    K = 1.0
    rho = (P/K)**(1/gamma)
    return rho 

def create_grid_with_ghost(N_cells, r_max, n_ghost=3):

    dr = r_max / N_cells
    r_face = np.linspace(0, r_max, N_cells + 1)
    
    # cell-centered
    r_center_physical = 0.5 * (r_face[:-1] + r_face[1:])
        
    N_total = N_cells + 2 * n_ghost
    r_center = np.zeros(N_total)
    
    # physcial range
    idx_start = n_ghost
    idx_end = n_ghost + N_cells
    r_center[idx_start:idx_end] = r_center_physical
    
    # left ghost
    for i in range(n_ghost):
        r_center[n_ghost-1-i] = -r_center_physical[i]
    
    # right ghost
    for i in range(n_ghost):
        r_center[idx_end + i] = r_max + (i + 1) * dr
    
    return r_face, r_center, dr

def P2C(rho, vr, vphi, r_center):
    # primitive to conserved
    # U = [rho, rho v^r, rho r² v^phi]    
    U = np.zeros((3, len(r_center)))
    U[0] = rho                         # density
    U[1] = rho * vr                    # momentum
    U[2] = rho * r_center**2 * vphi    # angular momentum
    return U


def C2P(U, r_center):
    # conserved to primitive
    rho = np.maximum(U[0], 0)
    vr = U[1] / (rho + 0)
    vphi = U[2] / (rho * r_center**2 + 0)
    return rho, vr, vphi

# =========================== 
# WENO5 reconstruction
# =========================== 


def weno5_reconstruction(U, n_ghost=3):

    N = U.shape[1]
    N_physical = N - 2 * n_ghost
    
    U_left = np.zeros_like(U)
    U_right = np.zeros_like(U)
    
    d = np.array([1/10, 6/10, 3/10])
    eps = 1e-6
    
    start_idx = n_ghost 
    end_idx = n_ghost + N_physical
    
    # normalization factor
    #scales = np.array([np.std(U[0]), np.std(U[1]), np.std(U[2])])
    #scales = np.maximum(scales, 1e-6)
    scales = np.ones(3)
    for i in range(N):
        if start_idx <= i < end_idx:
            #calculate shared smooth indicator
            beta0_combined = 0.0
            beta1_combined = 0.0
            beta2_combined = 0.0
            
            beta0_r_combined = 0.0
            beta1_r_combined = 0.0
            beta2_r_combined = 0.0
            
            for eq in range(3):
                v = U[eq] / scales[eq]  # normalize
                
                # left
                beta0 = 13/12 * (v[i-2] - 2*v[i-1] + v[i])**2 + \
                        1/4 * (v[i-2] - 4*v[i-1] + 3*v[i])**2
                beta1 = 13/12 * (v[i-1] - 2*v[i] + v[i+1])**2 + \
                        1/4 * (v[i-1] - v[i+1])**2
                beta2 = 13/12 * (v[i] - 2*v[i+1] + v[i+2])**2 + \
                        1/4 * (3*v[i] - 4*v[i+1] + v[i+2])**2
                
                # maximal smooth indicator
                beta0_combined = max(beta0_combined, beta0)
                beta1_combined = max(beta1_combined, beta1)
                beta2_combined = max(beta2_combined, beta2)
                
                # right
                beta0_r = 13/12 * (v[i+2] - 2*v[i+1] + v[i])**2 + \
                          1/4 * (v[i+2] - 4*v[i+1] + 3*v[i])**2
                beta1_r = 13/12 * (v[i+1] - 2*v[i] + v[i-1])**2 + \
                          1/4 * (v[i+1] - v[i-1])**2
                beta2_r = 13/12 * (v[i] - 2*v[i-1] + v[i-2])**2 + \
                          1/4 * (3*v[i] - 4*v[i-1] + v[i-2])**2
                
                beta0_r_combined = max(beta0_r_combined, beta0_r)
                beta1_r_combined = max(beta1_r_combined, beta1_r)
                beta2_r_combined = max(beta2_r_combined, beta2_r)
            
            #WENO weight
            alpha0 = d[0] / (eps + beta0_combined)**2
            alpha1 = d[1] / (eps + beta1_combined)**2
            alpha2 = d[2] / (eps + beta2_combined)**2
            w_sum = alpha0 + alpha1 + alpha2
            
            w0_L = alpha0 / w_sum
            w1_L = alpha1 / w_sum
            w2_L = alpha2 / w_sum
            
            # right
            alpha0_r = d[0] / (eps + beta0_r_combined)**2
            alpha1_r = d[1] / (eps + beta1_r_combined)**2
            alpha2_r = d[2] / (eps + beta2_r_combined)**2
            w_sum_r = alpha0_r + alpha1_r + alpha2_r
            
            w0_R = alpha0_r / w_sum_r
            w1_R = alpha1_r / w_sum_r
            w2_R = alpha2_r / w_sum_r
            
            # for each component
            for eq in range(3):
                v = U[eq]
                
                # left
                v0 = (2*v[i-2] - 7*v[i-1] + 11*v[i]) / 6
                v1 = (-v[i-1] + 5*v[i] + 2*v[i+1]) / 6
                v2 = (2*v[i] + 5*v[i+1] - v[i+2]) / 6
                
                U_left[eq, i] = w0_L * v0 + w1_L * v1 + w2_L * v2
                
                # right
                v0_r = (2*v[i+2] - 7*v[i+1] + 11*v[i]) / 6
                v1_r = (-v[i+1] + 5*v[i] + 2*v[i-1]) / 6
                v2_r = (2*v[i] + 5*v[i-1] - v[i-2]) / 6
                
                U_right[eq, i] = w0_R * v0_r + w1_R * v1_r + w2_R * v2_r
                
        else:
            # Ghost cells
            U_left[:, i] = U[:, i]
            U_right[:, i] = U[:, i]
    # left: u^-_{i+1/2} right: u^+_{i-1/2}
    return U_left, U_right


def weno5_flux(U, r_center, flux_function, n_ghost=3):

    N = U.shape[1]
    N_physical = N - 2 * n_ghost
    F_num = np.zeros((3, N+1))
    
    gamma = 4.0
    K = 1.0
        
    U_left, U_right = weno5_reconstruction(U, n_ghost)
    
    # 
    for i in range(n_ghost, n_ghost + N_physical):
        UL = U_left[:, i].copy()      #  ith  cell right
        UR = U_right[:, i+1].copy()   #  i+1 th cell left
        
        # density positive
        UL[0] = np.maximum(UL[0], 0)
        UR[0] = np.maximum(UR[0], 0)
        
        
        FL = flux_function(UL)
        FR = flux_function(UR)
        
        # max speed
        rho_L = UL[0]
        rho_R = UR[0]
        vr_L = UL[1] / rho_L
        vr_R = UR[1] / rho_R
        
        cs_L = np.sqrt(gamma * K * rho_L**(gamma-1) + 0)
        cs_R = np.sqrt(gamma * K * rho_R**(gamma-1) + 0)
        
        max_speed = max(abs(vr_L) + cs_L, abs(vr_R) + cs_R)
        
        # Lax-Friedrichs
        for eq in range(3):
            F_num[eq, i+1] = 0.5 * (FL[eq] + FR[eq]) - 0.5 * max_speed * (UR[eq] - UL[eq]) 
    #print("F_num[1]", F_num[1,0], F_num[1,1], F_num[1,2], F_num[1,3], F_num[1,4], F_num[1,5])
    return F_num, U_left, U_right

# ==========================
# Derivative
# ==========================

def derivative_centered(f, r_center, dr):
    # ordinary central derivative (not for flux)
    N = len(f)
    dfdr = np.zeros(N)
    
    # interior 
    dfdr[2:-2] = (f[3:-1] - f[1:-3]) / (2*dr)
    
    # boundary 
    dfdr[0] = (-3*f[0] + 4*f[1] - f[2]) / (2*dr)
    dfdr[1] = (-3*f[1] + 4*f[2] - f[3]) / (2*dr)
    dfdr[-2] = (3*f[-2] - 4*f[-3] + f[-4]) / (2*dr)
    dfdr[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (2*dr)
    
    return dfdr


def flux_divergence(F_num, r_face, r_center, n_ghost=3):

    # divergence for flux, calculate (1/r) * d/dr(r*F_num)
    # r_center is cell center r_face is grid-center

    N_physical = len(r_face) - 1
    N_total = len(r_center)
    dr = r_face[1] - r_face[0]
    
    divF = np.zeros((3, N_total))
    
    # in physical range
    for i in range(n_ghost, n_ghost + N_physical):
        r_i = r_center[i]

        # cell-centered grid, no need to worry about r=0
        r_face_right = r_face[i - n_ghost + 1]
        r_face_left = r_face[i - n_ghost]
        
        divF[:, i] = (r_face_right * F_num[:, i+1] - r_face_left * F_num[:, i]) / (r_i * dr)

    
    return divF

def fill_ghost_cells(U, r_center, n_ghost=3):

    U_filled = U.copy()
    N = U.shape[1]
    for i in range(n_ghost):
        ghost_idx = n_ghost - 1 - i
        mirror_idx = n_ghost + i
        
        # density is even: rho(r) = rho(-r)
        U_filled[0, ghost_idx] = U_filled[0, mirror_idx]
        
        # linear momentum density is odd: rho(r) v^r(r) = -rho(-r) v^r(-r)
        U_filled[1, ghost_idx] = -U_filled[1, mirror_idx]
        
        # angular momentum density is even: rho(r) r^2 v^\phi(r) = rho(-r) (-r)^2 v^\phi(-r)
        U_filled[2, ghost_idx] = U_filled[2, mirror_idx]
    
    for i in range(n_ghost):
        ghost_idx = N - n_ghost + i
        inner_idx = N - n_ghost - 1 - i
        # zero flux for outer boundary
        U_filled[:, ghost_idx] = U_filled[:, inner_idx]*10  
    return U_filled


# =========================== #
# Flux and source
# =========================== #

def flux(U):
    # F = [rho v^r, rho (v^r)^2, rho r² v^phi v^r]
    rho = U[0]
    vr = U[1] / (rho + 0)
    
    F = np.zeros_like(U)
    F[0] = U[1] 
    F[1] = U[1] * vr + rho2P(rho)
    F[2] = U[2] * vr  
    return F


def source(U, r, dr, mu=0.1):
    
    rho = np.maximum(U[0], 0)
    vr = U[1] / (rho + 0)
    vphi = U[2] / (rho * r**2 + 0)
    
    # pressure gradient: treate rho*\partial_r h as source
    #h = rho2h(rho)
    #dhdr = derivative_centered(h, r, dr)
    
    S = np.zeros_like(U)
    
    # centrifugal force
    centrifugal = np.zeros_like(r)
    #mask = np.abs(r) > 0
    centrifugal = U[2]**2 / (rho * r**3)

    # pressure geometric term: treate \nabla P in flux  
    P = rho2P(rho)    
    pressure_geom = P / r 
    
    # viscosity NOT TESTED
    if mu > 0:
        dvr_dr = derivative_centered(vr, r, dr)

        r_dvr_dr = r * dvr_dr
        laplacian_vr = np.zeros_like(r)
        laplacian_vr = derivative_centered(r_dvr_dr, r, dr) / r
                
        # radial viscosity
        viscous_radial = mu * (laplacian_vr - vr / (r**2 + 0))

        # angular viscosity
        dvphi_dr = derivative_centered(vphi, r, dr)
        flux = mu * r**3 * dvphi_dr
        viscous_azimuthal = derivative_centered(flux, r, dr) / (r + 0)
    else:
        viscous_radial = np.zeros_like(r)
        viscous_azimuthal = np.zeros_like(r)
    
    # source
    #S[1] = -rho * dhdr + centrifugal + viscous_radial  # radial
    S[1] = centrifugal + pressure_geom + viscous_radial  # radial
    S[2] = 0#viscous_azimuthal  # angular
    #print(viscous_radial[10], viscous_radial[10])
    return S

# =========================== #
# right hand side
# =========================== #

def rhs(U, r_face, r_center, dr, n_ghost=3, t=None):    
    N_physical = len(r_center) - 2 * n_ghost
    # fill ghost cell (bc applied)
    U_filled = fill_ghost_cells(U, r_center, n_ghost)

    # flux and divergence
    # F_num on grid center
    F_num, U_left, U_right = weno5_flux(U_filled, r_center, flux, n_ghost)    
    
    last_phys_interface = n_ghost + N_physical
    
    rho_boundary = np.maximum(U_left[0, last_phys_interface-1], 0)
    P_boundary = rho2P(rho_boundary)
    
    # mass flux = 0
    F_num[0, last_phys_interface] = 0.0
    F_num[0, last_phys_interface + 1] = 0.0
    
    # momentum flux = P
    F_num[1, last_phys_interface] = P_boundary
    F_num[1, last_phys_interface + 1] = P_boundary
    #if t is not None: Not tested
    #    F_num[1, last_phys_interface] = 0.95*P_boundary + 0.05*P_boundary*np.cos(2*np.pi*t)
    #    F_num[1, last_phys_interface + 1] = 0.95*P_boundary + 0.05*P_boundary*np.cos(2*np.pi*t)
    
    # angular momentum flux = 0
    F_num[2, last_phys_interface] = 0.0
    F_num[2, last_phys_interface + 1] = 0.0
    divF = flux_divergence(F_num, r_face, r_center, n_ghost)
        
    # source
    S = source(U_filled, r_center, dr, -0.1) #viscosity is not tested
    
    # RHS
    RHS = S - divF
    #print("divF[1]",divF[1,0],divF[1,1],divF[1,2],divF[1,3],divF[1,4],divF[1,5])
    #print("S[1]",S[1,0],S[1,1],S[1,2],S[1,3],S[1,4],S[1,5])
    return RHS


def RK3_step(U, r_face, r_center, dr, dt, n_ghost=3, t=None, U_boundary=None, x_fixed=0.0):

    N_physical = len(r_center) - 2 * n_ghost
    idx_physical = slice(n_ghost, n_ghost + N_physical) #index of physical range
    #n_fixed = int(x_fixed/dr)
    n_fixed = 5
    # 
    idx_fixed_left = slice(n_ghost, n_ghost + n_fixed)  # first five
    #idx_fixed_right = slice(n_ghost + N_physical - n_fixed, n_ghost + N_physical)  # last five
    idx_fixed_right = slice(n_ghost + N_physical - n_fixed, n_ghost + N_physical)  # last five
    # Stage 1
    k1 = rhs(U, r_face, r_center, dr, n_ghost,t)
    U1 = U.copy()
    U1[:, idx_physical] = U[:, idx_physical] + dt * k1[:, idx_physical]
    
    if U_boundary is not None:
        U1[:, idx_fixed_left] = U_boundary[:, idx_fixed_left]
        U1[:, idx_fixed_right] = U_boundary[:, idx_fixed_right]

    
    U1 = fill_ghost_cells(U1, r_center, n_ghost)
    U1[0] = np.maximum(U1[0], 0)
    
    
    # Stage 2
    k2 = rhs(U1, r_face, r_center, dr, n_ghost,t)
    U2 = U.copy()
    U2[:, idx_physical] = 0.75 * U[:, idx_physical] + 0.25 * (U1[:, idx_physical] + dt * k2[:, idx_physical])
    
    if U_boundary is not None:
        U2[:, idx_fixed_left] = U_boundary[:, idx_fixed_left]
        U2[:, idx_fixed_right] = U_boundary[:, idx_fixed_right]       
    
    U2 = fill_ghost_cells(U2, r_center, n_ghost)
    U2[0] = np.maximum(U2[0], 0)
    
    # Stage 3
    k3 = rhs(U2, r_face, r_center, dr, n_ghost,t)
    U_new = U.copy()
    U_new[:, idx_physical] = (1.0/3.0) * U[:, idx_physical] + (2.0/3.0) * (U2[:, idx_physical] + dt * k3[:, idx_physical])
    
    if U_boundary is not None:
        U_new[:, idx_fixed_left] = U_boundary[:, idx_fixed_left]
        U_new[:, idx_fixed_right] = U_boundary[:, idx_fixed_right]
        #rho, vr, vphi = C2P(U_new, r_center)
        #vphi = 0.1*np.ones(len(r_center))
        #vr = 0.0*np.ones(len(r_center))
        #U_new = P2C(rho, vr, vphi, r_center)        
    
    U_new = fill_ghost_cells(U_new, r_center, n_ghost)
    U_new[0] = np.maximum(U_new[0], 0)
    
    return U_new

def compute_dt(U, r_center, dr, CFL=0.5, n_ghost=3):    
    N_physical = len(r_center) - 2 * n_ghost
    
    # physical range
    idx_start = n_ghost
    idx_end = n_ghost + N_physical
    
    rho = U[0, idx_start:idx_end]
    vr = np.abs(U[1, idx_start:idx_end] / (rho + 0))
    
    # speed of sound
    gamma = 4.0
    K = 1.0
    cs = np.sqrt(gamma * K * rho**(gamma-1) + 0)
    max_speed = np.max(vr + cs)
    dt = CFL * dr / (max_speed )
    
    return dt

# =========================== #
# Evolution
# =========================== #

def evolve(U0, r_face, r_center, dr, t_final, name, CFL=0.5, 
                output_interval=0.1, output_dir="./output", n_ghost=3):
        
    os.makedirs(output_dir, exist_ok=True)
    
    U = U0.copy()
    t = 0.0
    step = 0
    frame = 0
    U_boundary = U0.copy()
    
    t = 0.0
    step = 0
    frame = 0
    
    N_physical = len(r_center) - 2 * n_ghost 
    idx_physical = slice(n_ghost, n_ghost + N_physical)
    r_physical = r_center[idx_physical]
    
    # initial conserved quantities
    rho0, _, vphi0 = C2P(U0[:, idx_physical], r_physical)
    mass0 = 2 * np.pi * np.sum(rho0 * r_physical) * dr
    J0 = 2 * np.pi * np.sum(rho0 * r_physical**3 * vphi0) * dr
    
    print(f"initial mass: {mass0:.6f}")
    print(f"initial angular momentum: {J0:.6f}")
    print("-" * 60)
    
    next_output = 0.0
    
    while t < t_final:

        if t >= next_output or t==0.0:            
            rho, vr, vphi = C2P(U[:, idx_physical], r_physical)
            mass = 2 * np.pi * np.sum(rho * r_physical) * dr
            J = 2 * np.pi * np.sum(rho * r_physical**3 * vphi) * dr
            if t==0.0:
                dt =0
                output_data = np.column_stack([r_physical, rho, vr, vphi])
                np.savetxt(f"evolution_initial_{name}_{N_physical:04d}_{CFL:.3f}.txt", output_data)
            print(f"t={t:.3f}: step={step}, dt={dt:.2e}, "
                  f"mass_err={(mass-mass0)/mass0:.2e}, J_err={(J-J0)/J0:.2e}")
            
            # plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            axes[0, 0].plot(r_physical, rho, 'b-', lw=2)
            axes[0, 0].set_ylabel('ρ')
            axes[0, 0].set_title(f'Density (t={t:.3f}), mass_err={(mass-mass0)/mass0:.2e}')
            axes[0, 0].grid(True)
            axes[0, 0].set_xlim([0, r_max])
            #axes[0, 0].set_ylim([0.9, 1.1])  #for steady
            #axes[0, 0].set_ylim([0.99, 1.01]) #for Gauss
            axes[0, 0].set_ylim([0.95, 1.05])  #for shock       

            axes[0, 1].plot(r_physical, vr, 'b-', lw=2)
            axes[0, 1].set_ylabel('v^r')
            axes[0, 1].set_title('Radial Velocity')
            axes[0, 1].grid(True)
            axes[0, 1].set_xlim([0, r_max])
            #axes[0, 1].set_ylim([-0.02, 0.02])  #for steady
            #axes[0, 1].set_ylim([-0.01, 0.01])  #for Gauss
            axes[0, 1].set_ylim([-0.05, 0.05])  #for shock

            axes[1, 0].plot(r_physical, vphi, 'b-', lw=2)
            axes[1, 0].set_ylabel('v^φ')
            axes[1, 0].set_xlabel('r')
            axes[1, 0].set_title('Angular Velocity')
            axes[1, 0].grid(True)
            axes[1, 0].set_xlim([0, r_max])
            axes[1, 0].set_ylim([-0.05, 0.15])  

            axes[1, 1].plot(r_physical, rho * r_physical**2 * vphi, 'b-', lw=2)
            axes[1, 1].set_ylabel('Angular Momentum Density')
            axes[1, 1].set_xlabel('r')
            axes[1, 1].set_title(f'J_err={(J-J0)/J0:.2e}')
            axes[1, 1].grid(True)
            axes[1, 1].set_xlim([0, r_max])
            #axes[1, 1].set_ylim([-0.05, 15])  # for steady
            axes[1, 1].set_ylim([-0.05, 0.3])  # for Gauss

            plt.tight_layout()
            plt.savefig(f"{output_dir}/frame_{frame:04d}.png", dpi=150)
            plt.close()
            frame += 1
            next_output += output_interval            

        dt = compute_dt(U, r_center, dr, CFL, n_ghost)
        if t + dt > t_final:
            dt = t_final - t
        #U = RK3_step(U, r_face, r_center, dr, dt, n_ghost)
        U = RK3_step(U, r_face, r_center, dr, dt, n_ghost, t) 
        #U = RK3_step(U, r_face, r_center, dr, dt, n_ghost, U_boundary)
        t += dt
        step += 1
        # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(r_physical, rho, 'b-', lw=2)
    axes[0, 0].set_ylabel('ρ')
    axes[0, 0].set_title(f'Density (t={t:.3f}), mass_err={(mass-mass0)/mass0:.2e}')
    axes[0, 0].grid(True)
    axes[0, 0].set_xlim([0, r_max])
    axes[0, 0].set_ylim([0.9, 1.1])  #for steady
    #axes[0, 0].set_ylim([0.99, 1.01]) #for Gauss
    axes[0, 0].set_ylim([0.95, 1.05])  #for shock           

    axes[0, 1].plot(r_physical, vr, 'b-', lw=2)
    axes[0, 1].set_ylabel('v^r')
    axes[0, 1].set_title('Radial Velocity')
    axes[0, 1].grid(True)
    axes[0, 1].set_xlim([0, r_max])
    #axes[0, 1].set_ylim([-0.02, 0.02])  #for steady
    #axes[0, 1].set_ylim([-0.01, 0.01])  #for Gauss  
    axes[0, 1].set_ylim([-0.05, 0.05])  #for shock

    axes[1, 0].plot(r_physical, vphi, 'b-', lw=2)
    axes[1, 0].set_ylabel('v^φ')
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_title('Angular Velocity')
    axes[1, 0].grid(True)
    axes[1, 0].set_xlim([0, r_max])
    axes[1, 0].set_ylim([-0.05, 0.15])  

    axes[1, 1].plot(r_physical, rho * r_physical**2 * vphi, 'b-', lw=2)
    axes[1, 1].set_ylabel('Angular Momentum Density')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_title(f'J_err={(J-J0)/J0:.2e}')
    axes[1, 1].grid(True)
    axes[1, 1].set_xlim([0, r_max])
    #axes[1, 1].set_ylim([-0.05, 15])  # for steady
    axes[1, 1].set_ylim([-0.05, 0.3])  # for Gauss

    plt.tight_layout()
    plt.savefig(f"{output_dir}/frame_{frame:04d}.png", dpi=150)
    plt.close()    
    print("-" * 60)
    print(f"finished: {step} step, t={t:.3f}")
    
    return U


# =========================== #
# initialize
# =========================== #

def initial_condition(r_center, name, rho0=1.0, omega0=0, n_ghost=3,
                      x1=2.0, x2=4.0, x3=6.0): 
    N_total = len(r_center)
    
    # density
    rho = rho0 * np.ones(N_total)
    rho = np.maximum(rho, 1e-6)
    
    # radial velocity
    vr = np.zeros(N_total)
    
    if name=='shock':
        mask_1 = (r_center >= x1) & (r_center < x2)
        vr[mask_1] = 0.005    
        mask_2 = (r_center >= x2) & (r_center < x3)
        vr[mask_2] = -0.02
        vphi = omega0 * np.exp(-r_center**2 / (2 * 2**2)) # Gaussian profile  
    elif name=='Gauss':
        vr = np.zeros(N_total)
        vphi = omega0 * np.exp(-r_center**2 / (2 * 2**2)) # Gaussian profile  
    elif name=='steady':
        vphi = omega0 * np.ones(N_total) # uniform     
        ber = 1
        h = ber + 0.5*vphi**2*r_center**2
        rho = h2rho(h)
    else:        
        print(ValueError ,"Not valid name")

    return rho, vr, vphi


# =========================== #
# MAIN      
# =========================== #

if __name__ == "__main__":
    name = sys.argv[1]
    N_cells = int(sys.argv[2])
    n_ghost = 3     
    r_max = 10.0
    
    r_face, r_center, dr = create_grid_with_ghost(N_cells, r_max, n_ghost)    
    N_physical = N_cells
    idx_physical = slice(n_ghost, n_ghost + N_physical)
    r_physical = r_center[idx_physical]
        
    rho0, vr0, vphi0 = initial_condition(r_center, name, rho0=1.0, omega0=0.1, n_ghost=n_ghost)
    
    U0 = P2C(rho0, vr0, vphi0, r_center)
    
    U0 = fill_ghost_cells(U0, r_center, n_ghost)
    
    
    t_final = 5
    CFL = float(sys.argv[3])
    output_interval = 5e-2
    

    U_final = evolve(U0, r_face, r_center, dr, t_final, name, CFL, 
                          output_interval, f"./water_cup_output_weno5_cc_{name}_{N_cells:04d}_{CFL:.3f}", n_ghost)
    

    rho_f, vr_f, vphi_f = C2P(U_final[:, idx_physical], r_physical)
    np.savetxt(f"evolution_final_{name}_{N_cells:04d}_{CFL:.3f}.txt", 
               np.column_stack([r_physical, rho_f, vr_f, vphi_f]),
               header="r, rho, vr, vphi")
    
