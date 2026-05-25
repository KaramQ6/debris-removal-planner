"""Low-Thrust Many-Revolution Orbital Rendezvous Propagator using Gauss's Variational Equations and Lyapunov Q-Law."""

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class LowThrustQPlanner:
    """
    Continuous low-thrust propagator implementing Gauss's Variational Equations (GVE)
    and a Lyapunov-based Q-Law controller to steer Keplerian elements (a, e, i).
    """
    def __init__(self, mu=398600.4418):
        self.mu = mu  # Earth's gravitational parameter (km^3/s^2)
        
    def propagate_transfer(self, x_init, x_target, max_accel=1e-5, dt=10.0, max_time=86400 * 5, tol_a=5.0, tol_e=0.005, tol_i=0.05):
        """
        Propagates many-revolution continuous low-thrust transfer.
        
        x_init, x_target: [a (km), e, i (deg), raan (deg), arg_p (deg), nu (deg)]
        max_accel: maximum thrust acceleration in km/s^2 (e.g. 1e-5 km/s^2 = 10 mm/s^2)
        dt: propagation time step in seconds
        max_time: maximum simulation time in seconds (default 5 days)
        """
        # Convert angles to radians
        a1, e1, i1, raan1, arg1, nu1 = x_init
        a_t, e_t, i_t, raan_t, arg_t, _ = x_target
        
        i1_rad = math.radians(i1)
        i_t_rad = math.radians(i_t)
        raan1_rad = math.radians(raan1)
        raan_t_rad = math.radians(raan_t)
        arg1_rad = math.radians(arg1)
        
        # State: [a, e, i_rad, raan_rad, arg_rad, nu_rad]
        state = np.array([a1, e1, i1_rad, raan1_rad, arg1_rad, math.radians(nu1)], dtype=float)
        
        trajectory = []
        t = 0.0
        total_dv = 0.0
        
        while t < max_time:
            a, e, i, raan, arg_p, nu = state
            
            # Check convergence
            da = a - a_t
            de = e - e_t
            di = math.degrees(i) - i_t
            
            if abs(da) < tol_a and abs(de) < tol_e and abs(di) < tol_i:
                print(f"Rendezvous accomplished at t = {t/3600:.2f} hours!")
                break
                
            # Orbital parameters
            p = a * (1.0 - e**2)
            r = p / (1.0 + e * math.cos(nu))
            h = math.sqrt(self.mu * p)
            if h < 1e-6:
                print("Error: Spacecraft crashed or orbit became singular.")
                break
                
            # Gauss's Planetary Equations (GVE) Matrix (rates of change per unit acceleration)
            # Thrust components: [u_r (radial), u_t (tangential), u_h (out-of-plane)]
            # We use tangential/normal/out-of-plane coordinates
            
            # a rate
            da_dur = 2.0 * a**2 * e * math.sin(nu) / h
            da_dut = 2.0 * a**2 * p / (r * h)
            da_duh = 0.0
            
            # e rate
            de_dur = p * math.sin(nu) / h
            de_dut = ((a + r) * math.cos(nu) + a * e) / h
            de_duh = 0.0
            
            # i rate
            di_dur = 0.0
            di_dut = 0.0
            di_duh = r * math.cos(arg_p + nu) / h
            
            # nu rate (Keplerian motion + perturbation)
            dnu_dt_kepler = h / r**2
            
            # Lyapunov control law (Q-Law feedback)
            # Distances to target
            dist_a = (a - a_t) / 1000.0  # scaled
            dist_e = e - e_t
            dist_i = i - i_t_rad
            
            # Partial derivatives of Lyapunov function V = 0.5 * (dist_a^2 + dist_e^2 + dist_i^2)
            # u vector selected to minimize V dot (make it as negative as possible)
            # V_dot = dist_a * da_dt + dist_e * de_dt + dist_i * di_dt
            # V_dot = u_r * (dist_a * da_dur + dist_e * de_dur) + u_t * (dist_a * da_dut + dist_e * de_dut) + u_h * (dist_i * di_duh)
            
            coeff_ur = dist_a * da_dur + dist_e * de_dur
            coeff_ut = dist_a * da_dut + dist_e * de_dut
            coeff_uh = dist_i * di_duh
            
            # Optimal thrust direction (negative gradient of V)
            u_r = -coeff_ur
            u_t = -coeff_ut
            u_h = -coeff_uh
            
            norm = math.sqrt(u_r**2 + u_t**2 + u_h**2)
            if norm > 1e-9:
                u_r = (u_r / norm) * max_accel
                u_t = (u_t / norm) * max_accel
                u_h = (u_h / norm) * max_accel
            else:
                u_r, u_t, u_h = 0.0, 0.0, 0.0
                
            # Apply GVE derivative
            da_dt = da_dur * u_r + da_dut * u_t + da_duh * u_h
            de_dt = de_dur * u_r + de_dut * u_t + de_duh * u_h
            di_dt = di_dur * u_r + di_dut * u_t + di_duh * u_h
            
            # Update state with Euler integration
            state[0] += da_dt * dt
            state[1] += de_dt * dt
            state[2] += di_dt * dt
            state[5] += dnu_dt_kepler * dt
            
            # Keep angles in bounds
            state[5] = state[5] % (2.0 * math.pi)
            
            # Fuel & Delta-V integration
            dv_step = math.sqrt(u_r**2 + u_t**2 + u_h**2) * dt
            total_dv += dv_step
            
            # Save trajectory logs
            trajectory.append({
                "time": t,
                "a": a,
                "e": e,
                "i": math.degrees(i),
                "nu": math.degrees(nu),
                "u_r": u_r * 1e6,  # mm/s^2
                "u_t": u_t * 1e6,  # mm/s^2
                "u_h": u_h * 1e6,  # mm/s^2
                "total_dv": total_dv * 1000.0,  # m/s
                "altitude": a - 6378.1  # km
            })
            
            t += dt
            
        return trajectory

def plot_trajectory(trajectory, save_path):
    times = [entry["time"] / 3600.0 for entry in trajectory]  # hours
    alts = [entry["altitude"] for entry in trajectory]
    ecc = [entry["e"] for entry in trajectory]
    incs = [entry["i"] for entry in trajectory]
    dvs = [entry["total_dv"] for entry in trajectory]
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    axs[0].plot(times, alts, color="blue", linewidth=2)
    axs[0].set_ylabel("Altitude (km)")
    axs[0].set_title("Low-Thrust Spiral Rendezvous Transfer (Gauss GVE + Q-Law)")
    axs[0].grid(True)
    
    axs[1].plot(times, ecc, color="green", linewidth=2)
    axs[1].set_ylabel("Eccentricity")
    axs[1].grid(True)
    
    axs[2].plot(times, incs, color="purple", linewidth=2)
    axs[2].set_ylabel("Inclination (deg)")
    axs[2].grid(True)
    
    axs[3].plot(times, dvs, color="red", linewidth=2)
    axs[3].set_ylabel("Delta-V Consumed (m/s)")
    axs[3].set_xlabel("Time (hours)")
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Trajectory plot saved successfully to: {save_path}")

if __name__ == "__main__":
    planner = LowThrustQPlanner()
    
    # Parking Orbit LEO: Alt = 600 km, e = 0.001, i = 53.0 deg
    r_earth = 6378.1
    x_init = [r_earth + 600.0, 0.001, 53.0, 100.0, 45.0, 0.0]
    
    # Debris Target Orbit LEO: Alt = 800 km, e = 0.01, i = 54.0 deg
    x_target = [r_earth + 800.0, 0.01, 54.0, 100.0, 45.0, 0.0]
    
    # Continuous thrust acceleration: 2e-5 km/s^2 (20 mm/s^2)
    print("Starting many-revolution low-thrust propagation...")
    traj = planner.propagate_transfer(
        x_init, x_target, 
        max_accel=2e-5, 
        dt=15.0, 
        max_time=86400 * 2,  # 2 days limit
        tol_a=5.0, tol_e=0.002, tol_i=0.02
    )
    
    if traj:
        print(f"Total time of flight: {traj[-1]['time']/3600:.2f} hours")
        print(f"Total Delta-V consumed: {traj[-1]['total_dv']:.1f} m/s")
        
        # Save to assets directory
        assets_dir = Path(r"c:\Users\ASUS\OneDrive\المستندات\GitHub\debris-removal-planner\assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = assets_dir / "low_thrust_q_law_spiral.png"
        plot_trajectory(traj, plot_path)
        
        # Write exact metrics to a json file
        import json
        metrics_path = Path(r"c:\Users\ASUS\OneDrive\المستندات\GitHub\debris-removal-planner\results\low_thrust_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "time_of_flight_hours": traj[-1]["time"]/3600,
            "total_delta_v_m_s": traj[-1]["total_dv"],
            "initial_altitude_km": x_init[0] - r_earth,
            "target_altitude_km": x_target[0] - r_earth,
            "initial_inclination_deg": x_init[2],
            "target_inclination_deg": x_target[2],
            "revolutions_approx": len(traj) * 15.0 / (2.0 * math.pi * math.sqrt(x_init[0]**3 / 398600.4418))
        }
        
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Low-thrust metrics successfully written to: {metrics_path}")
