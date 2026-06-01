import os
import zipfile
import shutil

def package_research():
    print("Starting packaging process...")
    repo_root = r"c:\Users\ASUS\OneDrive\المستندات\GitHub\debris-removal-planner"
    package_dir = os.path.join(repo_root, "Space_Debris_Planner_Research_Package")
    zip_path = os.path.join(repo_root, "Space_Debris_Planner_Research_Package.zip")
    
    # 1. Create directory if not exists
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir, exist_ok=True)
    
    # 2. Write the Core Idea and Methodology Summary document
    summary_content = """# Space Debris Planner: Core Idea & Scientific Methodology Summary

**Project Title:** Autonomous Propellant-Constrained Multi-Target Space Debris Removal Sequence Planning via Action-Masked Deep Reinforcement Learning and Decoupled RAG Operational Advisory

---

## 1. The Core Idea (الفكرة الأساسية للبحث)
The exponential accumulation of space debris in Low Earth Orbit (LEO) presents an escalating threat of collision cascade (the Kessler Syndrome). Active Debris Removal (ADR) chaser satellites must rendezvous with, capture, and de-orbit multiple debris objects. However, spacecraft are strictly constrained by their onboard propellant (Delta-V budget). Target sequencing (which debris to intercept next) is a highly complex, hybrid combinatorial-continuous 3D astrodynamics optimization problem.

**Our Core Concept:**
We propose a **hybrid dual-layer autonomous system** integrating:
1. **A Deep Reinforcement Learning (RL) Planning Core:** An action-masked Proximal Policy Optimization (MaskablePPO) agent that dynamically learns real-time, fuel-efficient target intercept sequences within physical propellant constraints in a 3D Keplerian surrogate simulator.
2. **A Decoupled RAG Safety Advisory Layer:** A local, guideline-grounded Retrieval-Augmented Generation (RAG) system utilizing Okapi BM25 and cosine tie-breaker algorithms to retrieve governing safety clauses (NASA-STD-8719.14 and IADC guidelines) for mission operator decision support.

---

## 2. Methodology (المنهجية العلمية والتطبيقية)

### A. 3D Keplerian Surrogate Simulator (ميكانيكا المدارات ثنائية وثلاثية الأبعاد)
We model a 3D orbital propagation space where spacecraft states are defined by classical Keplerian elements: $[a, e, i, \Omega, \omega, \nu]^T$ (Semi-major axis, eccentricity, inclination, RAAN, argument of periapsis, true anomaly).
To approximate orbital transfer costs, we implement analytical Delta-V impulse formulations for:
1. **Orbital Size Changes ($\Delta V_{size}$):** Circular-equivalent Hohmann transfers modifying mechanical orbital energy.
2. **Eccentricity Shape Changes ($\Delta V_{ecc}$):** Modifying orbital eccentricity.
3. **Orbital Plane Alignments ($\Delta V_{plane}$):** Spherical law of cosines to calculate non-coplanar plane rotation angles, executed at orbital apoapsis (the highest and slowest point) to maximize fuel efficiency.
4. **Apsidal Rotations ($\Delta V_{apsidal}$):** Applying orthogonal steering to rotate the orbital line of apsides.

### B. Action-Masked Reinforcement Learning (التعلم المعزز مع قناع العمل الفعال)
We formulate sequencing as a Markov Decision Process (MDP) with a 155-dimensional state observation space. Standard PPO agents suffer from policy divergence and sparse reward deadlocks in orbital planning because they frequently attempt invalid transfers (already intercepted targets or targets exceeding the propellant budget).
* **The Solution:** We implement **Action Masking**. Dynamically, at each step, we calculate the remaining fuel $f_{remaining}$ and mask out targets where $\Delta V_{target, i} > f_{remaining}$. By setting the raw policy network logits of invalid actions to $-\infty$, the dynamic softmax probability of choosing these actions drops to exactly zero ($e^{-\infty} = 0$).
* **Reward Structure ($R$):** Balances target interception bonuses ($+25.0$), risk weighting ($+30.0 \cdot r_i$ based on NASA's LEGEND explosion risk guidelines), fuel expenditure penalties ($-0.001 \cdot \Delta V$), and terminal completion rewards.

### C. Guideline-Grounded RAG Safety Advisor (نظام الاسترجاع المعزز بالوثائق)
To ensure compliance with international treaties, a local, decoupled BM25-based RAG layer ingests and chunks NASA and IADC safety guidelines. When queried, it retrieves precise safety clauses in real-time ($4.2\text{ ms}$ average latency) to prevent human operator violations.
* **Retriever Performance:** Evaluated on a test suite of 30 standard-specific operational queries, achieving **Precision@1 of 86.7%**, **Precision@3 of 93.3%**, and a **Mean Reciprocal Rank (MRR) of 0.90**.

### D. Continuous Q-Law Low-Thrust Feasibility (الدفع المستمر منخفض الثقة)
To confirm physical realization on continuous-thrust electric propulsion (like Hall effect thrusters), we developed a continuous steering controller based on **Gauss's Variational Equations (GVE)** and **Lyapunov steering (Q-law)**, successfully transferring a chaser across many-revolution spirals over 29.8 orbits, indicating feasibility that discrete sequencing outputs map to continuous engines.

---

## 3. Scientific References (المراجع العلمية المعتمدة)
The research is fully grounded in the following academic references:

1. **[Debris Risk & Kessler Syndrome]** D. J. Kessler and B. G. Cour-Palais, "Collision frequency of artificial satellites: The creation of a debris belt," *Journal of Geophysical Research*, vol. 83, no. A6, pp. 2637–2646, 1978.
2. **[Mission Concepts]** J. L. Forshaw et al., "RemoveDEBRIS: An in-orbit active debris removal demonstration mission," *Acta Astronautica*, vol. 127, pp. 448–463, 2016.
3. **[Genetic Optimization]** G. He and R. G. Melton, "Multiple small-satellite salvage mission sequence planning for debris mitigation," in *AAS/AIAA Astrodynamics Specialist Conference*, 2019.
4. **[Deep Reinforcement Learning (PPO)]** J. Schulman et al., "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
5. **[Action Masking Layer]** S. Huang and S. Ontañón, "A closer look at action masking in deep reinforcement learning," *arXiv preprint arXiv:2006.14171*, 2020.
6. **[RL in Astrodynamics]** J. Yang et al., "A Reinforcement Learning Scheme for Active Multi-Debris Removal Mission Planning," *IEEE Access*, vol. 8, pp. 34362–34372, 2020.
7. **[Low-Thrust Lyapunov steering (Q-law)]** A. E. Petropoulos, "Low-thrust orbit transfers using candidate Lyapunov functions," in *AIAA/AAS Astrodynamics Specialist Conference*, 2004.
8. **[NASA Debris Standard]** NASA, "Process for limiting orbital debris," *NASA Standard NASA-STD-8719.14B*, 2021.
9. **[IADC Guidelines]** Inter-Agency Space Debris Coordination Committee (IADC), "IADC space debris mitigation guidelines," *Report IADC-02-01*, Rev. 3, 2021.
10. **[Explosion Risk Guidelines]** J.-C. Liou et al., "LEGEND: A three-dimensional LEO-to-GEO debris evolutionary model," *Advances in Space Research*, vol. 34, no. 5, pp. 981–986, 2004.
"""
    with open(os.path.join(package_dir, "1_Core_Idea_and_Methodology_Summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_content)
    
    # 3. Copy files to package directory
    docs_dir = os.path.join(repo_root, "docs")
    shutil.copy(os.path.join(docs_dir, "AEECT_2026_Paper_Draft.md"), os.path.join(package_dir, "2_Full_Academic_Paper_Draft.md"))
    shutil.copy(os.path.join(docs_dir, "AEECT_2026_Paper.tex"), os.path.join(package_dir, "3_IEEE_LaTeX_Source.tex"))
    shutil.copy(os.path.join(docs_dir, "Low_Thrust_Debris_Removal_Research.md"), os.path.join(package_dir, "4_Low_Thrust_Continuous_Feasibility_Analysis.md"))
    
    # Copy TLE presets if they exist in directories
    results_dir = os.path.join(repo_root, "results")
    if os.path.exists(results_dir):
        package_results_dir = os.path.join(package_dir, "5_Evaluation_Results_Data")
        os.makedirs(package_results_dir, exist_ok=True)
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                shutil.copy(os.path.join(results_dir, filename), os.path.join(package_results_dir, filename))
                
    # 4. Create ZIP archive
    print(f"Creating ZIP archive at: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_root)
                zipf.write(file_path, rel_path)
                
    print("Packaging completed successfully!")
    print(f"ZIP file size: {os.path.getsize(zip_path)} bytes")

if __name__ == "__main__":
    package_research()
