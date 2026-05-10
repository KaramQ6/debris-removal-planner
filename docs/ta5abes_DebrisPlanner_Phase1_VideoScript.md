# Demo Video Script
**Target Length:** 2.5 to 3 minutes
**Tone:** Professional, technical, and forward-looking.

---

### Segment 1: The Problem (0:00 - 0:30)
**Visuals:**
- Screen recording showing the exponential growth of orbital debris (graphs or NASA/ESA orbital visualizations).
- Transition to a slide showing manual mission planning inefficiencies.

**Voiceover:**
"Earth's orbital environment is at a critical tipping point. With over a million dangerous debris fragments currently in orbit, the risk of a catastrophic Kessler Syndrome cascade grows every day. The biggest bottleneck in cleaning up our orbits isn't just capturing the debris—it's figuring out how to do it efficiently. Current manual planning for multi-target debris removal is slow, fuel-wasteful, and simply won't scale. That is why Team ta5abes developed the Intelligent Orbital Debris Removal Planner."

### Segment 2: The Solution Architecture (0:30 - 1:00)
**Visuals:**
- Display the dual-AI system architecture diagram.
- Highlight the "RL Core Agent" and "RAG Advisory System" boxes.

**Voiceover:**
"We designed a dual-AI mission planning system. First, a Reinforcement Learning core agent utilizes Proximal Policy Optimization to autonomously calculate fuel-optimal trajectories between multiple debris targets. Second, a Retrieval-Augmented Generation—or RAG—advisory layer provides real-time operational guidance indexed directly from NASA, ESA, and IADC safety guidelines. This ensures our AI doesn't just find the fastest route, but the safest and most compliant one."

### Segment 3: Lifecycle Logic & End-of-Life (1:00 - 1:30)
**Visuals:**
- Graphic showing the spacecraft lifecycle (Launch → Active Collection → Safe Disposal).
- Briefly show the RAG system interface querying a deorbiting protocol.

**Voiceover:**
"True sustainability requires lifecycle thinking. By drastically reducing the Delta-V consumed during the active mission phase, our spacecraft extends its operational life to clear more debris per launch. Just as importantly, our RAG system enforces strict end-of-life protocols, ensuring the removal spacecraft retains exact fuel reserves to safely deorbit itself when the mission concludes—leaving zero new debris behind."

### Segment 4: Simulation & Results (1:30 - 2:30)
**Visuals:**
- Screen recording of the terminal running `python -m simulation.evaluate` and `python -m simulation.visualize`.
- Show the side-by-side 3D/2D visualization of the Random baseline path versus the RL optimized path.
- Show the bar charts comparing Delta-V consumption.

**Voiceover:**
"To validate our concept, we built a custom Gymnasium orbital environment. Over a standardized scenario of eight targets in Low Earth Orbit, our RL agent achieved a 100% clearance rate. More importantly, it achieved a 35.9% reduction in fuel consumption compared to baseline random sequencing. As you can see in these 3D visualizations, the RL agent learns to group targets and minimize angular transfer costs, vastly outperforming manual approximations. Simultaneously, our RAG system can instantly retrieve critical mitigation protocols when queried by operators."

### Segment 5: Conclusion & Impact (2:30 - 3:00)
**Visuals:**
- Final impact summary slide.
- Team name and Hackathon logo.

**Voiceover:**
"Every percentage of fuel saved means lower launch mass, reduced carbon footprint, and more debris cleared from our skies. By combining cutting-edge path optimization with strict space agency safety protocols, the Intelligent Orbital Debris Removal Planner represents a scalable, autonomous solution for the long-term sustainability of space. Thank you from Team ta5abes."
