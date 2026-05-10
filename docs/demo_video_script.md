# Demo Video Script — 3 Minutes

**Team**: ta5abes
**Project**: Intelligent Orbital Debris Removal Planner
**Duration**: 3 minutes
**Format**: Screen recording with voiceover narration

---

## 0:00-0:20 | Hook

**Visual**: Open with the 3D orbital visualization showing debris scattered around Earth.

**Narration**:
> "36,000 pieces of debris orbit Earth right now.
> One collision creates thousands more. We built an AI that plans the optimal cleanup mission."

**Action**: Rotate the 3D view to show the scale of the problem.

---

## 0:20-0:50 | Architecture

**Visual**: Show the architecture diagram from `docs/architecture_diagram.md`

**Narration**:
> "Our system has four components: a simulation engine that models orbital mechanics, three baseline planning strategies for comparison, a Reinforcement Learning agent that learns fuel-optimal debris collection sequences, and a RAG advisory system for real-time operational guidance."

**Action**: Highlight each component as you mention it:
- Environment → 4 policies → Evaluation → RAG

---

## 0:50-1:30 | Live Evaluation

**Visual**: Terminal running evaluation

**Action**: Run the command:
```powershell
python -m simulation.evaluate --episodes 100 --model-path results\models\ppo_debris.zip
```

**Narration**:
> "Let's compare four planning strategies across 100 simulated missions. Each mission has 8 debris targets and a 1,200 meter-per-second fuel budget."

*Wait for output, then highlight the numbers:*

> "The RL agent achieves 683.2 m/s average delta-V — that's 35.9% less fuel than random planning, and successfully clears 100% of the debris targets."

**Visual**: Show the terminal comparison table with results.

---

## 1:30-2:00 | 3D Visualization — THE WOW MOMENT

**Visual**: Open `results/mission_interactive_nearest.html` in browser

**Narration**:
> "This is the mission trajectory visualization. Each line shows a transfer orbit between debris targets."

**Action**: Rotate the 3D view, zoom in on the trajectory.

> "This is the PPO agent's learned policy — you can see it avoids wasted maneuvers and plans globally optimal paths, not just going to the nearest target."

**Visual**: Show side-by-side polar plots for `nearest` vs `ppo` policy.

---

## 2:00-2:30 | RAG Demo

**Visual**: Terminal running RAG query

**Action**: Run:
```powershell
python -m rag.rag_system --docs docs --query "What is the recommended approach for high-inclination debris?"
```

**Narration**:
> "The RAG advisory system indexes NASA, ESA, and IADC debris mitigation guidelines. Mission operators can query it for real-time operational guidance — just like a human team would consult reference documents."

**Visual**: Show the system returning relevant passage with source attribution.

---

## 2:30-3:00 | Results + Close

**Visual**: Show `results/delta_v_comparison.png` chart

**Narration**:
> "Our RL agent achieves nearly 36% fuel reduction over the random baseline — making debris removal missions more economically viable. At $10,000 per kilogram of fuel, every percentage point matters."

> "This architecture — autonomous RL planning plus knowledge-grounded advisory — represents a scalable model for sustainable space operations. Thank you."

**Visual**: Show team name and hackathon badge.

---

## Recording Checklist

- [ ] Record terminal at 1920x1080, dark theme
- [ ] Pre-run all commands to ensure clean output
- [ ] Have all chart images open for quick switching
- [ ] Record voiceover separately for clean audio
- [ ] Export as MP4, H.264, 1080p
- [ ] Keep under 3 minutes
- [ ] Add team logo/name as intro slide
- [x] Updated placeholders with actual results after training
