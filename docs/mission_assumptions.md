# Mission Assumptions (Phase 1 Starter)

1. **Orbital geometry**: the environment is represented as a 2D circular plane with angular positions.
2. **Transfer cost**: each transfer uses a simplified delta-v estimate derived from angular separation.
3. **Single-capture actions**: each action targets exactly one debris object.
4. **Fuel model**: mission fuel is represented as an aggregate delta-v budget (m/s).
5. **Termination**: an episode ends when all targets are cleared, fuel is depleted, or max steps is reached.
6. **Risk weighting**: higher-risk objects receive stronger reward bonus to encourage earlier removal.

