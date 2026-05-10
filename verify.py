"""Quick verification script — tests all components."""
import sys
import traceback

def test_environment():
    print("=" * 60)
    print("TEST 1: Environment Creation & Reset")
    print("=" * 60)
    from simulation.orbit_env import OrbitDebrisEnv
    env = OrbitDebrisEnv()
    obs, info = env.reset()
    print(f"  OK - obs shape: {obs.shape}")
    print(f"  Info: cleared={info['cleared']}, targets={info['total_targets']}, fuel={info['fuel_remaining']:.1f}")
    
    # Test a step
    action = env.valid_actions()[0]
    obs, reward, term, trunc, info = env.step(int(action))
    print(f"  Step OK - reward={reward:.2f}, cleared={info['cleared']}")
    print()

def test_policies():
    print("=" * 60)
    print("TEST 2: Baseline Policies")
    print("=" * 60)
    import numpy as np
    from simulation.orbit_env import OrbitDebrisEnv
    from simulation.policies import random_policy, nearest_neighbor_policy, risk_weighted_policy
    
    env = OrbitDebrisEnv(seed=42)
    env.reset(seed=42)
    rng = np.random.default_rng(42)
    
    # Test each policy
    for name, fn in [("random", lambda: random_policy(env, rng)), 
                     ("nearest", lambda: nearest_neighbor_policy(env)),
                     ("risk_weighted", lambda: risk_weighted_policy(env))]:
        env.reset(seed=42)
        done = False
        info = {}
        while not done:
            action = fn()
            _, _, term, trunc, info = env.step(int(action))
            done = term or trunc
        print(f"  {name}: cleared={info['cleared']}/{info['total_targets']}, dv={info['total_delta_v']:.1f}")
    print()

def test_scenario_presets():
    print("=" * 60)
    print("TEST 3: Scenario Presets")
    print("=" * 60)
    from simulation.scenario import easy_scenario, medium_scenario, hard_scenario
    for name, fn in [("easy", easy_scenario), ("medium", medium_scenario), ("hard", hard_scenario)]:
        s = fn(seed=42)
        print(f"  {name}: {len(s.targets)} targets, fuel={s.fuel_budget}, max_steps={s.max_steps}")
    print()

def test_rag():
    print("=" * 60)
    print("TEST 4: RAG Advisory System")
    print("=" * 60)
    from pathlib import Path
    from rag.rag_system import SimpleRAGAdvisor
    
    advisor = SimpleRAGAdvisor()
    advisor.index_directory(Path("docs"))
    print(f"  Indexed {advisor.chunk_count} chunks")
    
    result = advisor.answer("What should we do when delta-v budget is low?")
    answer_lines = result["answer"].split("\n")
    print(f"  Query answered, {len(result['sources'])} sources found")
    print(f"  First line: {answer_lines[0]}")
    print()

def test_visualization_imports():
    print("=" * 60)
    print("TEST 5: Visualization Imports")
    print("=" * 60)
    import matplotlib
    print(f"  matplotlib: {matplotlib.__version__}")
    try:
        import plotly
        print(f"  plotly: {plotly.__version__}")
    except ImportError:
        print("  plotly: not installed (optional)")
    print()

if __name__ == "__main__":
    tests = [
        test_environment,
        test_policies,
        test_scenario_presets,
        test_rag,
        test_visualization_imports,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
