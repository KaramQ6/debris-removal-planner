import json, numpy as np

for fname in ['test_evaluation_summary.json', 'unmasked_ppo_evaluation_summary.json', 'test_new_metrics.json']:
    print(f"\n=== {fname} ===")
    try:
        d = json.load(open(f'results/{fname}'))
        for k in d:
            if isinstance(d[k], dict) and 'avg_cleared' in d[k]:
                s = d[k]
                cleared_arr = np.array(s.get('all_cleared', []))
                std_c = np.std(cleared_arr) if len(cleared_arr) > 0 else 'N/A'
                print(f"  {k}: cleared={s['avg_cleared']:.2f} +/- {std_c:.2f}, "
                      f"dv={s['avg_delta_v']:.1f} +/- {s.get('std_delta_v',0):.1f}, "
                      f"fpt={s['fuel_per_target']:.1f}")
            elif k == 'evaluation_parameters':
                print(f"  params: {d[k]}")
    except Exception as e:
        print(f"  Error: {e}")
