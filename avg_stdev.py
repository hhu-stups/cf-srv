import pandas as pd
import os

dirs = [
    'Shielded_Timed_Traces/',
    'Unshielded_Timed_Traces/',
]


for d in dirs:
    csv_cov = os.path.join(d, 'SimulationStatistics_Coverage.csv')
    csv_rwrd = os.path.join(d, 'SimulationStatistics_Reward.csv')

    df_cov = pd.read_csv(csv_cov)
    df_rwrd = pd.read_csv(csv_rwrd)

    trace_lengths = df_cov['Trace Length'] - 2
    coverages = df_cov['Estimated Value']
    rewards = df_rwrd['Estimated Value']

    print('#', d)
    print(f"Episode Length: {trace_lengths.mean():.2f} +- {trace_lengths.std():.2f}")
    print(f"Coverage: {coverages.mean():.4f} +- {coverages.std():.4f}")
    print(f"Reward: {rewards.mean():.2f} +- {rewards.std():.2f}")
    print()
