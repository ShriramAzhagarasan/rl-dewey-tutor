import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/monitor.csv")
df["r_ma"] = df["r"].rolling(50).mean()
df[["l","r_ma"]].plot(x="l", y="r_ma")
plt.xlabel("timesteps"); plt.ylabel("episode reward (50-ep MA)")
plt.title("PPO on TutorEnv"); plt.tight_layout()
plt.savefig("reports/figures/ppo_curve.png")
print("Saved reports/figures/ppo_curve.png")
