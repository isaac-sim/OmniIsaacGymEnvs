import matplotlib.pyplot as plt
import pandas as pd

# basic params
experiment = "Biped_07"
file = f"{experiment}.txt"
col_names = ["Ave Reward", "Progress Reward", "Alive Reward",
              "Up Reward", "Heading Reward", "Action Cost",
              "Energy Cost", "DOF Limit Cost"]

# parse log and make pd df
df = []
with open(file, 'r') as f:
    row = []
    for line in f:
        split = line.split(":")
        rew_type, val = split[0], split[1].strip()
        
        if rew_type[:4] != "Step":
            row.append(float(val))
        else:
            df.append(row)
            row = []

df = pd.DataFrame(df[1:][:])
df.columns = col_names
df = df.reset_index()

# plot
# Create a bar chart for stacked categories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
# fig, ax1 = plt.subplots(figsize=(10, 6))
stacked_cols = col_names[1:]
stacked_cols.append('index')
line_cols = ['Ave Reward', 'index']

df[stacked_cols].plot(x='index', kind='bar', stacked=True, ax=ax1)
ax1.set_xlabel('Episode Num * 100')
ax1.tick_params(axis='x', labelsize=5)
ax1.set_ylabel('Reward Components')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Create a line plot for 'Ave Reward'
df[line_cols].plot(x='index', kind='line', color='black', marker='o', ax=ax2)
ax2.set_xlabel('Episode Num * 100')
ax2.set_ylabel('Ave Reward')

plt.tight_layout()
plt.savefig(f"{experiment}.png")
plt.show()

