import pandas as pd

import lotus
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
    ]
}
df = pd.DataFrame(data)

# Basic usage - single sample (default)
user_instruction = "What is a similar course to {Course Name}. Be concise."
df1 = df.sem_map(user_instruction)
print("Single sample output:")
print(df1)
print("\n")

# Multiple samples with temperature to generate diverse responses
df2 = df.sem_map(
    user_instruction,
    nsample=3,  # Generate 3 responses per input
    temperature=0.8,  # Higher temperature for more diverse outputs
)
print("Multiple samples output:")
print(df2) 