sem_map
=================

Overview
----------
This operato performs a semantic projection over an input column. The langex parameter specifies this projection in natural language.

Motivation
-----------
The sem_map operator is useful for performing a row-wise operations over the data.

Example
----------
.. code-block:: python

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
    user_instruction = "What is a similar course to {Course Name}. Be concise."
    df = df.sem_map(user_instruction)
    print(df)

Output:

+---+----------------------------------------+----------------------------------------------------------------+
|   | Course Name                            | _map                                                           |
+===+========================================+================================================================+
| 0 | Probability and Random Processes       | A similar course to "Probability and Random Processes"...      |
+---+----------------------------------------+----------------------------------------------------------------+
| 1 | Optimization Methods in Engineering    | A similar course to "Optimization Methods in Engineering"...   |
+---+----------------------------------------+----------------------------------------------------------------+
| 2 | Digital Design and Integrated Circuits | A similar course to "Digital Design and Integrated Circuits"...|
+---+----------------------------------------+----------------------------------------------------------------+
| 3 | Computer Security                      | A similar course to "Computer Security" is "Cybersecurity"...  |
+---+----------------------------------------+----------------------------------------------------------------+

Multiple Samples Example
------------------------
.. code-block:: python

    # Generate multiple samples per row with increased temperature
    df = df.sem_map("What is a similar course to {Course Name}. Be concise.", 
                     nsample=3, 
                     temperature=0.8)
    print(df)

Output with multiple samples:

+---+----------------------------------------+------------------------------------+-------------------------------------+-------------------------------------+----------------------------------+
|   | Course Name                            | _map_1                             | _map_2                              | _map_3                              | _map_all                         |
+===+========================================+====================================+=====================================+=====================================+==================================+
| 0 | Probability and Random Processes       | Statistics and Data Analysis       | Stochastic Processes                | Statistical Inference               | [Statistics and Data Analysis... |
+---+----------------------------------------+------------------------------------+-------------------------------------+-------------------------------------+----------------------------------+
| 1 | Optimization Methods in Engineering    | Numerical Methods                  | Operations Research                 | Linear and Nonlinear Programming    | [Numerical Methods, Operations.. |
+---+----------------------------------------+------------------------------------+-------------------------------------+-------------------------------------+----------------------------------+
| 2 | Digital Design and Integrated Circuits | Computer Architecture              | VLSI Design                         | Embedded Systems Design             | [Computer Architecture, VLSI...  |
+---+----------------------------------------+------------------------------------+-------------------------------------+-------------------------------------+----------------------------------+
| 3 | Computer Security                      | Cybersecurity                      | Network Security                    | Cryptography                        | [Cybersecurity, Network Secur... |
+---+----------------------------------------+------------------------------------+-------------------------------------+-------------------------------------+----------------------------------+

Required Parameters
---------------------
- **user_instruction** : The user instruction for map.
- **postprocessor** : The postprocessor for the model outputs. Defaults to map_postprocess.

Optional Parameters
---------------------
- **return_explanations** : Whether to return explanations. Defaults to False.
- **return_raw_outputs** : Whether to return raw outputs. Defaults to False.
- **suffix** : The suffix for the new columns. Defaults to "_map".
- **examples** : The examples dataframe. Defaults to None.
- **strategy** : The reasoning strategy. Defaults to None.
- **nsample** : Number of samples to generate per document. When greater than 1, creates multiple columns with numbered suffixes. Defaults to 1.
- **temperature** : Temperature for sampling. Higher values produce more varied outputs. Only effective when nsample > 1. Defaults to None, which uses the model's default temperature.
