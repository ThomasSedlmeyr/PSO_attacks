# PSO_attacks

## Overview
The `PSO_attacks` project utilizes state-of-the-art data synthesis frameworks to generate high-quality synthetic data. This process is essential for developing robust machine learning models without compromising real data privacy.

## Abstract
Singling out is one of the concrete privacy violations acknowledged by the General DataProtection Regulation (GDPR) and plays a significant role in discussions about data protection risks. Recent research has introduced the concept of predicate singling out (PSO), where an adversary identifies a unique record in a dataset by exploiting the output of a data-release mechanism. This occurs when the adversary finds a specific condition, or predicate, that matches exactly one data sample with a probability significantly higher than a statistical baseline. A widely used approach to minimize and control the exposure of private information when publishing datasets is to release synthetic data using differential privacy, which is a formal technique to provide provable privacy protection.

In the first part of our study, we conduct an empirical analysis to assess the success rate of singling-out attacks on synthetic datasets, with a focus on how the chosen privacy budget impacts the effectiveness of these attacks. In the second part, we demonstrate that trained decision trees are especially susceptible to singling-out attacks. We introduce an attack method that identifies vulnerable nodes and leverages the corresponding decision paths to construct predicates capable of isolating a specific sample in the dataset. To mitigate this vulnerability, we also introduce an algorithm that prunes an already trained decision tree, thereby enhancing the robustness against singling-out attacks. The source code needed to reproduce our experiments can be found here: https://github.com/TUM-AIMED/PSO_attacks.

## Frameworks Used
We employ the following frameworks to generate synthetic data:

- **DPART**: [Learn more](https://github.com/hazy/dpart).
- **Synthetic Data Generation**: [Learn more](https://github.com/daanknoors/synthetic_data_generation).
- **DataSynthesizer**:   [Learn more](https://github.com/DataResponsibly/DataSynthesizer).


## Getting Started

Step 1: Create the Conda Environment
```bash
conda env create -f environment.yml
```

Step 2: Activate the Environment
```bash
conda activate pso_attacks
```

Step 3: Install dpart via pip
```bash
pip install dpart
```


