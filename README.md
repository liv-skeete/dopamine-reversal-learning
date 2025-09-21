# Dopamine-Driven Addiction in Reversal Learning (RL Agents)

**Hybrid V→Q agents implementing prediction error, hedonic, and incentive salience theories; pre/post-cocaine reversal learning analysis.**

Work by Olivia Skeete, Suri Le, and Chana Fink.

This repository models how addiction can hijack dopamine-driven reinforcement learning. We implement three dopamine-theory agents—Prediction Error (learning-rate modulation), Hedonic (reward scaling), and Incentive Salience (cue-action bias)—in a hybrid V→Q framework with meta-policy switching. Agents are evaluated in a monkey-inspired reversal learning task across pre- and post-cocaine conditions, with analyses of acquisition, reversal, and perseverative errors and publication-ready visualizations.

## Overview

This project implements computational agents that model different aspects of dopamine function in reinforcement learning tasks. The framework includes:

- **Prediction Error Agents**: Model temporal difference learning and prediction errors
- **Hedonic Agents**: Model reward sensitivity and pleasure responses  
- **Incentive Salience Agents**: Model motivational salience and cue reactivity

Each agent type is tested in acquisition-reversal learning paradigms to study behavioral differences between normal and addicted populations.

## Project Structure

```
dopamine-research/
├── src/                    # Source code
│   ├── agents/            # Agent implementations
│   │   ├── prediction_error_agent.py
│   │   ├── hedonic_agent.py
│   │   └── incentive_salience_agent.py
│   ├── experiments/       # Experiment framework
│   ├── analysis/         # Data analysis tools
│   └── utils/            # Utility functions
├── scripts/              # Main execution scripts
│   ├── run_prediction_error_experiment.py
│   ├── run_hedonic_experiment.py
│   └── run_incentive_salience_experiment.py
├── notebooks/            # Jupyter notebooks for exploration
├── data/                 # Data storage
│   ├── raw/             # Raw experimental data
│   ├── processed/       # Processed data
│   └── figures/         # Generated figures
├── config/              # Configuration files
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/dopamine-research.git
cd dopamine-research
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn

## Usage

### Running Experiments

Run individual experiments:

```bash
# Prediction Error experiment
python scripts/run_prediction_error_experiment.py

# Hedonic experiment  
python scripts/run_hedonic_experiment.py

# Incentive Salience experiment
python scripts/run_incentive_salience_experiment.py
```

### Using the Agents

```python
from src.agents.prediction_error_agent import PredictionErrorAgent
from src.agents.hedonic_agent import HedonicAgent
from src.agents.incentive_salience_agent import IncentiveSalienceAgent

# Create agents
pe_agent = PredictionErrorAgent("Normal", bias=0.0)
hedonic_agent = HedonicAgent("Addicted", bias=1.0)
salience_agent = IncentiveSalienceAgent("Addicted", bias=0.8)
```

## Experimental Design

Each experiment follows the same structure:

1. **Acquisition Phase**: 10 trials to learn initial reward contingencies
2. **Reversal Phase**: 20 trials with reversed reward contingencies
3. **Error Metrics**:
   - Acquisition errors
   - Reversal errors  
   - Perseverative errors

## Results

Results are saved in the `results/` directory including:
- CSV files with error metrics
- PNG files with visualization plots
- Meta-strategy usage plots
- Off-policy accuracy comparisons

## Theoretical Background

### Prediction Error Theory
Models dopamine as signaling prediction errors - the difference between expected and actual rewards. Addicted individuals show enhanced prediction error signaling.

### Hedonic Theory  
Models dopamine as mediating pleasure responses. Addicted individuals show enhanced reward sensitivity and reduced punishment sensitivity.

### Incentive Salience Theory
Models dopamine as assigning motivational value to cues. Addicted individuals show enhanced salience attribution to drug-related cues.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dopamine_research_2025,
  title = {Dopamine Research: Computational Models of Addiction},
  author = {Liv Skeete},
  year = {2025},
  url = {https://github.com/liv-skeete/dopamine-research}
}
```

## Contact

For questions or collaborations, please contact:
- Email: liv@di.st
- GitHub: @liv-skeete
