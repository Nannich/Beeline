# Scope
This guide outlines the steps required to reproduce the BEELINE benchmark results.

## 1. Environment Setup

```bash
git clone [https://github.com/Nannich/BEELINE.git](https://github.com/Nannich/BEELINE.git)
cd BEELINE
```

## 2. Data Acquisition
1. Download the BEELINE datasets and Networks from [Zenodo (Record 7682713)](https://zenodo.org/records/7682713).
2. Extract the downloaded Networks, scRNA-seq, Curated and Synthetic dataset folders directly into the `inputs/` directory.

The experimental scRNA-seq datasets were reduced to transcription factors + 500 genes using `utils/generateExpInputs.py`.

## 3. Figure and Table Mapping
For Running the Algorithms, use:
```bash
python BLRunner.py -c <path_to_config_file>
```

For evaluating the resulting rankedEdges.csv files, use:
```bash
python BLEvaluator.py -c <path_to_config_file> -a -e -t -r -s
```

For plotting the results, use:
```bash
python BLPlotter.py -c <path_to_config_file> -o ./plots --all
```

| Thesis Reference | Description | config-files/ |
| :--- | :--- | :--- |
| **Figure 4** | Hyperparameter Sweeps | `hyper.yaml` |
| **Figure 5** | Time Delay Sweep | `dt.yaml` |
| **Figure 6** | Benchmarking lagKAN | `all.yaml` |
| **Figure 7** | EPR comparison on scRNA-seq datasets (only lagKAN values) | `scRNAseq.yaml` |
```