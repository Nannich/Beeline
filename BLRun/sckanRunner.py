from pathlib import Path
import shutil
import os
import pandas as pd
import numpy as np

from BLRun.runner import Runner

class SCKANRunner(Runner):
    """Concrete runner for the scKAN GRN inference algorithm."""

    def generateInputs(self):
        """
        Function to generate desired inputs for scKAN.
        """
        # Create a subfolder named after the run ID inside working_dir
        run_env_dir = self.working_dir / self.input_dir.name
        run_env_dir.mkdir(parents=True, exist_ok=True)

        # Copy ExpressionData.csv adn GroundTruthNetwork.csv from the source input directory
        shutil.copy(self.input_dir / self.exprData, run_env_dir / "ExpressionData.csv")

        if self.ground_truth_file.exists():
            shutil.copy(self.ground_truth_file, run_env_dir / "refNetwork.csv")

    def run(self):
        """
        Function to run scKAN algorithm.
        """
        model_arch = self.params.get('model-arch', 'KAN')
        xai_method = self.params.get('xai-method', 'grad')

        uid = os.getuid()
        gid = os.getgid()

        cmd = (
            f"docker run --rm "
            f"--user {uid}:{gid} "
            f"-v /etc/passwd:/etc/passwd:ro "
            f"-v {self.working_dir}:/usr/working_dir "
            f"{self.image} "
            f"--dataset-path /usr/working_dir/{self.input_dir.name} "
            f"--save-path /usr/working_dir/logs/ "
            f"--model-arch {model_arch} "
            f"--xai-method {xai_method}"
        )
        self._run_docker(cmd, append=False)

    def parseOutput(self):
        """
        Function to parse outputs from scKAN.
        """
        # Locate the saved arrays inside logs/[run_id]
        save_path = self.working_dir / "logs" / self.input_dir.name
        gene_list_path = save_path / "gene_list.npy"

        gene_names = np.load(gene_list_path, allow_pickle=True)
        n_gene = len(gene_names)

        # Same standardization as in eval.py from scKAN source
        def standardize(x):
            mean_vals = x.mean(axis=1, keepdims=True)
            std_vals = x.std(axis=1, keepdims=True)
            return ((x - mean_vals) / std_vals > 1).mean(axis=0)

        # Same grn inference as in eval.py from scKAN source
        def build_grn():
            grn = np.zeros((n_gene, n_gene))
            for i in range(n_gene):
                npy_file = save_path / f"gene_{i}_t_all_best.npy"
                raw_weights = np.load(npy_file)
                grn[:, i] = standardize(abs(raw_weights)) * np.sign(raw_weights.mean(axis=0))
            return grn

        grn = build_grn()

        # Flatten the matrix into list of edges
        df_grn = pd.DataFrame(grn.T, index=gene_names, columns=gene_names)
        flat_edges = df_grn.stack().reset_index()
        flat_edges.columns = ['Gene1', 'Gene2', 'EdgeWeight']
        flat_edges = flat_edges.sort_values(by='EdgeWeight', ascending=False)

        self._write_ranked_edges(flat_edges)