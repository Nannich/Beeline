import os
import pandas as pd
import numpy as np
import lagkan
import time

from BLRun.runner import Runner

class LagKANRunner(Runner):
    """Concrete runner for the lagKAN GRN inference algorithm."""

    def generateInputs(self):
        pass

    def run(self):
        """
        Function to run lagKAN algorithm.
        """
        
        # Read expression and pseudotime matrix
        expression_df = pd.read_csv(self.input_dir / self.exprData, header=0, index_col=0)
        pt_df = pd.read_csv(self.input_dir / self.pseudoTimeData, header=0, index_col=0)
        
        # Make sure cells in the pseudotime and expression matrix match
        pt_df = pt_df.reindex(expression_df.columns)
        
        # Generate matrix inputs for lagKAN
        raw_counts = expression_df.values.T             # Shape: (n_cells, n_genes)

        # log1p normalize counts
        log_counts = np.log1p(raw_counts)

        # Replace na with 0
        pseudotime = pt_df.fillna(0.0).values           # Shape: (n_cells, n_lineages)

        # Replace na with false, not na with true
        lineage_assignment = pt_df.notna().values       # Shape: (n_cells, n_lineages)
        
        gene_names = expression_df.index.values

        # Extract parameters
        epochs = int(self.params.get('epochs', 400))
        lr = float(self.params.get('lr', 0.01))
        lamb_l1 = float(self.params.get('lamb_l1', 0.02))

        # Execute the network inference
        ranked_edges_df = lagkan.infer_grn(
            log_counts=log_counts,
            pseudotime=pseudotime,
            lineage_assignment=lineage_assignment,
            gene_names=gene_names,
            epochs=epochs,
            lr=lr,
            lamb_l1=lamb_l1
        )

        # Save the raw output table inside self.working_dir
        ranked_edges_df.to_csv(self.working_dir / "raw_predictions.tsv", sep="\t", index=False)

    def parseOutput(self):
        """
        Function to parse outputs from lagKAN.
        """
        raw_output_file = self.working_dir / "raw_predictions.tsv"
        if not raw_output_file.exists():
            print(f"{raw_output_file} does not exist, skipping ...")
            return

        # Read the raw predictions generated during the run() call
        out_df = pd.read_csv(raw_output_file, sep="\t", header=0)
        
        # Format explicitly into the exact 3 columns expected by the evaluator
        final_df = out_df[["Gene1", "Gene2", "EdgeWeight"]]
        
        self._write_ranked_edges(final_df)