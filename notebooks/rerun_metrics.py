import os
import subprocess


def main():
    log_root = "lightning_logs"
    if not os.path.exists(log_root):
        print(f"Error: {log_root} does not exist.")
        return

    evaluation_dirs = [
        d
        for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d)) and d.endswith("_evaluation")
    ]

    for eval_dir in evaluation_dirs:
        orig_dir = eval_dir.replace("_evaluation", "")
        llm_outputs_fake = os.path.join(log_root, orig_dir, "llm_outputs.csv")

        cmd = [
            "uv",
            "run",
            "python",
            "main.py",
            "evaluation",
            "--llm_outputs_path",
            llm_outputs_fake,
            "--use_existing_csv",
            "True",
        ]
        print(f"--- Running metrics recomputation for {eval_dir} ---")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully recomputed metrics for {eval_dir}\n")
        except subprocess.CalledProcessError as e:
            print(f"Failed to recompute metrics for {eval_dir}: {e}\n")


if __name__ == "__main__":
    main()
