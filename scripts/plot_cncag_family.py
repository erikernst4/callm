from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def main(N, ns, output_dir):

    output_dir.mkdir(exist_ok=True, parents=True)

    qe = np.arange(N) / N

    fig, ax = plt.subplots(1,1, figsize=(8, 6))

    # Cn for c = 0
    correct = 1-qe
    incorrect = correct - np.log(1-qe)
    ax.plot(qe, correct, label='n=0', c='k')
    ax.plot(qe, incorrect, c='k', ls='--')

    # Cn for n natural
    for i, n in enumerate(ns):

        correct = (1-qe)**(n+1)
        incorrect = correct + (n+1)/n * (1 - (1-qe)**n)

        ax.plot(qe, correct, label=f'n={n}', c=f'C{i}')
        ax.plot(qe, incorrect, c=f'C{i}', ls='--')

    ax.set_xlabel('$q_e$', fontsize=16)
    ax.set_ylabel('$C_n$-CAG', fontsize=16)
    ax.legend()
    ax.set_xlim(-0.01, 1.01)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(-0.01, 3)
    ax.set_yticks(np.arange(0, 3.1, 0.25))
    ax.grid()
    
    plt.savefig(output_dir / "cncag_family.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--output-dir", type=str, default=f"{str(Path(__file__).parent.parent)}/outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    main(N=args.N, ns=args.ns, output_dir=output_dir)

