"""APLS evaluation of predicted road networks (MALiBU3D paper, Tab. 2).

Thin command line around ``pointcept/evaluation/network/eval_apls.py`` (see its docstring
for the pipeline and the outputs). Example::

    python tools/eval_network_apls.py \\
        --data_root data/malibu3d \\
        --save_path exp/malibu3d/network_run/result \\
        --split_manifest_csv data/malibu3d/tiles.csv \\
        --split test \\
        --out_dir exp/malibu3d/network_run/result
"""

from pointcept.evaluation.network.eval_apls import main

if __name__ == "__main__":
    main()
