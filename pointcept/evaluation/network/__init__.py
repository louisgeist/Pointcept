"""Road network evaluation: predicted mask -> graph, and APLS against the ground-truth graphs.

- ``xy_raster_utils``: 1 m Lambert-93 grids, segment densification, masks, pixel graphs, RDP.
- ``label_utils``: read the ground-truth ``*_graph.gpkg``, write predicted graphs to GeoPackage.
- ``graph_pipeline``: predicted mask -> graph post-processing.
- ``prediction_stitch``: stitch per-tile predicted rasters onto one ROI raster.
- ``apls_metric``: APLS with a fixed protocol (``APLS_PROTOCOL``).
- ``eval_apls``: dataset-level evaluation (``run``), CLI in ``tools/eval_network_apls.py``.
"""
