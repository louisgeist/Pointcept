"""
FOR-instance V2 -- multi-site individual-tree LiDAR benchmark, used here as a
plain 3-class semseg dataset (ground / low vegetation / tree). See
`pointcept/datasets/preprocessing/forinstancev2/preprocess_forinstancev2.py`
for label/source definitions and on-disk layout.
"""

import json
import os

from .defaults import DefaultDataset
from .builder import DATASETS
from pointcept.utils.logger import get_root_logger


@DATASETS.register_module()
class ForInstanceV2Dataset(DefaultDataset):
    """Dataset for FOR-instance V2 preprocessed Pointcept scenes.

    Args:
        sources: Optional list of source names to keep (e.g. `["Yuchen"]` to
            train/eval on the single ULS-style site only). Matched against
            each scene's `meta.json["source"]` (written by the preprocessing
            script; one of BlueCat, CULS, NIBIO, NIBIO2, NIBIO_MLS, RMIT,
            SCION, TUWIEN, Yuchen). `None` (default) keeps every source.
    """

    def __init__(self, sources=None, **kwargs):
        if sources is not None and not isinstance(sources, (list, tuple)):
            raise TypeError("sources must be a list/tuple of source names, or None.")
        self.sources = list(sources) if sources is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        data_list = super().get_data_list()

        if self.sources is not None:
            wanted = set(self.sources)
            filtered = []
            seen_sources = set()
            for path in data_list:
                meta_path = os.path.join(path, "meta.json")
                if not os.path.isfile(meta_path):
                    raise FileNotFoundError(
                        f"Missing meta.json needed for `sources` filtering: {meta_path}"
                    )
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                source = meta.get("source")
                seen_sources.add(source)
                if source in wanted:
                    filtered.append(path)

            missing = wanted - seen_sources
            if missing:
                raise ValueError(
                    f"sources={sorted(missing)} not found among scenes in split "
                    f"'{self.split}' (available sources: {sorted(seen_sources)})."
                )

            get_root_logger().info(
                "ForInstanceV2 sources filter (split=%s, sources=%s): kept %d/%d scenes.",
                self.split,
                sorted(wanted),
                len(filtered),
                len(data_list),
            )
            data_list = filtered

        return data_list

    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])
