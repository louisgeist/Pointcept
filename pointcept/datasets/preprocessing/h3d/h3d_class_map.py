"""
H3D (Hessigheim 3D) semantic classes, March 2018 epoch (paper Table 1).

The LAS classification ids 0-10 are used as-is (no remap). ``CLASS_NAMES`` has a 12th
entry, "Void" (id 11), which configs use as ignore_index; it is not counted in
``NUM_CLASSES``.
"""

NUM_CLASSES = 11

CLASS_NAMES = [
    "Low Vegetation",
    "Impervious Surface",
    "Vehicle",
    "Urban Furniture",
    "Roof",
    "Façade",
    "Shrub",
    "Tree",
    "Soil or Gravel",
    "Vertical Surface",
    "Chimney",
    
    "Void",
]
