# Feature Alloys

Descriptor extraction scripts for H and OH adsorption structures on alloy
surfaces. The code reads VASP/POSCAR files, identifies local adsorption-site
atoms, and exports electronic/geometric descriptors for downstream machine
learning or statistical analysis.

## What This Repository Does

- Supports H adsorption and OH adsorption workflows.
- Reads `.vasp`, `.VASP`, `.POSCAR`, and `POSCAR` files with ASE.
- Uses an element feature table indexed by `AtomicNumber`.
- Computes local descriptors for center atoms, neighbor atoms, and their local
  environment.
- Writes descriptor tables to Excel.
- Records failed structures in a CSV file instead of silently skipping them.

## Repository Files

```text
Feature_alloys/
|-- feature_alloys.py          # Shared descriptor extraction implementation
|-- H_features_extracted.py    # H adsorption command-line entry point
|-- OH_features_extracted.py   # OH adsorption command-line entry point
|-- H_features_extracted       # Compatibility wrapper for old command style
|-- OH_features_extracted      # Compatibility wrapper for old command style
|-- requirements.txt           # Python dependencies
|-- requirements               # Compatibility copy of requirements.txt
`-- README.md
```

## Installation

Create an environment with Python 3.9+ and install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:

- `numpy`, `pandas`: numerical and tabular data processing
- `scipy`: geometric mean, KDTree, Voronoi and ConvexHull calculations
- `ase`: VASP/POSCAR structure reading and neighbor lists
- `psutil`: memory monitoring
- `openpyxl`: Excel input/output support

## Required Input Files

### Structure Directory

Place your VASP/POSCAR structures in a directory, for example:

```text
H_dataset.vasp/
|-- structure_001.vasp
|-- structure_002.vasp
`-- ...
```

or:

```text
OH_dataset.vasp/
|-- structure_001.vasp
|-- structure_002.vasp
`-- ...
```

### Element Feature Table

The default feature table name is:

```text
element futures.xlsx
```

The spelling is kept for backward compatibility with the original project. You
can use any file name by passing `--feature-table`.

The Excel file must contain:

- `AtomicNumber`

Recommended columns:

- `valence_electron_count`
- `electronegativity`
- `cohesive_energy`
- `d_electron_count`
- `p_electron_count`
- `s_electron_count`
- `first_ionization_energy`
- `covalent_radius`
- `electron_affinity_ev`
- `period_number`
- `atomization_enthalpy`

Some common variants are normalized automatically, such as columns containing
both `electron` and `affinity`, or both `covalent` and `radius`. Missing feature
columns are filled with `NaN`.

## Usage

### H Adsorption

```bash
python H_features_extracted.py \
  --input-dir ./H_dataset.vasp \
  --feature-table "./element futures.xlsx" \
  --output ./H_dataset.xlsx \
  --failed-log ./H_failed_files.csv
```

The historical extensionless command still works:

```bash
python H_features_extracted
```

### OH Adsorption

```bash
python OH_features_extracted.py \
  --input-dir ./OH_dataset.vasp \
  --feature-table "./element futures.xlsx" \
  --output ./OH_dataset.xlsx \
  --failed-log ./OH_failed_files.csv
```

The historical extensionless command still works:

```bash
python OH_features_extracted
```

### Recursive Search

Use `--recursive` when structures are stored in nested subdirectories:

```bash
python H_features_extracted.py --recursive --input-dir ./H_dataset.vasp
```

### Shared Entry Point

You can also call the shared implementation directly:

```bash
python feature_alloys.py --mode h
python feature_alloys.py --mode oh
```

## Output Columns

The output Excel file contains one row per successfully processed structure.

| Column | Meaning |
| --- | --- |
| `Structure` | Input structure file path |
| `Z_0` | Main atomic number among adsorption-site center atoms |
| `ME_1` | Mixing entropy of neighbor atoms |
| `SM_1` | Size mismatch of neighbor atoms |
| `dE0`, `pE0`, `sE0` | d/p/s electron descriptors for center atoms |
| `dE1`, `pE1`, `sE1` | d/p/s electron descriptors for neighbor atoms |
| `dE`, `pE`, `sE` | d/p/s electron descriptors for local center+neighbor atoms |
| `ValE0`, `ValE1`, `ValE` | Valence electron descriptors |
| `En0`, `En1`, `En` | Electronegativity descriptors |
| `IonE0`, `IonE1`, `IonE` | First ionization energy descriptors |
| `Rad0`, `Rad1`, `Rad` | Covalent radius descriptors |
| `CN` | Generalized coordination descriptor |
| `BondL` | Adsorbate-to-center bond-length descriptor |
| `Surf_R` | Surface roughness from z-coordinate standard deviation |
| `LD_0`, `LD_1` | Local density around center and neighbor atoms |
| `CellAR` | Cell aspect ratio |
| `CellAD` | Mean deviation of cell angles from 90 degrees |
| `V_Val` | Mean finite Voronoi volume estimate |
| `AtomEn0`, `AtomEn1`, `AtomEn` | Atomization enthalpy descriptors |

## Adsorption-Site Logic

### H Mode

- H atoms are treated as adsorbates.
- H atoms are removed before constructing surface neighbor lists.
- If multiple H atoms are present, the H atom with the lowest z-coordinate is
  selected.

### OH Mode

- O atoms are preferred as adsorption atoms.
- If no O atom is found, H is used as a fallback.
- O and H atoms are removed before constructing surface neighbor lists.

For both modes, surface atoms are first identified from the upper portion of the
slab along z. The code then tests whether the adsorbate is closer to a bridge,
hollow, or top-like local geometry using midpoint/centroid deviations.

## Failure Handling

Files that cannot be processed are written to the failed-log CSV:

```text
Structure,Error
path/to/bad_structure.vasp,Could not identify nearest neighbors around center atoms.
```

This makes batch processing easier to debug than silently skipping failed
structures.

## Notes and Limitations

- The Voronoi descriptor uses a non-periodic Voronoi estimate. Infinite regions
  are counted as zero.
- The default element feature table name remains `element futures.xlsx` for
  compatibility. If you rename it to `element features.xlsx`, pass the new path
  with `--feature-table`.
- Descriptor quality depends on consistent structure orientation and sensible
  adsorbate placement.
