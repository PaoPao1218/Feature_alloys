"""Descriptor extraction for alloy adsorption structures.

The module reads VASP/POSCAR structures, identifies adsorption-site atoms, and
exports local electronic/geometric descriptors for machine-learning workflows.

Two presets are built in:

* ``h``: H adsorption. H is treated as the adsorbate and removed from the slab
  before neighbor analysis.
* ``oh``: OH adsorption. O is preferred as the adsorption atom, H is used as a
  fallback, and both O/H are removed from the slab before neighbor analysis.

The command line entry points are intentionally thin wrappers around this file
so H and OH calculations share one implementation.

Reference
---------
Feature-extraction method adapted from:

    Wang, C., Wang, B., Wang, C., Li, A., Chang, Z., & Wang, R.
    "A machine learning model with minimized feature parameters for multi-type
    hydrogen evolution catalyst prediction."
    npj Computational Materials, 11, 111 (2025).
    https://doi.org/10.1038/s41524-025-01607-4
    Original code: https://github.com/wangchaobjut/Multi_Type_HERs
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LOGGER = logging.getLogger(__name__)

np = None
pd = None
psutil = None
neighborlist = None
read = None
ConvexHull = None
KDTree = None
Voronoi = None
gmean = None


def import_runtime_dependencies() -> None:
    """Import scientific dependencies lazily.

    Keeping these imports out of module import time lets ``--help`` work even
    before users install the scientific Python stack.
    """
    global np, pd, psutil, neighborlist, read, ConvexHull, KDTree, Voronoi, gmean

    if np is not None:
        return

    try:
        import numpy as _np
        import pandas as _pd
        import psutil as _psutil
        from ase import neighborlist as _neighborlist
        from ase.io import read as _read
        from scipy.spatial import ConvexHull as _ConvexHull
        from scipy.spatial import KDTree as _KDTree
        from scipy.spatial import Voronoi as _Voronoi
        from scipy.stats.mstats import gmean as _gmean
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    np = _np
    pd = _pd
    psutil = _psutil
    neighborlist = _neighborlist
    read = _read
    ConvexHull = _ConvexHull
    KDTree = _KDTree
    Voronoi = _Voronoi
    gmean = _gmean

FEATURE_COLUMNS = [
    "valence_electron_count",
    "electronegativity",
    "cohesive_energy",
    "d_electron_count",
    "p_electron_count",
    "s_electron_count",
    "first_ionization_energy",
    "covalent_radius",
    "electron_affinity_ev",
    "period_number",
    "atomization_enthalpy",
]

OUTPUT_COLUMNS = [
    "Structure",
    "Z_0",
    "ME_1",
    "SM_1",
    "dE0",
    "dE1",
    "dE",
    "pE0",
    "pE1",
    "pE",
    "sE0",
    "sE1",
    "sE",
    "ValE0",
    "ValE1",
    "ValE",
    "En0",
    "En1",
    "En",
    "IonE0",
    "IonE1",
    "IonE",
    "Rad0",
    "Rad1",
    "Rad",
    "CN",
    "BondL",
    "Surf_R",
    "LD_0",
    "LD_1",
    "CellAR",
    "CellAD",
    "V_Val",
    "AtomEn0",
    "AtomEn1",
    "AtomEn",
]


@dataclass(frozen=True)
class ModePreset:
    name: str
    default_input_dir: str
    default_output: str
    default_failed_log: str
    adsorbate_priority: tuple[int, ...]
    remove_atomic_numbers: tuple[int, ...]
    bridge_threshold: float
    hollow_threshold: float
    neighbor_cutoff: float
    fallback_neighbor_cutoff: float
    kdtree_radius_multiplier: float
    kdtree_min_radius: float


MODE_PRESETS = {
    "h": ModePreset(
        name="h",
        default_input_dir="./H_dataset.vasp",
        default_output="./H_dataset.xlsx",
        default_failed_log="./H_failed_files.csv",
        adsorbate_priority=(1,),
        remove_atomic_numbers=(1,),
        bridge_threshold=0.5,
        hollow_threshold=0.7,
        neighbor_cutoff=3.0,
        fallback_neighbor_cutoff=4.0,
        kdtree_radius_multiplier=1.2,
        kdtree_min_radius=2.0,
    ),
    "oh": ModePreset(
        name="oh",
        default_input_dir="./OH_dataset.vasp",
        default_output="./OH_dataset.xlsx",
        default_failed_log="./OH_failed_files.csv",
        adsorbate_priority=(8, 1),
        remove_atomic_numbers=(1, 8),
        bridge_threshold=0.7,
        hollow_threshold=0.9,
        neighbor_cutoff=4.0,
        fallback_neighbor_cutoff=5.0,
        kdtree_radius_multiplier=1.5,
        kdtree_min_radius=3.0,
    ),
}


@dataclass
class DescriptorConfig:
    mode: ModePreset
    input_dir: Path
    feature_table: Path
    output: Path
    failed_log: Path
    recursive: bool = False
    memory_threshold_gb: float = 8.0
    surface_z_threshold_ratio: float = 0.75
    local_density_cutoff: float = 5.0


@dataclass
class SurfaceMapping:
    surface: Atoms
    surface_to_original: list[int]


def get_memory_usage_mb() -> float:
    """Return current process memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def normalize_feature_columns(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported column-name variants and add missing columns."""
    feature_data = feature_data.copy()
    lower_to_original = {str(col).lower(): col for col in feature_data.columns}

    aliases = {
        "electron_affinity_ev": ("electron", "affinity"),
        "covalent_radius": ("covalent", "radius"),
        "atomization_enthalpy": ("atomization", "enthalpy"),
    }
    for target, keywords in aliases.items():
        if target in feature_data.columns:
            continue
        matches = [
            original
            for lower, original in lower_to_original.items()
            if all(keyword in lower for keyword in keywords)
        ]
        if matches:
            feature_data.rename(columns={matches[0]: target}, inplace=True)
        else:
            feature_data[target] = np.nan

    for col in FEATURE_COLUMNS:
        if col not in feature_data.columns:
            feature_data[col] = np.nan
    return feature_data


def load_feature_data(feature_table: Path) -> pd.DataFrame:
    """Load element features indexed by ``AtomicNumber``."""
    if not feature_table.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table}")

    feature_data = pd.read_excel(feature_table)
    if "AtomicNumber" not in feature_data.columns:
        raise ValueError("Feature table must contain an 'AtomicNumber' column.")

    feature_data.set_index("AtomicNumber", inplace=True)
    return normalize_feature_columns(feature_data)


def safe_gmean(values: Iterable[float]) -> float:
    """Geometric mean for finite values; returns NaN on invalid input."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    try:
        return float(gmean(arr))
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.nan


def valid_indices(indices: Iterable[int], structure: Atoms) -> list[int]:
    return [idx for idx in indices if 0 <= idx < len(structure)]


def get_main_atomic_number(atom_indices: Iterable[int], structure: Atoms) -> float:
    indices = valid_indices(atom_indices, structure)
    if not indices:
        return np.nan
    selected = [structure.get_atomic_numbers()[idx] for idx in indices]
    unique_nums, counts = np.unique(selected, return_counts=True)
    return float(unique_nums[np.argmax(counts)])


def calculate_surface_roughness(surface: Atoms) -> float:
    positions = surface.get_positions()
    return float(np.std(positions[:, 2])) if len(positions) else 0.0


def calculate_local_density(
    structure: Atoms,
    indices: Iterable[int],
    cutoff: float,
) -> float:
    indices = valid_indices(indices, structure)
    if not indices:
        return 0.0

    sphere_volume = (4.0 / 3.0) * np.pi * cutoff**3
    densities = []
    for center in indices:
        count = sum(
            1
            for idx in range(len(structure))
            if idx != center and structure.get_distance(center, idx, mic=True) < cutoff
        )
        densities.append(count / sphere_volume)
    return float(np.mean(densities)) if densities else 0.0


def calculate_mixing_entropy(indices: Iterable[int], structure: Atoms) -> float:
    indices = valid_indices(indices, structure)
    if not indices:
        return np.nan

    nums = [structure.get_atomic_numbers()[idx] for idx in indices]
    _, counts = np.unique(nums, return_counts=True)
    proportions = counts / len(nums)
    return float(-np.sum(proportions * np.log(proportions)))


def calculate_size_mismatch(
    indices: Iterable[int],
    structure: Atoms,
    feature_data: pd.DataFrame,
) -> float:
    indices = valid_indices(indices, structure)
    if len(indices) < 2:
        return np.nan

    atomic_numbers = structure.get_atomic_numbers()
    radii = []
    for idx in indices:
        atomic_number = atomic_numbers[idx]
        if atomic_number in feature_data.index:
            radius = feature_data.loc[atomic_number, "covalent_radius"]
            if pd.notna(radius):
                radii.append(float(radius))

    if len(radii) < 2:
        return np.nan
    mean_radius = np.mean(radii)
    if mean_radius == 0:
        return np.nan
    return float(np.std(radii) / mean_radius)


def calculate_lattice_strain(cell: np.ndarray) -> dict[str, float]:
    norms = [np.linalg.norm(axis) for axis in cell]
    if any(norm == 0 for norm in norms):
        return {"aspect_ratio": np.nan, "angle_deviation": np.nan}

    a, b, c = norms
    angles = []
    for i, j in [(1, 2), (0, 2), (0, 1)]:
        cosine = np.dot(cell[i], cell[j]) / (np.linalg.norm(cell[i]) * np.linalg.norm(cell[j]))
        cosine = np.clip(cosine, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))

    return {
        "aspect_ratio": float(c / ((a + b) / 2.0)),
        "angle_deviation": float(np.mean([abs(angle - 90.0) for angle in angles])),
    }


def calculate_voronoi_volume(structure: Atoms) -> float:
    """Return the mean finite Voronoi-cell volume for atom positions.

    This is a non-periodic Voronoi estimate. Infinite regions are counted as
    zero to preserve the conservative behavior of the original scripts.
    """
    try:
        vor = Voronoi(structure.get_positions())
        volumes = []
        for idx in range(len(structure)):
            region = vor.regions[vor.point_region[idx]]
            if -1 in region or len(region) == 0:
                volumes.append(0.0)
                continue
            volumes.append(float(ConvexHull(vor.vertices[region]).volume))
        return float(np.mean(volumes)) if volumes else 0.0
    except Exception as exc:
        LOGGER.debug("Voronoi volume failed: %s", exc)
        return 0.0


def find_surface_atoms(surface: Atoms, z_threshold_ratio: float) -> list[int]:
    positions = surface.get_positions()
    if len(positions) == 0:
        return []

    z_coords = positions[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_threshold = z_min + (z_max - z_min) * z_threshold_ratio
    surface_atoms = [idx for idx, z in enumerate(z_coords) if z >= z_threshold]
    return surface_atoms if surface_atoms else list(range(len(surface)))


def make_surface_without_adsorbates(
    structure: Atoms,
    remove_atomic_numbers: Sequence[int],
) -> SurfaceMapping:
    """Remove adsorbate atoms while preserving a surface-to-original map."""
    remove_set = set(remove_atomic_numbers)
    surface_to_original = [
        idx
        for idx, atomic_number in enumerate(structure.get_atomic_numbers())
        if atomic_number not in remove_set
    ]
    surface = structure[surface_to_original].copy()
    surface.set_pbc(structure.get_pbc())
    surface.set_cell(structure.get_cell())
    return SurfaceMapping(surface=surface, surface_to_original=surface_to_original)


def choose_adsorbate_atom(structure: Atoms, mode: ModePreset) -> int:
    """Choose the adsorption atom according to the mode priority."""
    atomic_numbers = structure.get_atomic_numbers()
    positions = structure.get_positions()

    for atomic_number in mode.adsorbate_priority:
        candidates = [
            idx for idx, number in enumerate(atomic_numbers)
            if number == atomic_number
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            z_values = [positions[idx][2] for idx in candidates]
            return candidates[int(np.argmin(z_values))]

    return int(np.argmax(positions[:, 2]))


def get_geometric_center_atoms(
    adsorbate_idx: int,
    structure: Atoms,
    candidate_original_indices: list[int],
    mode: ModePreset,
    num_candidates: int = 6,
) -> list[int]:
    if not candidate_original_indices:
        return []

    ads_pos = structure.get_positions()[adsorbate_idx]
    distances = [
        (idx, np.linalg.norm(ads_pos - structure.get_positions()[idx]))
        for idx in candidate_original_indices
    ]
    candidates = [idx for idx, _ in sorted(distances, key=lambda item: item[1])[:num_candidates]]

    if len(candidates) >= 2:
        midpoint = (
            structure.get_positions()[candidates[0]]
            + structure.get_positions()[candidates[1]]
        ) / 2.0
        if np.linalg.norm(ads_pos - midpoint) < mode.bridge_threshold:
            return candidates[:2]

    if len(candidates) >= 3:
        centroid = (
            structure.get_positions()[candidates[0]]
            + structure.get_positions()[candidates[1]]
            + structure.get_positions()[candidates[2]]
        ) / 3.0
        if np.linalg.norm(ads_pos - centroid) < mode.hollow_threshold:
            return candidates[:3]

    return candidates[:1]


def fallback_center_atoms(
    adsorbate_idx: int,
    structure: Atoms,
    excluded_indices: Iterable[int],
    mode: ModePreset,
) -> list[int]:
    excluded = set(excluded_indices)
    other_atoms = [idx for idx in range(len(structure)) if idx not in excluded]
    if not other_atoms:
        return []

    positions = structure.get_positions()
    ads_pos = positions[adsorbate_idx]
    other_positions = [positions[idx] for idx in other_atoms]
    tree = KDTree(other_positions)

    if len(structure) >= 2:
        sample_size = min(5, len(structure))
        distances = [
            np.linalg.norm(positions[i] - positions[j])
            for i in range(sample_size)
            for j in range(i + 1, sample_size)
        ]
        avg_bond_length = np.mean(distances) if distances else 1.0
    else:
        avg_bond_length = 1.0

    search_radius = max(avg_bond_length * mode.kdtree_radius_multiplier, mode.kdtree_min_radius)
    kd_indices = tree.query_ball_point(ads_pos, r=search_radius)
    return [other_atoms[idx] for idx in kd_indices]


def build_neighbor_list(surface: Atoms, cutoff: float) -> neighborlist.NeighborList:
    nl = neighborlist.NeighborList(
        [cutoff] * len(surface),
        skin=0.3,
        self_interaction=False,
        bothways=True,
    )
    nl.update(surface)
    return nl


def get_surface_neighbors(
    surface: Atoms,
    surface_center_indices: list[int],
    cutoff: float,
) -> tuple[list[int], neighborlist.NeighborList]:
    nl = build_neighbor_list(surface, cutoff)
    center_set = set(surface_center_indices)
    neighbors: set[int] = set()

    for idx in surface_center_indices:
        if 0 <= idx < len(surface):
            neighbors.update(nl.get_neighbors(idx)[0].tolist())

    return sorted(neighbors - center_set), nl


def electronic_descriptor(
    indices: Iterable[int],
    structure: Atoms,
    feature_data: pd.DataFrame,
) -> tuple[float, ...]:
    indices = valid_indices(indices, structure)
    if not indices:
        return tuple([np.nan] * 12)

    atomic_numbers = structure.get_atomic_numbers()
    valid_nums = [
        atomic_numbers[idx]
        for idx in indices
        if atomic_numbers[idx] in feature_data.index
    ]
    if not valid_nums:
        return tuple([np.nan] * 12)

    data = feature_data.loc[valid_nums]
    values = [safe_gmean(data[col]) for col in FEATURE_COLUMNS]
    valence, electronegativity, cohesive = values[0], values[1], values[2]
    psi = valence**2 / electronegativity if electronegativity else np.nan

    return (
        valence,
        electronegativity,
        psi,
        -cohesive,
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
    )


def descriptor_row(
    structure_path: Path,
    structure: Atoms,
    feature_data: pd.DataFrame,
    config: DescriptorConfig,
) -> dict[str, float | str]:
    structure.set_pbc([True, True, True])
    adsorbate_idx = choose_adsorbate_atom(structure, config.mode)

    mapping = make_surface_without_adsorbates(
        structure,
        config.mode.remove_atomic_numbers,
    )
    surface = mapping.surface
    if len(surface) == 0:
        raise ValueError("No surface atoms remain after removing adsorbates.")

    surface_atoms = find_surface_atoms(surface, config.surface_z_threshold_ratio)
    candidate_original_indices = [
        mapping.surface_to_original[idx] for idx in surface_atoms
    ]

    center_original_indices = get_geometric_center_atoms(
        adsorbate_idx,
        structure,
        candidate_original_indices,
        config.mode,
    )
    if not center_original_indices:
        center_original_indices = fallback_center_atoms(
            adsorbate_idx,
            structure,
            excluded_indices=[adsorbate_idx],
            mode=config.mode,
        )
    if not center_original_indices:
        raise ValueError("Could not identify center atoms around adsorbate.")

    original_to_surface = {
        original_idx: surface_idx
        for surface_idx, original_idx in enumerate(mapping.surface_to_original)
    }
    surface_center_indices = [
        original_to_surface[idx]
        for idx in center_original_indices
        if idx in original_to_surface
    ]
    if not surface_center_indices:
        raise ValueError("Center atoms could not be mapped to adsorbate-free surface.")

    neighbor_surface_indices, nl = get_surface_neighbors(
        surface,
        surface_center_indices,
        config.mode.neighbor_cutoff,
    )
    if not neighbor_surface_indices:
        neighbor_surface_indices, nl = get_surface_neighbors(
            surface,
            surface_center_indices,
            config.mode.fallback_neighbor_cutoff,
        )
    if not neighbor_surface_indices:
        raise ValueError("Could not identify nearest neighbors around center atoms.")

    neighbor_original_indices = [
        mapping.surface_to_original[idx] for idx in neighbor_surface_indices
    ]
    local_original_indices = sorted(set(center_original_indices + neighbor_original_indices))

    e0 = electronic_descriptor(center_original_indices, structure, feature_data)
    e1 = electronic_descriptor(neighbor_original_indices, structure, feature_data)
    e_all = electronic_descriptor(local_original_indices, structure, feature_data)

    (
        valence_electron_count_0,
        electronegativity_0,
        _psi0,
        cohesive_energy_0,
        d_electron_count_0,
        p_electron_count_0,
        s_electron_count_0,
        first_ionization_energy_0,
        covalent_radius_0,
        _electron_affinity_0,
        _period_number_0,
        atomization_enthalpy_0,
    ) = e0
    (
        valence_electron_count_1,
        electronegativity_1,
        _psi1,
        cohesive_energy_1,
        d_electron_count_1,
        p_electron_count_1,
        s_electron_count_1,
        first_ionization_energy_1,
        covalent_radius_1,
        _electron_affinity_1,
        _period_number_1,
        atomization_enthalpy_1,
    ) = e1
    (
        valence_electron_count,
        electronegativity,
        _psi,
        cohesive_energy,
        d_electron_count,
        p_electron_count,
        s_electron_count,
        first_ionization_energy,
        covalent_radius,
        _electron_affinity,
        _period_number,
        atomization_enthalpy,
    ) = e_all

    surface_coordination = [len(nl.get_neighbors(idx)[0]) for idx in range(len(surface))]
    max_coordination = max(surface_coordination) if surface_coordination else 0
    generalized_cn = (
        sum(surface_coordination[idx] for idx in neighbor_surface_indices) / max_coordination
        if max_coordination > 0
        else 0.0
    )

    center_distances = [
        structure.get_distance(adsorbate_idx, idx, mic=True)
        for idx in center_original_indices
    ]
    bond_length = np.round(safe_gmean(center_distances), 6) if center_distances else np.nan
    strain_features = calculate_lattice_strain(np.asarray(structure.get_cell()))

    return {
        "Structure": str(structure_path),
        "Z_0": get_main_atomic_number(center_original_indices, structure),
        "ME_1": calculate_mixing_entropy(neighbor_original_indices, structure),
        "SM_1": calculate_size_mismatch(neighbor_original_indices, structure, feature_data),
        "dE0": d_electron_count_0,
        "dE1": d_electron_count_1,
        "dE": d_electron_count,
        "pE0": p_electron_count_0,
        "pE1": p_electron_count_1,
        "pE": p_electron_count,
        "sE0": s_electron_count_0,
        "sE1": s_electron_count_1,
        "sE": s_electron_count,
        "ValE0": valence_electron_count_0,
        "ValE1": valence_electron_count_1,
        "ValE": valence_electron_count,
        "En0": electronegativity_0,
        "En1": electronegativity_1,
        "En": electronegativity,
        "IonE0": first_ionization_energy_0,
        "IonE1": first_ionization_energy_1,
        "IonE": first_ionization_energy,
        "Rad0": covalent_radius_0,
        "Rad1": covalent_radius_1,
        "Rad": covalent_radius,
        "CN": generalized_cn,
        "BondL": bond_length,
        "Surf_R": calculate_surface_roughness(surface),
        "LD_0": calculate_local_density(
            structure,
            center_original_indices,
            config.local_density_cutoff,
        ),
        "LD_1": calculate_local_density(
            structure,
            neighbor_original_indices,
            config.local_density_cutoff,
        ),
        "CellAR": strain_features["aspect_ratio"],
        "CellAD": strain_features["angle_deviation"],
        "V_Val": calculate_voronoi_volume(structure),
        "AtomEn0": atomization_enthalpy_0,
        "AtomEn1": atomization_enthalpy_1,
        "AtomEn": atomization_enthalpy,
    }


def discover_structure_files(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    patterns = ["*.vasp", "*.VASP", "*.POSCAR", "POSCAR"]
    files: list[Path] = []
    for pattern in patterns:
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        files.extend(path for path in iterator if path.is_file())
    return sorted(set(files))


def process_files(config: DescriptorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    import_runtime_dependencies()

    feature_data = load_feature_data(config.feature_table)
    structure_files = discover_structure_files(config.input_dir, config.recursive)
    if not structure_files:
        raise FileNotFoundError(f"No VASP/POSCAR files found in {config.input_dir}")

    rows = []
    failures = []
    memory_limit_mb = config.memory_threshold_gb * 1024

    for idx, structure_path in enumerate(structure_files, start=1):
        if get_memory_usage_mb() > memory_limit_mb:
            failures.append(
                {
                    "Structure": str(structure_path),
                    "Error": f"Skipped because memory usage exceeded {config.memory_threshold_gb} GB",
                }
            )
            continue

        try:
            structure = read(structure_path)
            rows.append(descriptor_row(structure_path, structure, feature_data, config))
        except Exception as exc:
            failures.append({"Structure": str(structure_path), "Error": str(exc)})
            LOGGER.warning("Failed to process %s: %s", structure_path, exc)
        finally:
            if idx % 5 == 0:
                gc.collect()

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    failed = pd.DataFrame(failures, columns=["Structure", "Error"])
    return result, failed


def ensure_parent_dir(path: Path) -> None:
    if path.parent and path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def build_parser(default_mode: str = "h") -> argparse.ArgumentParser:
    if default_mode not in MODE_PRESETS:
        raise ValueError(f"Unknown default mode: {default_mode}")
    preset = MODE_PRESETS[default_mode]

    parser = argparse.ArgumentParser(
        description="Extract alloy adsorption descriptors from VASP/POSCAR files.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS),
        default=default_mode,
        help="Adsorption preset. Use 'h' for H adsorption or 'oh' for OH adsorption.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(preset.default_input_dir),
        help="Directory containing .vasp/POSCAR structure files.",
    )
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=Path("./element futures.xlsx"),
        help="Excel file containing elemental features indexed by AtomicNumber.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(preset.default_output),
        help="Output Excel file for descriptors.",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=Path(preset.default_failed_log),
        help="CSV file for files that could not be processed.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for structure files recursively.",
    )
    parser.add_argument(
        "--memory-threshold-gb",
        type=float,
        default=8.0,
        help="Skip new files when process memory usage exceeds this value.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> DescriptorConfig:
    return DescriptorConfig(
        mode=MODE_PRESETS[args.mode],
        input_dir=args.input_dir,
        feature_table=args.feature_table,
        output=args.output,
        failed_log=args.failed_log,
        recursive=args.recursive,
        memory_threshold_gb=args.memory_threshold_gb,
    )


def run(default_mode: str = "h", argv: Sequence[str] | None = None) -> int:
    parser = build_parser(default_mode)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    start_time = time.time()
    config = config_from_args(args)
    result, failed = process_files(config)

    if result.empty:
        LOGGER.warning("No descriptor rows were generated.")
    else:
        ensure_parent_dir(config.output)
        result.to_excel(config.output, index=False)
        LOGGER.info("Saved descriptors to %s", config.output)

    if not failed.empty:
        ensure_parent_dir(config.failed_log)
        failed.to_csv(config.failed_log, index=False)
        LOGGER.warning("Saved %d failed-file records to %s", len(failed), config.failed_log)

    LOGGER.info("Processed descriptor rows: %d", len(result))
    LOGGER.info("Elapsed time: %.3f seconds", time.time() - start_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
