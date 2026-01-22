Atomic Adsorption Descriptor Calculation Code
=============================================
This code is designed to calculate structural and electronic descriptors for H adsorption and OH adsorption systems based on VASP/POSCAR files. It automates the identification of adsorption sites, neighbor atoms, and computes key features for material property analysis.

1. Core Functions
-----------------
1.1 Supported Adsorption Systems
- OH Adsorption: Prioritizes O atoms as adsorption centers; falls back to H atoms or the atom with the highest Z-coordinate if O is not found.
- H Adsorption: Takes H atoms as adsorption centers; identifies the closest surface atoms for descriptor calculation.

1.2 Key Descriptors Calculated
1. Electronic Descriptors: Valence electron count, electronegativity, ionization energy, d/p/s electron counts, cohesive energy, atomization enthalpy, etc.
2. Structural Descriptors: Coordination number (CN), bond length, surface roughness, local density, lattice strain, Voronoi volume.
3. Mixing & Size Descriptors: Mixing entropy (ME_1), size mismatch (SM_1) of neighbor atoms (ME_0/SM_0 are excluded).

1.3 Adsorption Site Identification
Automatically classifies adsorption sites for OH systems based on atomic coordinates:
- Bridge site: Deviation from the midpoint of two candidate atoms < 0.7 Å.
- Hollow site: Deviation from the centroid of three candidate atoms < 0.9 Å.
- Top site: Default if bridge/hollow site conditions are not met.

2. Environment Requirements
---------------------------
Install dependencies via pip:
pip install numpy pandas scipy ase psutil openpyxl

Dependency Explanation:
- numpy/pandas: Data processing and calculation.
- scipy: Geometric mean, KDTree, Voronoi/ConvexHull calculations.
- ase: Atomic structure reading, neighbor list construction (VASP/POSCAR support).
- psutil: Memory usage monitoring to avoid overflow.
- openpyxl: Excel result file writing.

3. File Structure
-----------------
project_root/
├── element futures.xlsx  # Element feature data (electronegativity, radius, etc.)
├── H_dataset.vasp/       # Folder containing H adsorption VASP/POSCAR files
├── OH_dataset.vasp/      # Folder containing OH adsorption VASP/POSCAR files
├── H_calculation.py      # H adsorption descriptor calculation code
├── OH_calculation.py     # OH adsorption descriptor calculation code
├── README.txt            # Project documentation
├── requirements.txt      # Dependencies configuration file
├── H_dataset.xlsx        # Output result file (auto-generated)
└── OH_dataset.xlsx       # Output result file (auto-generated)

4. Usage Guide
--------------
4.1 Prepare Input Files
1. Place VASP/POSCAR files of adsorption systems into the corresponding folder (H_dataset.vasp or OH_dataset.vasp).
2. Prepare element futures.xlsx with the following columns (case-insensitive matching supported):
   - AtomicNumber (index column, mandatory)
   - Valence electron count, electronegativity, cohesive energy
   - d/p/s electron counts, first ionization energy
   - Covalent radius, electron affinity, atomization enthalpy
   Missing columns will be filled with NaN automatically.

4.2 Run the Code
Execute the script corresponding to the adsorption system:
# For H adsorption systems
python H_calculation.py

# For OH adsorption systems
python OH_calculation.py

5. Output Explanation
---------------------
Results are saved as H_dataset.xlsx or OH_dataset.xlsx with the following key columns:

Column Name       | Description
------------------|-------------------------------------------
Structure         | Name of the input VASP/POSCAR file
Z_0               | Atomic number of the main center atom
ME_1 / SM_1       | Mixing entropy / size mismatch of neighbor atoms
dE0/dE1/dE        | d-electron count of center/neighbor/local atoms
En0/En1/En        | Electronegativity of center/neighbor/local atoms
CN / BondL        | Coordination number / average bond length
Surf_R / V_Val    | Surface roughness / average Voronoi volume

6. Notes
--------
- Memory Control: The code monitors memory usage (default threshold: 8 GB) and skips files when overflow is detected; adjust memory_threshold in the main function if needed.
- Neighbor Search: OH system uses 4.0 Å radius first, expands to 5.0 Å if no neighbors are found; H system uses 3.0 Å (adjustable in neighborlist.NeighborList).
- Error Handling: Skips corrupted files and prints detailed error logs via traceback for debugging.
- Surface Atoms: Identified by Z-coordinate threshold (75% of the Z-range); modify z_threshold_ratio in find_surface_atoms to adjust.

7. Troubleshooting
------------------
- "No VASP files found": Ensure the input folder path is correct and files end with .vasp or .POSCAR.
- Empty result file: Check if input structures have valid adsorption atoms (H/O) or if all files failed due to memory issues.
- NaN values in results: Missing element features in element futures.xlsx; supplement the corresponding columns.
