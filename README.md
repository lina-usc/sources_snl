# Usage

1. Clone `git clone https://github.com/lina-usc/sources_snl.git`
2. Install `pip install --editable sources_snl`
3. Use:

```python
from sources_snl import SourceEstimator

estimator = SourceEstimator(root_path = "/Users/christian/Library/CloudStorage/OneDrive-UniversityofSouthCarolina/Data/Roozbeh",  # Root path for the data
                            subjects_dir = "/Applications/freesurfer/7.2.0/subjects",   # FreeSurfer subjects folder
                            group = "Aphasia",                                          # Subdirectory for the group 
                            subjects = ["ASubject51", "ASubject52"],                    # List of subjects to process
                            recompute = False,                                          # Recompute artifacts that have already been computed?
                            time_downsample_factor = 20,                                # Time downsampling factor for saving the sources as a 4D NIfTI file
                            event_types = {                                             # Event types compute the sources for and the info for epoching
                                            "Stimulus/S  2": {"tmin": -0.2, "tmax": 1.0, "baseline": (-0.2, 0)},
                                            "Stimulus/S 11": {"tmin": -0.2, "tmax": 1.0, "baseline": (-0.2, 0)}
                                          },
                            src_kwargs = {"method": "eLORETA", "lambda2": 0.1},         # Source estimation parameters
                            lesion_margin = 3,                                          # Border to remover around the lesion (in mm); set to 0 for deactivated lesion border removal
                            result_dir = "./results")                                   # Where the save the results and the figures

estimator.process_all_subjects()
```
