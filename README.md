# Usage

## Installation
1. Clone `git clone https://github.com/lina-usc/sources_snl.git`
2. Install `pip install --editable sources_snl`

## Updating and existing installation

1. Using the `cd` command, move to the directory `source_snl` containing the source of this package.
2. Update your local version of the source by running `git pull --rebased`

## Running the pipeline

Use (adjusting the configuration used in this example to fit your needs):

```python
from sources_snl import SourceEstimator

# Dictionary of subjects. The keys of the dictionary are the group identity
# an should be "Aphasia" and "Control". The values of these dictionaries are
# the list of subjects to process. Subjects in the group "Aphasia" will be processed
# with individual head models; subjects in the group "Control" will be processed
# with the standard fsaverage template.
subjects = {"Aphasia": ["ASubject51", "ASubject52"],
            "Control": ["CSubject20"]}

# Dictionary of the event types for which to compute the sources and the info for epoching. 
# The keys for this dictionary is the name of the events (as in the EEG file). Sources
# will be computed for each of the event types separately. The values for this dictionary
# are dictionaries themselves with the parameters used for epoching, such as the epoching 
# windows and the baseline for normalization. The full list of available arguments are as
# defined in the construction of the mne.Epochs class in 
# https://mne.tools/1.0/generated/mne.Epochs.html
event_types = {                                             
                "Stimulus/S  2": {"tmin": -0.2, "tmax": 1.0, "baseline": (-0.2, 0)},
                "Stimulus/S 11": {"tmin": -0.2, "tmax": 1.0, "baseline": (-0.2, 0)}
              }

estimator = SourceEstimator(root_path = "/Users/christian/Library/CloudStorage/OneDrive-UniversityofSouthCarolina/Data/Roozbeh",  # Root path for the data
                            subjects_dir = "/Applications/freesurfer/7.2.0/subjects",   # FreeSurfer subjects folder
                            subjects = subjects, 
                            recompute = False,                                          # Recompute artifacts that have already been computed?
                            time_downsample_factor = 5,                                 # Time downsampling factor for saving the sources as a 4D NIfTI file
                            event_types = event_types,
                            src_kwargs = {"method": "eLORETA", "lambda2": 0.1},         # Source estimation parameters
                            lesion_margin = 3,                                          # Border to remover around the lesion (in mm); set to 0 for deactivated lesion border removal
                            result_dir = "./results")                                   # Where the save the results and the figures

estimator.process_all_subjects()
```

# Output

All generated files are saved in the folder specified as `result_dir` as specificed in the example above. This folder contains validation plots, an nii.gz file containing the surface sources co-registered to the cortical ribbon of the subject, and the .stc files containing the native sources computed by `mne-python`.

