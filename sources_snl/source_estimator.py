import matplotlib 

from pathlib import Path
import mne
import numpy as np
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import matplotlib.image as mpimg
import subprocess as sp
import shutil

from scipy.spatial import KDTree
from mne.transforms import apply_trans
from mne.surface import read_surface, _compute_nearest
from scipy import sparse
import xarray as xr 
from nibabel.funcs import concat_images

import trimesh
from nibabel.freesurfer.io import read_geometry, write_geometry
from copy import deepcopy


def run_command(command, dry_run=False):
    print("------------------ RUNNING A COMMAND IN BASH ------------------")
    print(command)    
    print("-------------------------- OUTPUT -----------------------------")
    if not dry_run:
        bash(command.replace("\\", "").split())
    print("------------------------ END OUTPUT ---------------------------\n")    


class VerboseCalledProcessError(sp.CalledProcessError):
    def __str__(self):
        if self.returncode and self.returncode < 0:
            try:
                msg = "Command '%s' died with %r." % (
                    self.cmd, signal.Signals(-self.returncode))
            except ValueError:
                msg = "Command '%s' died with unknown signal %d." % (
                    self.cmd, -self.returncode)
        else:
            msg = "Command '%s' returned non-zero exit status %d." % (
                self.cmd, self.returncode)

        return f'{msg}\n' \
               f'Stdout:\n' \
               f'{self.output}\n' \
               f'Stderr:\n' \
               f'{self.stderr}'


def bash(cmd, print_stdout=True, print_stderr=True):
    proc = sp.Popen(cmd, stderr=sp.PIPE, stdout=sp.PIPE)

    all_stdout = []
    all_stderr = []
    while proc.poll() is None:
        for stdout_line in proc.stdout:
            if stdout_line != '':
                if print_stdout:
                    print(stdout_line.decode(), end='')
                all_stdout.append(stdout_line)
        for stderr_line in proc.stderr:
            if stderr_line != '':
                if print_stderr:
                    print(stderr_line.decode(), end='', file=sys.stderr)
                all_stderr.append(stderr_line)

    stdout_text = ''.join([x.decode() for x in all_stdout])
    stderr_text = ''.join([x.decode() for x in all_stderr])
    if proc.wait() != 0:
        raise VerboseCalledProcessError(proc.returncode, cmd, stdout_text, stderr_text)



def get_epochs(raw,
               tmin=-0.2,            # start of each epoch (200ms before the trigger)
               tmax=1.0,             # end of each epoch (1000ms after the trigger)
               baseline=(-0.2, 0)):  # means from the first instant to t = 0)

    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=baseline)
    return epochs
    




def fix_intersecting_surfaces(inner_surface_path, outer_surface_path,
                              move_margin=1.5, out_path=None):

    inner_mesh = trimesh.Trimesh(*read_geometry(inner_surface_path, read_metadata=False)) 
    outer_mesh = trimesh.Trimesh(*read_geometry(outer_surface_path, read_metadata=False))  

    edges = np.array(trimesh.graph.vertex_adjacency_graph(outer_mesh).edges)
        
    ray_interceptor_inner = trimesh.ray.ray_triangle.RayMeshIntersector(inner_mesh)
    #ray_interceptor_inner = trimesh.ray.ray_pyembree.RayMeshIntersector

    outer_surface_path = Path(outer_surface_path)
    stem = outer_surface_path.stem
    name = outer_surface_path.name
    if out_path is None:
        out_path = outer_surface_path.parent / (stem + suffix + name[len(stem):])    
    
    out_vertices = deepcopy(outer_mesh.vertices)    
    
    # Instead of using the vertex normal, we take the median of the 
    # normal of the neighbor vertex in order for this direction
    # to be more robust to sharp angular changes.
    x = np.concatenate([edges[:, [1, 0]], edges])
    normal = [np.median(outer_mesh.vertex_normals[x[x[:, 0] == i][:, 1]], 0) 
              for i in np.arange(outer_mesh.vertices.shape[0])] 
    
    # Intersecting surface correction (from outer to inner)
    intersections, inds = ray_interceptor_inner.intersects_location(ray_origins=outer_mesh.vertices, 
                                                                    ray_directions=normal,
                                                                    multiple_hits=True)[:2]    
    delta1 = np.zeros(out_vertices.shape[0])
    
    print("inner: ", inner_surface_path)
    print("outer: ", outer_surface_path)

    if len(inds):
        dist = np.sqrt(((outer_mesh.vertices[inds, :]-intersections)**2).sum(axis=1))
        # <5 to avoid picking very long distance points        
        inds = inds[dist < 5]   
        dist = dist[dist < 5]
        if len(inds):
            msg = "The inner surface intersect the outer surface. Pushing back the outer " + \
                  "surface {} mm out of the inner surface. Saving the outer ".format(move_margin) + \
                  "surface as {}.".format(out_path)
            print(msg)
            # <5 to avoid picking very long distance points
            delta1[inds] = dist + move_margin

    # Intersecting surface correction (from inner to outer)
    delta2 = np.zeros(out_vertices.shape[0])

    closest, distance, triangle_id = trimesh.proximity.closest_point(outer_mesh, inner_mesh.vertices)

    normal = (inner_mesh.vertices - closest)/distance[:, None]
    face_normals = outer_mesh.face_normals[triangle_id]

    angles = trimesh.geometry.vector_angle(np.stack([normal, face_normals], axis=1))

    new_deltas_df = pd.DataFrame(dict(vertex_id=outer_mesh.faces[triangle_id[angles < 1.0]].ravel(), 
                                      delta=np.tile(distance[angles < 1.0] + move_margin, [3, 1]).T.ravel())
                                 ).groupby("vertex_id").max().reset_index()

    delta2[new_deltas_df.vertex_id] = new_deltas_df.delta    

    deltas = np.stack([delta1, delta2]).max(axis=0)    
    
    if np.all(deltas == 0):
        return
        
    out_vertices += deltas[:, None]*outer_mesh.vertex_normals

    volume_info = read_geometry(outer_surface_path, read_metadata=True)[2]
    write_geometry(out_path, out_vertices, outer_mesh.faces, volume_info=volume_info)
    return deltas


def correct_intersecting_meshes(fs_subject_path, subject, suffix=""):
    deltas = {}
    pial_lh_path = fs_subject_path / subject / "surf" / "lh.pial"
    pial_rh_path = fs_subject_path / subject / "surf" / "rh.pial"

    inner_skull_path_pattern = fs_subject_path / subject / "bem" / 'inner_skull{}.surf'
    outer_skull_path_pattern = fs_subject_path / subject / "bem" / 'outer_skull{}.surf'
    outer_skin_path_pattern = fs_subject_path / subject / "bem" / 'outer_skin{}.surf'       
    if suffix != "":  
        for path_pattern in [inner_skull_path_pattern, outer_skull_path_pattern, outer_skin_path_pattern]:
            shutil.copy(str(path_pattern).format(""), str(path_pattern).format(suffix))

    inner_skull_path = Path(str(inner_skull_path_pattern).format(suffix))
    outer_skull_path = Path(str(outer_skull_path_pattern).format(suffix))
    outer_skin_path = Path(str(outer_skin_path_pattern).format(suffix))
        
    inner_paths = [pial_lh_path, pial_rh_path, inner_skull_path, outer_skull_path]
    outer_paths = [inner_skull_path, inner_skull_path, outer_skull_path, outer_skin_path] 
    for inner_path, outer_path in zip(inner_paths, outer_paths):
        deltas[inner_path.name] = fix_intersecting_surfaces(inner_path, outer_path, out_path=outer_path, 
                                                            move_margin=0.5)
    return deltas


def compute_and_save_mri_fid(fs_subject, fs_subject_dir="/usr/local/freesurfer/subjects/", coord_frame="mri"):
    fid_path = Path(fs_subject_dir) / fs_subject / "bem" / f"{fs_subject}-fiducials.fif"
    try:
        fid_path.unlink()
    except FileNotFoundError:
        pass

    digs_mri = mne.coreg.get_mni_fiducials(fs_subject, subjects_dir=fs_subject_dir)
    mne.io.meas_info.write_fiducials(fid_path, digs_mri, coord_frame=coord_frame)
    
    
def read_fid(fs_subject, fs_subject_dir="/usr/local/freesurfer/subjects/", coord_frame="mri"):
    fid_path = Path(fs_subject_dir) / fs_subject / "bem" / f"{fs_subject}-fiducials.fif"
    return mne.io.meas_info.read_fiducials(fid_path)


def compute_and_save_trans(fs_subject, info, fs_subject_dir="/usr/local/freesurfer/subjects/",
                           scale=True):

    head_pts = np.array([dig["r"] for dig in info["dig"][:3]])

    raw_dir = Path(fs_subject_dir) / fs_subject / "bem"
    digs_mri = mne.io.meas_info.read_fiducials(raw_dir / f"{fs_subject}-fiducials.fif")[0]
    mri_pts = np.array([dig["r"] for dig in digs_mri])

    #trans = fit_fiducials(head_pts, mri_pts, n_scale_params=n_scale_params)
    trans = mne.coreg.fit_matched_points(head_pts, mri_pts, out='trans', scale=scale)
    trans = mne.Transform('head', 'mri', trans)

    mne.write_trans(mne.coreg.trans_fname.format(raw_dir=raw_dir, subject=fs_subject), 
                    trans, overwrite=True)
    

def validate_coregistration(fs_subject, info, subject, fs_subject_dir="/usr/local/freesurfer/subjects/", 
                            save=True, trans=None, src=None, bem_sol=None):

    fig = mne.viz.plot_alignment(info, trans=trans,
                                 subject=fs_subject,
                                 subjects_dir=fs_subject_dir, surfaces='head',
                                 show_axes=True, dig="fiducials", 
                                 eeg=["original", "projected"],
                                 coord_frame='mri', mri_fiducials=True,
                                 src=src, bem=bem_sol)

    fig.plotter.off_screen = True

    Path(subject).mkdir(exist_ok=True)
    mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80, distance=0.6)
    if save:
        fig.plotter.screenshot(f"{subject}/coregistration_{fs_subject}_1.png")

        mne.viz.set_3d_view(figure=fig, azimuth=45, elevation=80, distance=0.6)
        fig.plotter.screenshot(f"{subject}/coregistration_{fs_subject}_2.png")

        mne.viz.set_3d_view(figure=fig, azimuth=270, elevation=80, distance=0.6)
        fig.plotter.screenshot(f"{subject}/coregistration_{fs_subject}_3.png")
    

#https://mne.tools/0.21/auto_tutorials/source-modeling/plot_background_freesurfer_mne.html
def get_transposed_long_format(img):
    inds = np.where(np.ones_like(img.dataobj))
    table = np.array([*inds, np.asanyarray(img.dataobj)[inds].astype(float)]).T  
    tmp = table.copy()
    tmp[:, -1] = 0
    tmp[:, :3] = nib.affines.apply_affine(img.affine, tmp[:, :3])
    
    tmp[:, -1] = table[:, -1]
    return pd.DataFrame(tmp, columns=["x", "y", "z", "values"])





def remove_lesion_borders(lesion_margin, fwd, lesion_img, subjects_dir, subject, trans, info, bem_sol):
    if lesion_margin == 0:
        return fwd

    pruned_src = deepcopy(fwd["src"])
    sources_left = pd.DataFrame(pruned_src[0]["rr"][pruned_src[0]["inuse"].astype('bool')]*1000)
    sources_right = pd.DataFrame(pruned_src[1]["rr"][pruned_src[1]["inuse"].astype('bool')]*1000)

    rr_mm, tris, volume_info = mne.read_surface(f"{subjects_dir}/{subject}/surf/lh.pial", 
                                read_metadata=True)
    lesion = get_transposed_long_format(lesion_img)
    vox = lesion[lesion["values"].astype(bool)].values
    lesion = pd.DataFrame(vox[:, :3] - volume_info["cras"])

    kdtree = KDTree(sources_left)
    pruned_sources = kdtree.query_ball_point(lesion[[0, 1, 2]], r=lesion_margin)
    pruned_sources = np.unique(np.concatenate(pruned_sources))
    
    inuse_inds = np.where(pruned_src[0]["inuse"])[0][~np.in1d(np.arange(pruned_src[0]["nuse"]), pruned_sources)]

    pruned_src[0]["inuse"] = np.in1d(np.arange(pruned_src[0]["np"]), inuse_inds).astype(int)
    pruned_src[0]["nuse"] = np.sum(pruned_src[0]["inuse"])

    return mne.make_forward_solution(info, trans=trans, src=pruned_src,
                                     bem=bem_sol, eeg=True, mindist=5.0)




def save_surface_src_volume(stc, subject, subjects_dir, time_downsample_factor=20, 
                            save_file_name="data.nii.gz"):
    n_vertices = sum(len(v) for v in stc.vertices)
    offset = 0
    surf_to_mri = 0.
    for hi, hemi in enumerate(['lh', 'rh']):
        ribbon = nib.load(Path(subjects_dir / subject / 'mri' / f'{hemi}.ribbon.mgz'))
        xfm = ribbon.header.get_vox2ras_tkr()
        mri_data = np.asanyarray(ribbon.dataobj)
        ijk = np.array(np.where(mri_data)).T
        xyz = apply_trans(xfm, ijk) / 1000.
        row_ind = np.where(mri_data.ravel())[0]
        data = np.ones(row_ind.size)
        rr = read_surface(Path(subjects_dir / subject / 'surf'/ f'{hemi}.white'))[0]
        rr /= 1000.
        rr = rr[stc.vertices[hi]]
        col_ind = _compute_nearest(rr, xyz) + offset
        surf_to_mri = surf_to_mri + sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(mri_data.size, n_vertices))
        offset += len(stc.vertices[hi])

    source_data = xr.DataArray(stc.data, dims=["sources", "time"]) \
                    .rolling(time=time_downsample_factor, center=True) \
                    .mean()[:, time_downsample_factor//2:-time_downsample_factor//2:time_downsample_factor]

    data_all_times = []
    for time_data in source_data.transpose("time", "sources").values:
        data = surf_to_mri.dot(time_data)
        data = data.reshape(mri_data.shape).astype("float32")
        data_all_times.append(nib.Nifti1Image(data, ribbon.affine))

    nib.save(concat_images(data_all_times), save_file_name)


def get_stcs(epochs, fwd, src_kwargs): 
    noise_cov = mne.compute_covariance(epochs, tmax=0, method=['shrunk', 'empirical'], rank=None, verbose=True)
    fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info)
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)
    return mne.minimum_norm.apply_inverse(epochs.average(), inverse_operator, pick_ori=None, 
                                          verbose=True, **src_kwargs)


class SourceEstimator:

    def __init__(self, root_path, subjects_dir, group, subjects, recompute, 
                 time_downsample_factor, event_types, src_kwargs, 
                 lesion_margin, result_dir="./results", plot=True,
                 preprocess=True):

        self.root_path = Path(root_path)
        self.subjects_dir = Path(subjects_dir)
        self.group = group
        self.subjects = subjects
        self.recompute = recompute 
        self.time_downsample_factor = time_downsample_factor

        self.event_types = event_types
        self.src_kwargs = src_kwargs

        self.lesion_margin = lesion_margin
        self.result_dir = Path(result_dir)
        self.plot = plot
        self.preprocess = preprocess

        self.result_dir.mkdir(exist_ok=True, parents=True)


    def get_file_name(self, subject):
        files = list((self.root_path / self.group / subject).glob("*.set"))

        if len(files) == 0:
            raise ValueError(f"There is not .set file in {self.root_path / self.group / subject}.")
        if len(files) > 1:
            raise ValueError(f"There is more than one .set file in {self.root_path / self.group / subject}.")
        return files[0]

    def get_raw(self, subject):
        file_name = self.get_file_name(subject)  
        raw = mne.io.read_raw_eeglab(file_name, preload=True)
        raw.load_data()

        raw.set_eeg_reference(projection=True)

        # Setting the montage
        # Data were recorded with a 64 BrainVision actiCAP active electrodes. 
        # As far as I know, this cap follows the 10-20 system.
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False)

        if self.plot:
            fig = raw.plot_psd()
            fig.savefig(self.result_dir / f"{subject}_psd.png")

        return raw

    def get_mri_images(self, subject, image_kinds=("T1.nii", "rT2.nii", "rLesion.nii")):
        images = {}
        path = self.root_path / self.group / subject
        for image_kind in image_kinds:
            paths = [path for path in path.glob("*") 
                            if image_kind in path.name]
            if len(paths) == 0:
                raise ValueError(f"There is no file with '{image_kind}' in {path}.")
            if len(paths) > 1:
                raise ValueError(f"There is more than one file with '{image_kind}' in {path}.")

            if self.preprocess:
                path_out = self.subjects_dir / subject / "mri" / paths[0].with_suffix(".nii.gz").name
                
                command = f"mri_convert --conform {paths[0]} {path_out}"    
                
                print("Running command:", command)
                run_command(command)
            else:
                path_out = paths[0]

            images[image_kind] = nib.load(path_out)
            
        return images

    def load_src(self, subject):
        sources_dir = self.subjects_dir / subject / "sources"
        sources_dir.mkdir(exist_ok=True)

        src_path = sources_dir / f"{subject}_oct6_src.fif"

        if not src_path.exists() or self.recompute:
            src = mne.setup_source_space(subject, spacing='oct6', surface='white', 
                                        subjects_dir=self.subjects_dir, add_dist=True, 
                                        n_jobs=1, verbose=None)
            src.save(src_path, overwrite=True)
        else:
            src = mne.read_source_spaces(src_path)

        return src


    def make_bem(self, subject):

        ### Use the watershed algorithm to get the surfaces
        bem_dir = self.subjects_dir / subject / "bem"

        if not bem_dir.exists() or self.recompute:
            mne.bem.make_watershed_bem(subject, subjects_dir=self.subjects_dir, show=True,
                                    overwrite=True, verbose=None)

        if self.plot:
            plot_bem_kwargs = dict(
                subject=subject, subjects_dir=self.subjects_dir,
                brain_surfaces='white', orientation='coronal',
                slices=[50, 100, 150, 200])

            fig = mne.viz.plot_bem(**plot_bem_kwargs)   
            fig.savefig(self.result_dir / f"{subject}_bem_surfaces.png")        



        ### Correct intersecting surfaces
        if self.recompute:
            correct_intersecting_meshes(self.subjects_dir, subject)


        ### Make a BEM model based on the available surfaces
        bem_surfaces_path = bem_dir / f"{subject}_bem.h5"

        if not bem_surfaces_path.exists() or self.recompute:
            bem_surfaces = mne.make_bem_model(subject, ico=4, conductivity=(0.3, 0.006, 0.3), 
                                            subjects_dir=self.subjects_dir, verbose=None)

            mne.write_bem_surfaces(bem_surfaces_path, bem_surfaces, 
                                overwrite=True, verbose=None)
        else:
            bem_surfaces = mne.read_bem_surfaces(bem_surfaces_path)


        ### Make the conductor model
        bem_sol_path = bem_dir / f"{subject}_bem_sol.h5"

        if not bem_sol_path.exists() or self.recompute:
            bem_sol = mne.make_bem_solution(bem_surfaces, verbose=None)
            mne.write_bem_solution(bem_sol_path, bem_sol, 
                                overwrite=True, verbose=None)
        else:
            bem_sol = mne.read_bem_solution(bem_sol_path)    

        return bem_sol

    def coregister(self, subject, info, src, bem_sol):
        trans_path = self.subjects_dir / subject / "bem" / f"{subject}-trans.fif"

        if not trans_path.exists() or self.recompute:
            compute_and_save_mri_fid(subject, fs_subject_dir=self.subjects_dir, coord_frame="mri")
            compute_and_save_trans(subject, info, fs_subject_dir=self.subjects_dir, scale=True)
            trans = mne.read_trans(trans_path)
            validate_coregistration(subject, info, subject, self.subjects_dir, True, 
                                    trans, src, bem_sol)
        else:
            trans = mne.read_trans(trans_path)

        if self.plot:
            fig, axes = plt.subplots(1, 3, figsize=(25, 8))

            for i, ax in enumerate(axes):
                img = mpimg.imread(f"{subject}/coregistration_{subject}_{i+1}.png")
                ax.imshow(img)
                ax.set_axis_off()
                
            fig.tight_layout()
            fig.savefig(self.result_dir / f"{subject}_coregistration.png")                

        return trans


    def process_all_subjects(self):
        for subject in self.subjects:
            self.process_subject(subject)

    def process_subject(self, subject):
        raw = self.get_raw(subject)
        images = self.get_mri_images(subject)
        src = self.load_src(subject)
        bem_sol = self.make_bem(subject)
        trans = self.coregister(subject, raw.info, src, bem_sol)
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                        bem=bem_sol, eeg=True, mindist=5.0)
        fwd = remove_lesion_borders(self.lesion_margin, fwd, images["lesion"], self.subjects_dir, 
                                    subject, trans, raw.info, bem_sol)

        for event_type, kwargs in self.event_types.items():
            epochs = get_epochs(raw, **kwargs)
            stcs = get_stcs(epochs, fwd, self.src_kwargs)
            file_name = f'eLORETA_surface_{subject}_{event_type.replace("/", "_").replace(" ", "_")}.nii.gz'
            save_surface_src_volume(stcs, subject, self.subjects_dir, self.time_downsample_factor,
                                    save_file_name=self.result_dir / file_name)


