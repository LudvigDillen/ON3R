import argparse
from pathlib import Path
import h5py
import numpy as np
import cv2
import time

from gluefactory.datasets.generate_megadepth1500_st_tuples import plot_tuples

from ..settings import DATA_PATH
from gluefactory.utils.misc import start_debug
from .megadepth_utils import sample_star_topology_tuples


def load_pose(path):
    """ Load a pose from file. The resulting pose is w2cam (4x4 matrix)."""
    return np.linalg.inv(np.loadtxt(path, delimiter=' '))


def load_K(path):
    return np.loadtxt(path, delimiter=' ')[:-1, :-1]


def load_color_img(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


def to_str(x):
    """bytes → str, anything else → str untouched"""
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def canonical_key(scene, query_str, refs_seq):
    """Return a hashable key with refs in sorted order"""
    return (scene, query_str, tuple(sorted(refs_seq)))


def generate_data(args):
    data_folder = DATA_PATH / Path(args.subfolder)
    existing_combinations = set()
    t1 = time.time()  # Start the timer
    t2 = t1  # Initialize t2 to t1
    with open(data_folder / f"test_tuples/tuples_combinations_{args.text_file_id}.txt", "w") as tuple_combinations, \
         open(data_folder / f"test_tuples/tuples_{args.text_file_id}.txt", "w") as tuple_file, \
         h5py.File(data_folder / "scannet1500_overlap.h5", "r") as f:
        k_queries = args.num_tuples
        queries_sampled = 0
        while queries_sampled < k_queries and (t2 - t1) < 360:
            print(f"Sampling tuples: {queries_sampled}/{k_queries} after {t2 - t1:.2f} seconds")
            for i, (scene_name, grp) in enumerate(f.items()):
                if queries_sampled >= k_queries:
                    break
                overlap_matrix = grp["matrix"][:]
                frames = grp["frames"][:]

                queries, k_n_ref_tuples, sampling_success = sample_star_topology_tuples(
                    overlap_matrix, 1, args.tuple_length, start_query_ref_threshold=0.30)
                
                if args.plot:
                    n_queries = len(queries)
                    color_folder = Path(data_folder / scene_name / "color")
                    for i in range(n_queries):
                        if i > 5:
                            break
                        q_ind = queries[i]
                        ref_inds = k_n_ref_tuples[i]
                        q_img = load_color_img(color_folder / f"{int(frames[q_ind])}.jpg")
                        ref_imgs = [load_color_img(color_folder / f"{int(frames[ref_ind])}.jpg")
                                    for ref_ind in ref_inds]
                        plot_tuples(q_ind, ref_inds, overlap_matrix, args.tuple_length, q_img, ref_imgs)

                if sampling_success:
                    assert len(queries) == 1, "Only one query is expected in star topology tuples."
                    query_str = to_str(frames[queries[0]])
                    refs_list = [to_str(frames[idx]) for idx in k_n_ref_tuples[0]]
                    key = canonical_key(scene_name, query_str, refs_list)

                    if key in existing_combinations:
                        continue                                # skip duplicate combo
                    existing_combinations.add(key)

                    # pretty line — bytes removed, refs sorted for stability
                    pretty_line = (
                        f"scene: {scene_name}; "
                        f"query: {query_str}; "
                        f"refs: [{', '.join(sorted(refs_list))}]\n"
                    )
                    tuple_combinations.write(pretty_line)

                    queries_sampled += len(queries)
                    # Save the queries and tuples to text file
                    frames_q = frames[queries]
                    frames_refs = frames[k_n_ref_tuples]
                    frames_q_and_refs = np.concatenate((frames_q[:, None], frames_refs), axis=1)

                    rows, cols = frames_q_and_refs.shape
                    for j in range(rows):
                        # Image names
                        image_names = ""
                        for k in range(cols):
                            image_names += f"{scene_name}/color/{int(frames_q_and_refs[j, k])}.jpg "
                        
                        # Intrinsics (K)
                        intrinsic_str = ""
                        for k in range(cols):
                            K = load_K(data_folder / scene_name / "intrinsic/intrinsic_color.txt")
                            intrinsic_str += " ".join(map(str, K.flatten())) + " "

                        # Absolute poses (w2cam)
                        pose_str = ""
                        for k in range(cols):
                            pose = load_pose(
                                data_folder / scene_name / "pose" / f"{int(frames_q_and_refs[j, k])}.txt")[:3]
                            R = pose[:3, :3]
                            t = pose[:3, 3]
                            pose_str += " ".join(map(str, R.flatten())) + " "
                            pose_str += " ".join(map(str, t.flatten())) + " "

                        # Write the row to the file
                        str_row = (image_names + intrinsic_str + pose_str).strip()
                        tuple_file.write(str_row + "\n")
            t2 = time.time()
    print(f"Sampled {queries_sampled} tuples in {t2 - t1:.2f} seconds.")
    success_str = "successful" if queries_sampled >= k_queries else "unsuccessful"
    print(f"Thus, it was {success_str} to sample {k_queries} tuples.")

if __name__ == "__main__":
    """ Use one of the following to run the script:
    python -m gluefactory.datasets.generate_scannet1500_st_tuples -tf_id star_topology_2tuples -d
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("-n", "--num_tuples", type=int, default=750)
    parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
    parser.add_argument("-tf_id", "--text_file_id", type=str, default="")
    parser.add_argument("-t", "--tuple_length", type=int, default=2)
    parser.add_argument("-sf", "--subfolder", type=str, default="scannet1500")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot the tuples")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    if args.debug:
        start_debug()
    generate_data(args)
