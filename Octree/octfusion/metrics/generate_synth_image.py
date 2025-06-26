import trimesh
import os
from multiprocessing import Pool, current_process
import multiprocessing as mp
from tqdm import tqdm
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(current_dir))
from utils.render_utils import generate_image_for_fid

os.environ['EGL_DEVICE_ID'] = '1'
category = "airplane"
snc_category_to_synth_id_13 = {
    'airplane': '02691156',
    'bench': '02828884',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'monitor': '03211117',
    'lamp': '03636649',
    'loudspeaker': '03691459',
    'rifle': '04090263',
    'sofa': '04256520',
    'table': '04379243',
    'telephone': '04401088',
    'vessel': '04530566',
}

cond = True
root_dir = "logs/airplane_union/uncond_1000epoch_lr2e-4"
fid_dir = f'{root_dir}/fid_images_{category}'
mesh_dir = f'{root_dir}/results_{category}'


os.makedirs(fid_dir, exist_ok=True)

meshes = os.listdir(mesh_dir)

def process_mesh(mesh):
    name = mesh[:-4]
    mesh_path = os.path.join(mesh_dir, mesh)
    mesh = trimesh.load(mesh_path, force="mesh")

    # Set the GPU for this process
    os.environ['EGL_DEVICE_ID'] = str(current_process()._identity[0] % 4)
    try:
        generate_image_for_fid(mesh, fid_dir, name)
    except:
        print(f'The mesh {name} occurs an error!')
        return
    print(f'The mesh {name} finish rendering')

num_processes = 20 # mp.cpu_count()
if num_processes > 1:
    with Pool(num_processes) as pool:  # Create a pool with 4 processes
        list(tqdm(pool.imap(process_mesh, meshes), total=len(meshes)))
else:
    for mesh in tqdm(meshes):
        process_mesh(mesh)
