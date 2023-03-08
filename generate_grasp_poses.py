import copy
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import os
from models.dt_object import DTObject
from models.grippers import OnRobot_RG2
import numpy as np
from lib.geometry import homogeneous_mat_from_RT, distance_from_matrices, rotation_between_vectors
import begin

# --- candidate sampling ---
# curvature threshold (smaller more restrictive)
curvature_threshold = 0.006
# parallel threshold for the surfaces of the right and left finger
parallel_threshold = -0.97
n_candidates = 5000
gripper_width_margin = 0.03      # discard grasps with a too wide gripper width
min_grip_width = 0.001  # discard grips at too small features
# --- gpose generation
n_angle_variation_per_grasp = 24    # generate n grasps per surface point pair
rot_thr = 40/180*np.pi
voxel_size = 0.02  # seems to be object dependent
# --- collision validation
concave_LR_tolerance = 0.004       # allow mesh to curve behind finger by this much
# how long the fingers are beyond the grasp point
finger_over_length = 0.012
gripper_height = 0.03               # height of grippers for collision

# --- visualiztaion
show_mesh = False
show_curvature = False
show_grasp_candidates = False


def main(obj_id: str, force=False):
    t = time.perf_counter()
    obj = DTObject(meshpath=os.path.join("data", "models", obj_id,
                                         f"{obj_id}.ply"), show_axes=True, name='object')
    gripper = OnRobot_RG2()
    mesh = obj.dt_mesh

    flat_mesh = extract_flats_from_mesh(mesh)

    points, candidates = find_grasp_lines_on_mesh(
        flat_mesh, mesh, gripper)

    grasp_poses, grasp_widths = generate_grasp_poses_from_lines(
        points, candidates)

    if False:
        grasp_poses = np.array([np.eye(4)]*51)
        grasp_widths = [0.05]*len(grasp_widths)
        grasp_poses[:, 0, 3] = np.linspace(0.1, 0.2, 51)
        grasp_poses[:, :3, :] = [R.from_euler(
            'y', 30, degrees=True).as_matrix() @ g[:3, :] for g in grasp_poses]

    # add flipped poses
    rot = np.eye(4)
    rot[:3, :3] = R.from_euler(
        'xyz', [0, 0, 180], degrees=True).as_matrix()
    flipped = [pose @ rot for pose in grasp_poses]
    grasp_poses = np.concatenate([grasp_poses, flipped], axis=0)
    grasp_widths = np.concatenate([grasp_widths, grasp_widths], axis=0)

    grasp_poses, grasp_widths = voxel_down_sample_grasps(
        grasp_poses, grasp_widths, mesh)

    grasp_poses, grasp_widths = filter_colliding_gposes(
        grasp_poses, grasp_widths, gripper, mesh)

    print(
        f"Found {len(obj.gripping_poses)} grasp poses ({time.perf_counter()-t:.2f} sec)")

    if not force:
        ans = input("Overwrite gripping poses? (y|n): ")
        if ans != 'y':
            print("Discarded.")
            exit()

    obj.delete_gripping_poses()
    for pose, width in tqdm(zip(grasp_poses, grasp_widths), total=len(grasp_poses)):
        obj.register_gripping_pose(
            pose, correct_width=False, grasp_width=width, write=False)

    obj._write_gpose()

    print("Saving poses...")

    from visualize_grasp_poses import main
    main(obj_id, all=True)


def voxel_down_sample_grasps(grasp_poses, grasp_widths, mesh):
    min_bounds = mesh.get_min_bound()
    max_bounds = mesh.get_max_bound()
    center = (max_bounds + min_bounds) * 0.5

    grid_size = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int) + 1
    x, y, z = np.meshgrid(
        range(grid_size[0]), range(grid_size[1]), range(grid_size[2]), indexing='ij')
    x = (x - grid_size[0]//2) * voxel_size + center[0]
    y = (y - grid_size[1]//2) * voxel_size + center[1]
    z = (z - grid_size[2]//2) * voxel_size + center[2]

    coors = np.stack([x, y, z], -1).reshape((-1, 3))

    # anchor_pcd = o3d.geometry.PointCloud()
    # anchor_pcd.points = o3d.utility.Vector3dVector(coors)
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([anchor_pcd, mesh, origin])

    keep = []
    # rot_thr = 15/180*np.pi
    inds = np.arange(len(grasp_poses))

    from scipy.cluster.hierarchy import linkage, fcluster, leaders
    from scipy.spatial.distance import pdist

    print("Downsampling grasp candidates...")
    for anchor in tqdm(coors):
        pos_dists = np.linalg.norm(grasp_poses[:, : 3, 3] - anchor, axis=-1)
        at_anchor = pos_dists < voxel_size/2.0
        at_anchors = inds[at_anchor]
        if np.count_nonzero(at_anchor) > 0:
            # try to find rotated clusters, then pick one grasp of each rotation cluster that is closest to anchor
            rots = R.from_matrix(grasp_poses[at_anchors, :3, :3]).as_quat()
            rots *= np.sign(rots[:, -1:])  # flip quats positive (?)

            t = 1-np.cos(rot_thr)

            # dists = pdist(rots, metric=lambda a, b: (R.from_quat(
            #    b) * R.from_quat(a).inv()).magnitude())  # 'cosine')

            dists = pdist(rots, metric='cosine')
            Z = linkage(dists, method='complete')
            clusters = fcluster(Z, t, criterion='distance')
            #print(f"t: {t}, min: {np.min(dists)}, max {np.max(dists)}")

            #print(f"Adding {len(np.unique(clusters))}...")
            anchor_clusters = []
            for c in np.unique(clusters):
                inds_of_cluster = at_anchors[clusters == c]
                cluster_gposes = grasp_poses[inds_of_cluster]
                avg_z = np.mean(cluster_gposes[:, :3, 3], axis=0)
                dists = np.linalg.norm(
                    cluster_gposes[:, :3, 3] - avg_z, axis=1)

                anchor_clusters.append(inds_of_cluster[np.argmin(dists)])

            keep.extend(anchor_clusters)

    print(f"Reduced to {len(keep)} poses with {len(coors)} anchors.")
    return grasp_poses[keep], grasp_widths[keep]


def extract_flats_from_mesh(mesh):
    mesh = mesh.simplify_vertex_clustering(
        voxel_size=0.001,
        contraction=o3d.geometry.SimplificationContraction.Quadric)
    mesh = mesh.subdivide_loop(number_of_iterations=3)
    mesh = mesh.simplify_vertex_clustering(
        voxel_size=0.0016,
        contraction=o3d.geometry.SimplificationContraction.Quadric)

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()

    if show_mesh:
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

    # crop to head of duck
    # bbox = o3d.geometry.AxisAlignedBoundingBox(
    #    [-1.0, 0.02, -1.0], [1.0, 1.0, 1.0])
    # pcd = pcd.crop(bbox)

    print("Calculating surface curvature...")
    surface_curvature = caculate_surface_curvature(mesh)

    # colors = np.stack([surface_curvature, surface_curvature,
    #                  surface_curvature], axis=-1)
    colors = np.where(surface_curvature[:, np.newaxis] < curvature_threshold, [
        0.0, 1.0, 0.0], [1.0, 0.0, 0.0])

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    flat_mesh = o3d.geometry.TriangleMesh(mesh.vertices, mesh.triangles)
    flat_mesh.vertex_colors = mesh.vertex_colors
    flat_mesh.compute_vertex_normals()
    flat_mesh.remove_vertices_by_mask(colors[:, 0] > 0)

    if show_curvature:
        mesh_vis = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        o3d.visualization.draw_geometries(
            [mesh_vis, flat_mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

    return flat_mesh


def find_grasp_lines_on_mesh(flat_mesh, mesh, gripper):
    print("Sampling grasp points...")
    # mesh = mesh.subdivide_loop(number_of_iterations=1)
    # mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    # mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()

    # o3d.visualization.draw_geometries([mesh])
    flats = flat_mesh.sample_points_poisson_disk(n_candidates)
    flats.paint_uniform_color([0.0, 1.0, 0.0])

    # flat mesh can have holes! but then it will try to grasp from invalid points...
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    starts = np.asarray(flats.points)
    dirs = - np.asarray(flats.normals)

    # TODO
    # [x] cast a cone of lays and use the best aligned one (kinda like friction cone)
    # [ ] cast rays through object, make non-convex objects work

    rays = [np.concatenate([starts, dirs], axis=1)]
    to_ups = [rotation_between_vectors(
        dir, np.array([0., 0., 1.])) for dir in dirs]
    max_side = np.pi-np.arccos(parallel_threshold)
    rot_step = 137.5/180*np.pi
    n = 10
    for i in range(n):
        to_side_and_around = R.from_euler(
            'xz', [i/n*max_side, i*rot_step])
        rots = [to_up.inv()*to_side_and_around*to_up for to_up in to_ups]
        new_dirs = [rot.apply(dir) for rot, dir in zip(rots, dirs)]
        new_rays = np.array(np.concatenate([starts, new_dirs], axis=1))
        rays.append(new_rays)

    rays = np.stack(rays, axis=0).transpose([1, 0, 2])

    rays_t = o3d.core.Tensor.from_numpy(rays.astype(np.float32))
    res = scene.cast_rays(rays_t)
    dists = res['t_hit'].numpy()  # [n_starts, n_rays+1]
    tri_normals = res['primitive_normals'].numpy()  # [n_starts, n_rays+1, 3]

    print("Finding possible grasp locations...")
    points = []
    normals = []
    new_points = []
    new_normals = []
    for i in range(len(starts)):
        start = starts[i]  # (3, )
        normal = - dirs[i]
        dists[i, dists[i, :] < 1e-3] = 1.0
        connected_points = start + rays[i, :, 3:] * dists[i, :, np.newaxis]
        other_normals = tri_normals[i, :, :]

        # dists = np.linalg.norm(connected_points - start, axis=1)
        is_valid_width = dists[i] < (
            gripper.MAX_WIDTH - gripper_width_margin)
        is_valid_width = np.logical_and(
            is_valid_width, dists[i] > min_grip_width)

        connection = start - connected_points
        con_norm = np.linalg.norm(connection, axis=-1, keepdims=True)
        connection /= con_norm

        opposing_aligned = np.einsum('ij, ij->i', connection, other_normals)
        start_aligned = -normal @ connection.T
        alignments = np.mean([opposing_aligned, start_aligned], axis=0)

        is_aligned = alignments < parallel_threshold
        is_grasp = np.logical_and(is_valid_width, is_aligned)
        grasp_inds = np.where(is_grasp)[0]

        valid_alignments = alignments[is_grasp]
        if len(valid_alignments) == 0:
            continue

        best_grasp = grasp_inds[np.argmin(valid_alignments)]
        points.append(start)
        normals.append(normal)
        new_points.append(connected_points[best_grasp])
        new_normals.append(other_normals[best_grasp])

    points = np.array(points)
    normals = np.array(normals)
    new_points = np.array(new_points)
    new_normals = np.array(new_normals)

    candidates = list(zip(range(len(points)), range(
        len(points), len(points)+len(points))))

    points = np.concatenate([points, new_points], 0)
    normals = np.concatenate([normals, new_normals], 0)

    if show_grasp_candidates:
        start_pcd = o3d.geometry.PointCloud()
        start_pcd.points = o3d.utility.Vector3dVector(
            rays[:, :, :3].reshape((-1, 3)))
        start_pcd.normals = o3d.utility.Vector3dVector(
            -rays[:, :, :3].reshape((-1, 3)))
        start_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        ray_dists = dists.reshape((-1, 1))
        target_points = rays[:, :, :3].reshape(
            (-1, 3)) + rays[:, :, 3:].reshape((-1, 3)) * ray_dists
        target_normals = tri_normals.reshape((-1), 3)

        valid = np.isinf(ray_dists).squeeze()
        target_points = target_points[valid]
        target_normals = target_normals[valid]

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        target_pcd.normals = o3d.utility.Vector3dVector(target_normals)
        target_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        correspondences = o3d.geometry.LineSet()
        correspondences.points = o3d.utility.Vector3dVector(points)
        correspondences.lines = o3d.utility.Vector2iVector(candidates)
        o3d.visualization.draw_geometries(
            [start_pcd, correspondences])
    return points, candidates


def find_grasp_lines_on_mesh(flat_mesh, mesh, gripper):
    print("Sampling grasp points...")
    # mesh = mesh.subdivide_loop(number_of_iterations=1)
    # mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    # mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()

    # o3d.visualization.draw_geometries([mesh])
    flats = flat_mesh.sample_points_poisson_disk(2000)
    flats.paint_uniform_color([0.0, 1.0, 0.0])

    # flat mesh can have holes! but then it will try to grasp from invalid points...
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    starts = np.asarray(flats.points)
    dirs = - np.asarray(flats.normals)

    # TODO
    # [x] cast a cone of lays and use the best aligned one (kinda like friction cone)
    # [ ] cast rays through object, make non-convex objects work

    rays = [np.concatenate([starts, dirs], axis=1)]
    to_ups = [rotation_between_vectors(
        dir, np.array([0., 0., 1.])) for dir in dirs]
    max_side = np.pi-np.arccos(parallel_threshold)
    rot_step = 137.5/180*np.pi
    n = 10
    for i in range(n):
        to_side_and_around = R.from_euler(
            'xz', [i/n*max_side, i*rot_step])
        rots = [to_up.inv()*to_side_and_around*to_up for to_up in to_ups]
        new_dirs = [rot.apply(dir) for rot, dir in zip(rots, dirs)]
        new_rays = np.array(np.concatenate([starts, new_dirs], axis=1))
        rays.append(new_rays)

    rays = np.stack(rays, axis=0).transpose([1, 0, 2])

    rays_t = o3d.core.Tensor.from_numpy(rays.astype(np.float32))
    res = scene.cast_rays(rays_t)
    dists = res['t_hit'].numpy()  # [n_starts, n_rays+1]
    tri_normals = res['primitive_normals'].numpy()  # [n_starts, n_rays+1, 3]

    print("Finding possible grasp locations...")
    points = []
    normals = []
    new_points = []
    new_normals = []
    for i in range(len(starts)):
        start = starts[i]  # (3, )
        normal = - dirs[i]
        dists[i, dists[i, :] < 1e-3] = 1.0
        connected_points = start + rays[i, :, 3:] * dists[i, :, np.newaxis]
        other_normals = tri_normals[i, :, :]

        # dists = np.linalg.norm(connected_points - start, axis=1)
        is_valid_width = dists[i] < (
            gripper.MAX_WIDTH - gripper_width_margin)
        is_valid_width = np.logical_and(
            is_valid_width, dists[i] > min_grip_width)

        connection = start - connected_points
        con_norm = np.linalg.norm(connection, axis=-1, keepdims=True)
        connection /= con_norm

        opposing_aligned = np.einsum('ij, ij->i', connection, other_normals)
        start_aligned = -normal @ connection.T
        alignments = np.mean([opposing_aligned, start_aligned], axis=0)

        is_aligned = alignments < parallel_threshold
        is_grasp = np.logical_and(is_valid_width, is_aligned)
        grasp_inds = np.where(is_grasp)[0]

        valid_alignments = alignments[is_grasp]
        if len(valid_alignments) == 0:
            continue

        best_grasp = grasp_inds[np.argmin(valid_alignments)]
        points.append(start)
        normals.append(normal)
        new_points.append(connected_points[best_grasp])
        new_normals.append(other_normals[best_grasp])

    points = np.array(points)
    normals = np.array(normals)
    new_points = np.array(new_points)
    new_normals = np.array(new_normals)

    candidates = list(zip(range(len(points)), range(
        len(points), len(points)+len(points))))

    points = np.concatenate([points, new_points], 0)
    normals = np.concatenate([normals, new_normals], 0)

    if show_grasp_candidates:
        start_pcd = o3d.geometry.PointCloud()
        start_pcd.points = o3d.utility.Vector3dVector(
            rays[:, :, :3].reshape((-1, 3)))
        start_pcd.normals = o3d.utility.Vector3dVector(
            -rays[:, :, :3].reshape((-1, 3)))
        start_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        ray_dists = dists.reshape((-1, 1))
        target_points = rays[:, :, :3].reshape(
            (-1, 3)) + rays[:, :, 3:].reshape((-1, 3)) * ray_dists
        target_normals = tri_normals.reshape((-1), 3)

        valid = np.isinf(ray_dists).squeeze()
        target_points = target_points[valid]
        target_normals = target_normals[valid]

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        target_pcd.normals = o3d.utility.Vector3dVector(target_normals)
        target_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        correspondences = o3d.geometry.LineSet()
        correspondences.points = o3d.utility.Vector3dVector(points)
        correspondences.lines = o3d.utility.Vector2iVector(candidates)
        o3d.visualization.draw_geometries(
            [start_pcd, correspondences])
    return points, candidates


def generate_grasp_poses_from_lines(points, lines):
    # sample grasps from these lines
    angles = np.linspace(
        0, 2*np.pi, n_angle_variation_per_grasp, endpoint=False)
    grasp_poses = []
    grasp_widths = []
    print("Generating grasp pose candidates...")
    for idx in tqdm(range(len(lines))):
        i, j = lines[idx]
        midpoint = (points[i] + points[j])/2
        line_vec = points[j] - points[i]
        # corresponds with y axis of pose
        line_vec /= np.linalg.norm(line_vec)

        initial_z_vec = np.cross(line_vec, [1.0, 0.0, 0.0])
        initial_z_vec /= np.linalg.norm(initial_z_vec)

        z_vecs = np.array([R.from_rotvec(line_vec * angle).as_matrix() @ initial_z_vec.T
                           for angle in angles])

        x_vecs = np.array([np.cross(line_vec, z_vec) for z_vec in z_vecs])
        x_vecs = [x / np.linalg.norm(x) for x in x_vecs]
        rot_mats = [np.stack([x, line_vec, z], axis=-1)
                    for x, z in zip(x_vecs, z_vecs)]

        grasp_poses.extend([homogeneous_mat_from_RT(rot, midpoint)
                            for rot in rot_mats])
        grasp_width = np.linalg.norm(points[i] - points[j])
        grasp_widths.extend([grasp_width] * n_angle_variation_per_grasp)
    print(f"Generated {len(grasp_poses)} candidiates.")
    return np.array(grasp_poses), np.array(grasp_widths)


def cluster_grasp_poses(grasp_poses, grasp_widths):
    print(f"Downsampling and clustering grasp pose candidates...")
    clusters = np.zeros(len(grasp_poses))
    cluster_counter = 0

    arg_shuffle = list(range(len(grasp_poses)))
    np.random.shuffle(arg_shuffle)
    grasp_poses = np.array(grasp_poses)[arg_shuffle]
    grasp_widths = np.array(grasp_widths)[arg_shuffle]

    keep_inds = []

    # clusters as radius search, then start the next cluster as close as possible
    i = 0
    to_cluster = len(grasp_poses)
    previous_clustered = 0
    bar = tqdm(total=to_cluster)
    while True:
        if clusters[i] < 1:
            cluster_counter += 1
            current_cluster_id = cluster_counter
            clusters[i] = current_cluster_id
            keep_inds.append(i)  # keep the cluster intiating point
        else:
            break
            continue

        current_cluster_id = clusters[i]

        gpose = grasp_poses[i]
        other_poses = grasp_poses

        rel_grasp_points = other_poses[:, : 3, 3] - gpose[: 3, 3]
        # is inverted (in relative grasp pose frame)
        # rel_grasp_points = rel_grasp_points @ gpose[: 3, : 3].T
        pos_dists = np.linalg.norm(rel_grasp_points, axis=-1)

        rot_dists = R.from_matrix(gpose[:3, :3]).inv()
        rot_dists *= R.from_matrix(other_poses[:, :3, :3])
        rot_dists = rot_dists.magnitude()
        # print(rot_dists)
        # find rotation to antipodal grasp and take the minimum rot dist
        rot_dists_anti = R.from_matrix(gpose[:3, :3]).inv()
        rot_dists_anti *= R.from_euler('y', 180, degrees=True)
        rot_dists_anti *= R.from_matrix(other_poses[:, :3, :3])
        rot_dists_anti = rot_dists_anti.magnitude()

        rot_dists = np.min([rot_dists, rot_dists_anti], axis=0)

        # is_in_cluster = np.logical_and(rot_dists < rot_thr, pos_dists < pos_thr)
        normalized_dist = np.stack(
            [rot_dists/rot_thr, pos_dists/pos_thr], axis=-1)

        # normalized_dist = np.linalg.norm(normalized_dist, axis=-1)
        normalized_dist = np.sum(normalized_dist, axis=-1)

        is_in_cluster = normalized_dist < 1
        clusters[is_in_cluster] = current_cluster_id

        # print(f"In this cluster: {np.count_nonzero(is_in_cluster)}")

        # not_in_cluster = np.logical_and(normalized_dist >= 1, clusters == 0)
        # not_in_cluster = np.where(not_in_cluster)[0]
        # print(not_in_cluster)
        # if len(not_in_cluster) == 0:
        #    break
        not_clustered = np.logical_or(normalized_dist < 1, clusters > 0)
        normalized_dist[not_clustered] = np.inf
        i = np.argmin(normalized_dist)

        already_clustered = np.count_nonzero(clusters > 0)
        newly_clustered = already_clustered - previous_clustered
        previous_clustered = already_clustered
        bar.update(newly_clustered)

    bar.close()
    print(
        f"Found {len(np.unique(clusters))} clusters for {len(grasp_poses)} grasps")

    # for cluster in np.unique(clusters):
    # in_cluster = np.argwhere(clusters == cluster).T[0]  # squeeze to list
    # poses = grasp_poses[in_cluster]
    # mean_pos = np.mean(poses[:, :3, 3], axis=0)
    # dists = np.linalg.norm(poses[:, :3, 3] - mean_pos, axis=1)
    # keep = in_cluster[np.argmin(dists)]
    # alignments = grasp_alignments[in_cluster]
    # keep = in_cluster[np.argmin(alignments)]

    # keep_inds.append(keep)

    return grasp_poses[keep_inds], grasp_widths[keep_inds]


def filter_colliding_gposes(grasp_poses, grasp_widths, gripper, mesh):
    print("Filtering colliding grasps...")
    filtered = []
    filtered_grasp_widths = []
    left_right_filtered = 0
    depth_filterd = 0
    mesh_t_original = o3d.t.geometry.TriangleMesh.from_legacy(
        mesh).to(o3d.core.Device('CUDA:0'))

    for ind in tqdm(range(len(grasp_poses))):
        grasp_width = grasp_widths[ind]
        gpose = grasp_poses[ind]
        gdepth = gripper.get_valid_grasp_depth(grasp_width)

        clip_points = np.array([
            [-gripper_height/2.0, 0.0, 0.0, 1.0],
            [gripper_height/2.0, 0.0, 0.0, 1.0],
            [0.0, -grasp_width/2.0-concave_LR_tolerance, 0.0, 1.0],
            [0.0, grasp_width/2.0+concave_LR_tolerance, 0.0, 1.0],
            [0.0, 0.0, -gdepth, 1.0],
            [0.0, 0.0, finger_over_length, 1.0]
        ])
        clip_points = clip_points @ gpose.T
        clip_points = clip_points[:, :3]
        x_neg_point = clip_points[0]
        x_pos_point = clip_points[1]
        y_neg_point = clip_points[2]
        y_pos_point = clip_points[3]
        z_neg_point = clip_points[4]
        z_pos_point = clip_points[5]

        # remove everything at z+ from the gripper
        mesh_t = mesh_t_original.clone().clip_plane(
            point=z_pos_point, normal=-gpose[:3, 2])

        # remove everything 'above and below' the gripper's fingers
        mesh_t = mesh_t.clip_plane(point=x_pos_point, normal=-gpose[:3, 0])
        mesh_t = mesh_t.clip_plane(point=x_neg_point, normal=gpose[:3, 0])

        # o3d.visualization.draw_geometries([mesh_t.to_legacy()])

        # check for collisions on the y_pos side of the gripper
        left = mesh_t.clone().clip_plane(
            point=y_pos_point, normal=gpose[:3, 1])

        # check for collisions on the other side
        right = mesh_t.clone().clip_plane(
            point=y_neg_point, normal=-gpose[:3, 1])

        # check if the safe grip depth was violated
        depth = mesh_t.clip_plane(
            point=z_neg_point, normal=-gpose[:3, 2])

        left_right = left.is_empty() and right.is_empty()
        left_right_filtered += not left_right
        depth_filterd += not depth.is_empty()

        if left_right and depth.is_empty():
            filtered.append(gpose)
            filtered_grasp_widths.append(grasp_width)
        else:
            continue
            pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(clip_points))
            gpose_vis = copy.copy(origin)
            gpose_vis2 = copy.copy(origin)
            mesh2 = copy.copy(mesh)
            mesh2.translate([0.2, 0.0, 0.0])
            gpose_vis.transform(gpose)
            gpose_vis2.transform(gpose)
            gpose_vis2.translate([0.2, 0.0, 0.0])
            print("New")
            if not left.is_empty():
                print("\tleft")
            if not right.is_empty():
                print("\tright")
            if not depth.is_empty():
                print("\tdepth")

            right = right.to_legacy().paint_uniform_color([0.0, 1.0, 0.0])
            left = left.to_legacy().paint_uniform_color([1.0, 0.0, 0.0])
            depth = depth.to_legacy().paint_uniform_color([0.0, 0.0, 1.0])
            mesh_t = mesh_t.to_legacy().paint_uniform_color([0.9, 0.9, 0.9])

            o3d.visualization.draw_geometries(
                [mesh_t, pcd, left, right, depth, gpose_vis2, mesh2], mesh_show_back_face=True)

    print(
        f"Found {len(filtered)} valid grasps (LR: {left_right_filtered}, depth: {depth_filterd})")
    return np.array(filtered), np.array(filtered_grasp_widths)


def pca_compute(data, sort=True):
    """
        SVD decomposition
    """
    average_data = np.mean(data, axis=0)
    decentration_matrix = data - average_data
    H = np.dot(decentration_matrix.T, decentration_matrix)
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
    return eigenvalues


def caculate_surface_curvature(mesh):
    mesh.compute_adjacency_list()
    adjacency_list = mesh.adjacency_list
    points = np.asarray(mesh.vertices)
    num_points = len(points)
    curvature = []
    for i in tqdm(range(num_points)):
        idx = list(adjacency_list[i])

        # extend by neighbors of neighbors
        # n_of_n = [list(adjacency_list[n]) for n in idx]
        # n_of_n = [n for n_s in n_of_n for n in n_s]
        # idx.extend(n_of_n)
        idx = np.unique(idx).tolist()

        neighbors = points[idx, :]
        w = pca_compute(neighbors)
        delt = np.divide(w[2], np.sum(w), out=np.zeros_like(
            w[2]), where=np.sum(w) != 0)
        curvature.append(delt)
    curvature = np.array(curvature)

    # erode step
    filtered = curvature.copy()
    for i in range((num_points)):
        neighbors = list(adjacency_list[i])
        # min for dilate # max for erode
        filtered[i] = np.max(curvature[neighbors])

    filtered2 = filtered.copy()
    for i in range((num_points)):
        neighbors = list(adjacency_list[i])
        # min for dilate # max for erode
        filtered2[i] = np.max(filtered[neighbors])

    # smoothing step
    smooth_curvature = filtered2.copy()
    for i in range((num_points)):
        neighbors = list(adjacency_list[i])
        weights = np.linalg.norm(points[neighbors] - points[i], axis=1)
        weights /= np.sum(weights)
        smooth_curvature[i] = np.sum(filtered[neighbors] * weights)

    return smooth_curvature


@ begin.start
def run(obj_id):
    main(obj_id)
