import open3d as o3d
# import matplotlib.pyplot as plt
# import numpy as np

path = "L:/Master/TU/3_HTCV_Proj/dnerf-hook-bin/exports/pointcloud/n-alex/point_cloud.ply"
pcd = o3d.io.read_point_cloud(path)
o3d.visualization.draw_geometries([pcd])

# print("ball pivoting...")
# radii = [0.005, 0.01, 0.02, 0.04, .1, .5]   
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([rec_mesh])
# help(o3d.geometry.TriangleMesh.create_from_point_cloud_poisson)
        # pcd (open3d.geometry.PointCloud): PointCloud from which the TriangleMesh surface is reconstructed. Has to contain normals.
        # depth (int, optional, default=8): Maximum depth of the tree that will be used for surface reconstruction. Running at depth d corresponds to solving on a grid whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the reconstructor adapts the octree to the sampling density, the specified reconstruction depth is only an upper bound.
        # width (float, optional, default=0): Specifies the target width of the finest level octree cells. This parameter is ignored if depth is specified
        # scale (float, optional, default=1.1): Specifies the ratio between the diameter of the cube used for reconstruction and the diameter of the samples' bounding cube.
        # linear_fit (bool, optional, default=False): If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.
        # n_threads (int, optional, default=-1): Number of threads used for reconstruction. Set to -1 to automatically determine it.

# print("poisson")
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd,
#         depth=7,
#         linear_fit=False)
    
# # o3d.visualization.draw_geometries([mesh])
# # o3d.io.write_triangle_mesh(filename="L:/Master/TU/3_HTCV_Proj/dnerf-hook-bin/exports/poisson/n-alex/o3d_poisson.ply", mesh=mesh)

# densities = np.asarray(densities)
# density_colors = plt.get_cmap('plasma')(
#     (densities - densities.min()) / (densities.max() - densities.min()))
# density_colors = density_colors[:, :3]
# density_mesh = o3d.geometry.TriangleMesh()
# density_mesh.vertices = mesh.vertices
# density_mesh.triangles = mesh.triangles
# density_mesh.triangle_normals = mesh.triangle_normals
# density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
# o3d.visualization.draw_geometries([density_mesh])