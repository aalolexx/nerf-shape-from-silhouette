# ShNeRF: Silhouette Neural Radiance Fields
This Project is an experimental approach using the rather re-
cently introduced NeRF (Neural Radiance Fields) technology
to create 3D Meshes out of silhouette Images. Usually, the
Space Carving Method is used for this kind of task, but in the
scope of this project we aim to investigate the possibilities of
using NeRF for this purpose. Using existing NeRF Models as a
base, we created our own NeRF adaption ShNeRF, used it for
a mesh export and evaluated it’s performance. In the scope of
this project, we did not only implement a NeRF adaption but
also integrated a silhouette image generation (or image seg-
mentation) pipeline into a complete start-to-end configurable
wrapper pipeline.