import numpy as np
np.infty = np.inf
#!/usr/bin/env python3
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio

p=argparse.ArgumentParser()
p.add_argument('--input', default='stem_with_tomatoes.glb')
p.add_argument('--output', default='stem_studio.mp4')
p.add_argument('--resolution', type=int, default=768)
p.add_argument('--frames', type=int, default=120)
p.add_argument('--fps', type=int, default=15)
p.add_argument('--distance', type=float, default=2.4)
p.add_argument('--exposure', type=float, default=1.0)
p.add_argument('--shadows', action='store_true', help='Enable directional shadows (slower)')
a=p.parse_args()

source=trimesh.load(a.input, force='scene', process=False)
if not source.geometry: raise ValueError('GLB contains no geometry')
bounds=source.bounds
center=np.mean(bounds,axis=0)
extent=float(np.max(bounds[1]-bounds[0]))
scene=pyrender.Scene.from_trimesh_scene(source, bg_color=[1,1,1,1], ambient_light=[0.65,0.65,0.65])

def look_at(eye,target):
    forward=np.asarray(target,dtype=float)-np.asarray(eye,dtype=float)
    forward/=np.linalg.norm(forward)
    up=np.array([0.,1.,0.])
    right=np.cross(forward,up)
    right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    mat=np.eye(4)
    mat[:3,0]=right
    mat[:3,1]=up
    mat[:3,2]=-forward
    mat[:3,3]=eye
    return mat

camera=pyrender.PerspectiveCamera(yfov=np.deg2rad(38),aspectRatio=1)
cam_node=scene.add(camera,pose=np.eye(4))

light_settings=[
    ((-1.8, 2.8, 3.0), 4.0),  # key: upper/front-left
    (( 2.8, 1.1, 1.0), 2.5),  # fill: front-right
    (( 0.6, 2.2,-3.0), 2.0),  # rim: behind
]
for direction,intensity in light_settings:
    light=pyrender.DirectionalLight(color=np.ones(3),intensity=intensity)
    eye=center+extent*np.asarray(direction,dtype=float)
    scene.add(light,pose=look_at(eye,center))

renderer=pyrender.OffscreenRenderer(viewport_width=a.resolution,viewport_height=a.resolution)
flags=pyrender.RenderFlags.RGBA
if a.shadows:
    flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL

def postprocess(rgb):
    img=np.asarray(rgb[:,:,:3],dtype=np.float32)/255.
    if a.exposure != 1.0:
        # Gamma-aware exposure adjustment, leaving whites at white.
        linear=np.power(np.clip(img,0,1),2.2)
        linear=1.0-np.power(1.0-linear,a.exposure)
        img=np.power(np.clip(linear,0,1),1/2.2)
    return np.uint8(np.clip(img*255+0.5,0,255))

try:
    with imageio.get_writer(a.output,fps=a.fps,codec='libx264',quality=8,macro_block_size=1) as writer:
        for i in range(a.frames):
            angle=2*np.pi*i/a.frames
            dist=max(extent*a.distance,0.1)
            eye=center+dist*np.array([np.cos(angle),0.18,np.sin(angle)])
            scene.set_pose(cam_node,look_at(eye,center))
            rgba,_=renderer.render(scene,flags=flags)
            writer.append_data(postprocess(rgba))
            if (i+1)%15==0: print(f'{i+1}/{a.frames} frames',flush=True)
finally:
    renderer.delete()
print('Saved:',os.path.abspath(a.output))
