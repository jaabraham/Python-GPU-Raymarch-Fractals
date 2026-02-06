# 3D Mandelbrot Raymarcher

A real-time interactive 3D Mandelbrot set renderer using raymarching with modern OpenGL. Features smooth iteration-based coloring, ambient occlusion, volumetric fog, and a dynamic z-slice mode for exploring the fractal's interior.

![3D Mandelbrot](screenshot.png)

## Features

### Core Rendering
- **Smooth Iteration Rendering** - Uses floating-point iteration counts for buttery-smooth color transitions and surfaces (no more banding artifacts)
- **Raymarched 3D Mandelbrot** - True 3D quaternion-based Mandelbrot set with distance field estimation
- **Binary Search Refinement** - Sub-step surface detection for precise hit points
- **Real-time Performance** - GPU-accelerated fragment shader rendering

### Visual Effects
- **Ambient Occlusion** - Screen-space ambient occlusion for added depth and crevice darkening
- **Volumetric Fog** - Distance-based atmospheric fog with light scattering
- **Smooth Color Palette** - 8-color gradient system with interpolation
- **Orbit Trap Coloring** - Additional detail from minimum orbit distance
- **Anti-Aliasing** - 4x SSAA (Super Sample Anti-Aliasing) toggle
- **Dynamic Shadows** - Shadow rays for crisp self-shadowing

### Interactive Features
- **Z-Slice Mode** - Cut the fractal at z=0 to reveal interior structure
- **Iteration Control** - Adjust min/max iteration bounds in real-time (float precision)
- **Camera Controls** - Full 6-DOF movement with pan, rotate, and zoom
- **Live Parameter Editing** - Adjust AO, fog, lighting on the fly

## Requirements

- Python 3.7+
- OpenGL 3.3+ capable GPU
- Dependencies:
  - `glfw`
  - `moderngl`
  - `numpy`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/3d-mandelbrot-raymarcher.git
cd 3d-mandelbrot-raymarcher

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install glfw moderngl numpy

# Run
python fractal.py
```

## Controls

### Movement
| Key | Action |
|-----|--------|
| **W / S** | Move forward / backward |
| **↑ / ↓** | Pan camera up / down |
| **← / →** | Pan camera left / right |

### Rotation
| Key | Action |
|-----|--------|
| **A / D** | Rotate left / right (yaw) |
| **Page Up / Page Down** | Rotate up / down (pitch) |

### View Settings
| Key | Action |
|-----|--------|
| **Q / E** | Scale (zoom in / out) |
| **R / F** | Step size adjustment (finer / coarser) |
| **U / J** | Max ray distance |
| **SPACE** | Toggle z-slice mode |
| **X** | Toggle anti-aliasing (4x SSAA) |

### Iteration Control (Float Precision)
| Key | Action |
|-----|--------|
| **T / G** | Max iterations (+/- 0.5) |
| **Y / H** | Min iterations (+/- 0.25) |

The smooth iteration system allows fractional iteration bounds, creating fluid transitions between shells.

### Visual Effects
| Key | Action |
|-----|--------|
| **V / B** | Ambient occlusion strength |
| **N / M** | Fog density |
| **, / .** | Volumetric light scattering |

### Lighting
| Key | Action |
|-----|--------|
| **I / K** | Move light up / down |
| **J / L** | Move light left / right |
| **C** | Randomize color palette |

### System
| Key | Action |
|-----|--------|
| **F1** | Print current state to console |
| **0** | Reset all parameters |
| **ESC** | Exit |

## Usage Tips

### Exploring the Fractal
1. **Start with defaults** - The default view shows the "main bulb" from a good angle
2. **Use smooth iterations** - Press T/G and Y/H to adjust iteration bounds smoothly
3. **Enable slice mode** - Press SPACE to cut the fractal and see interior iteration bands
4. **Fine step size** - Press R to decrease step size for sharper details (slower)
5. **Adjust fog** - Use N/M to control depth perception

### Performance
- Disable anti-aliasing (X) for better FPS on slower GPUs
- Increase step size (F) for faster but less precise rendering
- Reduce max iterations (G) for speed

### Finding Good Views
- The fractal has interesting structure around iteration bands 4-20
- Try min_iter=6, max_iter=12 for the first main shell
- Zoom in (Q) and decrease step size (R) for surface detail
- Use pan (arrow keys) to explore without changing angle

## Technical Details

### Rendering Pipeline
1. **Ray Generation** - Cast rays from camera through each pixel
2. **Marching Loop** - Step along ray, testing for Mandelbrot boundary crossing
3. **Binary Refinement** - When crossing detected, binary search for exact surface
4. **Shading** - Calculate normals, lighting, AO, and fog
5. **Slice Overlay** - In slice mode, render z=0 intersection with interior coloring

### The 3D Mandelbrot
The "Mandelbulb" uses the iteration:
```
z = z^2 + c  (in 3D quaternion-like algebra)
```

Where the 3D version uses spherical coordinates for the power operation:
```
x' = r^2 * sin(2*theta) * cos(2*phi) + c.x
y' = r^2 * sin(2*theta) * sin(2*phi) + c.y
z' = r^2 * cos(2*theta) + c.z
```

### Smooth Iterations
Instead of integer iteration counts, we use:
```
smooth_iter = i - log(log(|z|)) / log(2)
```

This provides sub-integer precision for color gradients and surface positioning.

## Window Title Info

The window title shows real-time stats:
```
3D Mandelbrot | FPS:60 | Pos:(x,y,z) | Dist:D | Step:S | Scale:Z | AA
```

- **FPS** - Current frame rate
- **Pos** - Camera position
- **Dist** - Distance from origin
- **Step** - Current ray step size
- **Scale** - Zoom level
- **AA** - Anti-aliasing on/off

## Troubleshooting

**Black screen**: Check OpenGL 3.3+ support
**Low FPS**: Disable AA, increase step size, or lower resolution
**No slice visible**: Ensure min_iter < max_iter and you're in slice mode (SPACE)
**Shader compile error**: Check that all dependencies are installed

## License

MIT License - Feel free to use, modify, and distribute!

## Credits

Created with ModernGL and GLFW for Python.

Inspired by the Mandelbulb and quaternion Julia sets.
