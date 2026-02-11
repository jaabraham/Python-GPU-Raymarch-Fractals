# 3D Mandelbrot Raymarcher

A real-time 3D Mandelbrot fractal visualizer using raymarching with OpenGL.

![3D Mandelbrot](https://user-images.githubusercontent.com/placeholder.png)

## Features

- **Real-time Raymarching**: Renders 3D Mandelbrot set using GPU-accelerated raymarching
- **Iteration Shells**: Displays layered "onion-like" structure at iteration boundaries
- **Interactive Controls**: Full camera movement and parameter adjustment
- **Lighting & Shadows**: Diffuse lighting with soft shadows and orbit trap coloring
- **Z-Slice Mode**: View cross-sections of the fractal
- **Anti-Aliasing**: Optional 4x SSAA for smoother rendering
- **Smart Rendering**: Fractal only re-renders when you press keys (saves GPU/CPU when idle)

## Requirements

```bash
pip install moderngl glfw numpy
```

## Usage

Run the main fractal viewer:

```bash
python fractal.py
```

The program uses **smart rendering** - the fractal is only redrawn when you interact with it (key presses, camera movement). When idle, the display stays static and consumes minimal resources.

## Controls

### Movement & Camera

| Key | Action |
|-----|--------|
| `W` / `S` | Move forward / backward |
| `A` / `D` | Rotate left / right (yaw) |
| `↑` / `↓` / `←` / `→` | Pan view (move camera up/down/left/right) |
| `Page Up` / `Page Down` | Rotate up / down (pitch) |
| `Q` / `E` | Scale in / out |

### Rendering Parameters

| Key | Action |
|-----|--------|
| `R` / `F` | Adjust step size (precision) |
| `T` / `G` | Increase / decrease max iterations |
| `Y` / `H` | Increase / decrease min iterations (hollow threshold) |
| `+` / `-` | Increase / decrease max ray distance |
| `Space` | Toggle Z-slice mode |
| `X` | Toggle anti-aliasing (4x SSAA) |

### Lighting & Visual Effects

| Key | Action |
|-----|--------|
| `I` / `K` | Move light up / down |
| `J` / `L` | Move light left / right |
| `V` / `B` | Increase / decrease ambient occlusion strength |
| `N` / `M` | Increase / decrease fog density |
| `,` / `.` | Increase / decrease volumetric light scattering |
| `C` | Randomize color palette |

### General

| Key | Action |
|-----|--------|
| `F1` | Print current state to console |
| `0` | Reset all settings |
| `ESC` | Exit |

## File Overview

| File | Description |
|------|-------------|
| `fractal.py` | **Main program** - Full-featured 3D Mandelbrot raymarcher |
| `mandel_raymarch.py` | Alternative raymarching implementation |
| `mandelbrot3d.py` | 3D Mandelbrot utilities |
| `main.py` / `main_fixed.py` | Earlier implementations |
| `debug_mandel.py` | Debugging utilities |
| `test_simple.py` | Simple test cases |

## Technical Details

- **Resolution**: 1280×720 (windowed, resizable)
- **Algorithm**: Raymarching with analytic distance estimator
- **Shader**: GLSL fragment shader with orbit trap coloring
- **Default Params**: 100 max iterations, 6 min iterations
- **Rendering**: Smart render-on-demand (no continuous redraw)

## License

MIT License - feel free to use and modify!
