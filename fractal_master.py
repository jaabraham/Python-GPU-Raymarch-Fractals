"""
3D Mandelbrot Raymarcher - Iteration Shells with Hollow Center
Based on VEX/OpenCL logic - Shows shells at iteration boundaries
Window: 1280x720
"""
import glfw
import moderngl
import numpy as np
import math

# -------------------------------------------------
# Initialize GLFW
# -------------------------------------------------
if not glfw.init():
    raise RuntimeError("GLFW init failed")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

WIDTH, HEIGHT = 1280, 720
window = glfw.create_window(WIDTH, HEIGHT, "3D Mandelbrot - Iteration Shells", None, None)
glfw.make_context_current(window)

# -------------------------------------------------
# OpenGL Context
# -------------------------------------------------
ctx = moderngl.create_context()
ctx.viewport = (0, 0, WIDTH, HEIGHT)  # Explicitly set viewport
print(f"OpenGL: {ctx.info['GL_VERSION']}")
print(f"Renderer: {ctx.info['GL_RENDERER']}")
print(f"Viewport: {ctx.viewport}")

# -------------------------------------------------
# Fullscreen Quad
# -------------------------------------------------
vertices = np.array([
    -1, -1,  1, -1, -1,  1,
    -1,  1,  1, -1,  1,  1,
], dtype="f4")
vbo = ctx.buffer(vertices.tobytes())

# -------------------------------------------------
# Fragment Shader - Shell-based Raymarching
# -------------------------------------------------
FRAG_SHADER = """
#version 330

uniform vec2 u_resolution;
uniform vec3 u_cam_pos;
uniform mat3 u_cam_rot;
uniform float u_scale;
uniform vec3 u_offset;
uniform int u_iter_max;
uniform float u_escape;
uniform float u_step_size;
uniform vec3 u_light_pos;
uniform float u_min_iter;
uniform float u_max_iter_float;
uniform int u_slice_mode;
uniform int u_aa_samples;
uniform vec3 u_colors[8];
uniform float u_max_dist;
uniform float u_ao_strength;
uniform float u_fog_density;
uniform float u_volumetric_light;

out vec4 fragColor;

// ------------------------------------------------------------
// Mandelbrot with smooth (float) iteration count
// ------------------------------------------------------------
bool mandelbrot_smooth(
    vec3 c, int max_iter, float escape_radius,
    out float smooth_iter, out float orbit_min
){
    float zx = 0.0, zy = 0.0, zz = 0.0;
    float esc2 = escape_radius * escape_radius;
    orbit_min = 1e20;

    for (int i = 0; i < max_iter; i++) {
        float zx2 = zx*zx, zy2 = zy*zy, zz2 = zz*zz;
        float r2 = zx2 + zy2 + zz2;

        // Store previous r2 for smooth iteration calculation
        float prev_r2 = r2;

        float nx = zx2 - zy2 - zz2 + c.x;
        float ny = 2.0*zx*zy + c.y;
        float nz = zx*zz + zy*zz + c.z;

        zx = nx; zy = ny; zz = nz;

        r2 = zx*zx + zy*zy + zz*zz;
        orbit_min = min(orbit_min, r2);

        if (r2 > esc2) {
            // Smooth iteration count: i + fractional part based on how close to escape
            // log(log(r)) gives smooth transition at escape boundary
            float log_zn = log(r2) / 2.0;
            float nu = log(log_zn / log(escape_radius)) / log(2.0);
            smooth_iter = float(i) - nu;
            return true;
        }
    }

    // For non-escaping points, return max iterations
    smooth_iter = float(max_iter);
    return false;
}

// ------------------------------------------------------------
// Distance estimator
// ------------------------------------------------------------
float mandelbrot_de(vec3 c){
    float zx = 0.0, zy = 0.0, zz = 0.0;
    float dr = 1.0;
    float esc2 = u_escape * u_escape;

    for (int i = 0; i < u_iter_max; i++) {
        float zx2 = zx*zx, zy2 = zy*zy, zz2 = zz*zz;
        float r2 = zx2 + zy2 + zz2;

        if (r2 > esc2) {
            float r = sqrt(r2);
            return 0.5 * r * log(r) / dr;
        }

        dr = 2.0 * sqrt(r2) * dr + 1.0;

        float nx = zx2 - zy2 - zz2 + c.x;
        float ny = 2.0*zx*zy + c.y;
        float nz = zx*zz + zy*zz + c.z;

        zx = nx; zy = ny; zz = nz;
    }
    return 0.0;
}

// ------------------------------------------------------------
// DE normal
// ------------------------------------------------------------
vec3 get_normal(vec3 p){
    float eps = 0.001 * u_scale;
    vec3 e = vec3(eps,0,0);

    float dx = mandelbrot_de((p+e.xyy)/u_scale+u_offset)
             - mandelbrot_de((p-e.xyy)/u_scale+u_offset);
    float dy = mandelbrot_de((p+e.yxy)/u_scale+u_offset)
             - mandelbrot_de((p-e.yxy)/u_scale+u_offset);
    float dz = mandelbrot_de((p+e.yyx)/u_scale+u_offset)
             - mandelbrot_de((p-e.yyx)/u_scale+u_offset);

    return normalize(vec3(dx,dy,dz));
}

// ------------------------------------------------------------
// Ambient Occlusion - Sample distance field along normal
// ------------------------------------------------------------
float ambient_occlusion(vec3 p, vec3 n){
    float ao = 0.0;
    float weight = 1.0;
    float dist = 0.01 * u_scale;
    
    for(int i = 0; i < 5; i++){
        vec3 sample_pos = p + n * dist;
        float d = mandelbrot_de(sample_pos/u_scale + u_offset) * u_scale;
        // If distance is less than expected, we're near a surface (occlusion)
        ao += weight * max(0.0, dist - d);
        weight *= 0.6;
        dist *= 2.0;
    }
    
    return 1.0 - clamp(ao * u_ao_strength, 0.0, 1.0);
}

// ------------------------------------------------------------
// Crisp shadow ray (using smooth iterations)
// ------------------------------------------------------------
float shadow_ray(vec3 ro, vec3 rd){
    float t = 0.05;

    for (int i = 0; i < 128; i++) {
        vec3 p = ro + rd*t;

        if (u_slice_mode > 0 && p.z > 0.0) break;

        vec3 c = p/u_scale + u_offset;

        float smooth_it;
        float trap;
        bool esc = mandelbrot_smooth(c,u_iter_max,u_escape,smooth_it,trap);

        // Use smooth comparison for shadow boundaries
        float alpha = smoothstep(u_min_iter - 0.5, u_min_iter + 0.5, smooth_it) * 
                      (1.0 - smoothstep(u_iter_max - 0.5, u_iter_max + 0.5, smooth_it));
        if (esc && alpha > 0.5)
            return 0.0;

        float h = mandelbrot_de(c) * u_scale;
        t += max(h,0.02);
        if (t > 20.0) break;
    }
    return 1.0;
}

// ------------------------------------------------------------
// Smooth palette
// ------------------------------------------------------------
vec3 iter_color(float it){
    float t = clamp(
        (it - u_min_iter) /
        (u_max_iter_float - u_min_iter),
        0.0,1.0
    );
    t = t*t*(3.0-2.0*t);

    float x = t*7.0;
    int i = int(floor(x));
    float f = fract(x);
    i = clamp(i,0,6);

    return mix(u_colors[i],u_colors[i+1],f);
}

// ------------------------------------------------------------
// Background hatch with depth
// ------------------------------------------------------------
vec3 background(vec2 uv){
    float d = length(uv);
    float hatch =
        sin(uv.x*30.0) *
        sin(uv.y*30.0);

    hatch = smoothstep(-0.4,0.4,hatch);

    vec3 col = vec3(0.08,0.09,0.12);
    col += hatch * 0.05;
    col *= 1.0 - d*0.4;

    return col;
}

// ------------------------------------------------------------
// Binary search refinement for smooth surface
// ------------------------------------------------------------
vec3 refine_surface_hit(vec3 ro, vec3 rd, float t_outside, float t_inside, float target_min) {
    float t_low = t_outside;
    float t_high = t_inside;
    vec3 hit = ro + rd * t_inside;
    
    for (int i = 0; i < 6; i++) {
        float t_mid = (t_low + t_high) * 0.5;
        vec3 p_mid = ro + rd * t_mid;
        vec3 c_mid = p_mid / u_scale + u_offset;
        
        float smooth_it;
        float trap;
        bool esc = mandelbrot_smooth(c_mid, u_iter_max, u_escape, smooth_it, trap);
        
        // Check if inside the shell
        bool inside = esc && smooth_it >= target_min && smooth_it < u_max_iter_float;
        
        if (inside) {
            t_high = t_mid;
            hit = p_mid;
        } else {
            t_low = t_mid;
        }
    }
    return hit;
}

// ------------------------------------------------------------
// Apply fog - defined before use
// ------------------------------------------------------------
vec3 apply_fog(vec3 col, float t, vec3 ro, vec3 rd) {
    float cam_dist = length(u_cam_pos);
    float fog_amount = 1.0 - exp(-u_fog_density * t * 0.1 * (1.0 + cam_dist * 0.5));
    
    vec3 fog_color = vec3(0.15, 0.18, 0.22);
    if (u_volumetric_light > 0.0) {
        vec3 to_light = normalize(u_light_pos - (ro + rd * t));
        float light_scatter = max(0.0, dot(rd, to_light));
        fog_color += vec3(0.5, 0.45, 0.35) * light_scatter * u_volumetric_light;
    }
    
    return mix(col, fog_color, fog_amount);
}

// ------------------------------------------------------------
// Raymarch with optional z-slice cut
// Simple version: in slice mode, only render where z < 0
// ------------------------------------------------------------
vec3 raymarch(vec2 offs){
    vec2 uv = (2.0*(gl_FragCoord.xy+offs)-u_resolution.xy)/u_resolution.y;
    vec3 ro = u_cam_rot * u_cam_pos;
    vec3 rd = u_cam_rot * normalize(vec3(uv,-1));
    vec3 bg = background(uv);

    float t = 0.0;
    float t_prev = 0.0;
    vec3 hit;
    float smooth_it = 0.0;
    float orbit = 0.0;
    bool hit_flag = false;
    bool prev_inside = false;
    bool hit_cut = false;

    for (int i = 0; i < 4000; i++) {
        vec3 p = ro + rd * t;
        
        // In slice mode: skip z > 0 region entirely
        if (u_slice_mode > 0 && p.z > 0.0) {
            t_prev = t;
            t += u_step_size;
            prev_inside = false;
            continue;
        }

        vec3 c = p / u_scale + u_offset;
        float smooth_it_local;
        float trap;
        bool esc = mandelbrot_smooth(c, u_iter_max, u_escape, smooth_it_local, trap);

        bool inside = esc && smooth_it_local >= u_min_iter && smooth_it_local < u_max_iter_float;
        
        if (inside && !prev_inside && i > 0) {
            hit = refine_surface_hit(ro, rd, t_prev, t, u_min_iter);
            
            // In slice mode: if surface is beyond z=0, render cut plane instead
            if (u_slice_mode > 0 && hit.z > 0.0) {
                // Surface is in cut-away region - render z=0 slice
                hit_cut = true;
                hit.z = 0.0; // Clamp to z=0 plane
            }
            
            vec3 c_hit = hit / u_scale + u_offset;
            mandelbrot_smooth(c_hit, u_iter_max, u_escape, smooth_it, orbit);
            hit_flag = true;
            break;
        }
        
        prev_inside = inside;
        t_prev = t;
        t += u_step_size;
        if (t > u_max_dist) break;
    }
    
    if (!hit_flag)
        return bg;

    vec3 col;
    float t_hit = length(hit - ro);
    
    if (hit_cut) {
        // Render flat cut surface at z=0
        vec3 base = iter_color(smooth_it);
        float trap = exp(-orbit * 2.0);
        vec3 trap_col = vec3(trap, trap * 0.5, 1.0 - trap);
        col = mix(base, trap_col, 0.3);
        
        // Simple lighting for flat cut
        vec3 n = vec3(0.0, 0.0, 1.0);
        vec3 l = normalize(u_light_pos - hit);
        float diff = max(dot(n, l), 0.0);
        col *= (0.3 + diff * 0.7);
        col *= 0.9;
    } else {
        // Render normal 3D surface
        vec3 n = get_normal(hit);
        vec3 l = normalize(u_light_pos - hit);
        float diff = max(dot(n, l), 0.0);
        float sh = shadow_ray(hit, l);
        float ao = ambient_occlusion(hit, n);

        vec3 base = iter_color(smooth_it);
        float trap = exp(-orbit * 2.0);
        vec3 trap_col = vec3(trap, trap * 0.5, 1.0 - trap);
        col = mix(base, trap_col, 0.35);
        col *= (0.1 * ao + diff * sh * 0.9);
    }

    return apply_fog(col, t_hit, ro, rd);
}

// ------------------------------------------------------------
void main(){
    vec3 col = vec3(0);

    if (u_aa_samples <= 1){
        col = raymarch(vec2(0));
    } else {
        col += raymarch(vec2(-0.25,-0.25));
        col += raymarch(vec2( 0.25,-0.25));
        col += raymarch(vec2(-0.25, 0.25));
        col += raymarch(vec2( 0.25, 0.25));
        col *= 0.25;
    }

    col = pow(col, vec3(0.4545));
    fragColor = vec4(col,1.0);
}

"""

# -------------------------------------------------
# Compile Shader
# -------------------------------------------------
try:
    prog = ctx.program(
        vertex_shader="""
        #version 330
        in vec2 in_pos;
        out vec2 v_uv;
        void main() {
            v_uv = in_pos * 0.5 + 0.5;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """,
        fragment_shader=FRAG_SHADER
    )
    print("✅ Shader compiled successfully!")
except Exception as e:
    print(f"❌ Shader error: {e}")
    exit(1)

vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

# -------------------------------------------------
# Camera / Controls Setup
# -------------------------------------------------
cam_pos = np.array([-0.300, 0.000, 1.150], dtype=np.float32)
yaw = 0.0
pitch = 0.0

# Fractal parameters
scale = 1.0         # Start at 1.0 scale
offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
iter_max = 100.0   # Max iterations to compute (float for smooth transitions)
min_iter = 6.0     # Don't show shells below this iteration (float)
escape_radius = 2.0
step_size = 0.005   # Step size (can go down to 0.0001)
max_dist = 100.0    # Max ray distance
slice_mode = 1     # Enable z-slice
aa_samples = 4     # Anti-aliasing samples (1 = off, 4 = 4x SSAA)

light_pos = np.array([1.20, 0.50, 1.15], dtype=np.float32)  # Right and above camera

# Ambient occlusion and fog settings
ao_strength = 2.0      # Ambient occlusion strength (0-5)
fog_density = 0.5      # Volumetric fog density (0-2)
volumetric_light = 1.0 # Volumetric light scattering (0-3)

# 8-color complementary palette for smooth gradient (low to high iterations)
# Custom order: Deep Blue -> Yellow -> ... -> Dark Red -> White
colors = np.array([
    [0.0, 0.2, 0.6],      # 0: Deep blue (first)
    [1.0, 0.9, 0.0],      # 1: Yellow (second)
    [0.0, 0.6, 0.7],      # 2: Cyan/Teal
    [0.2, 0.8, 0.3],      # 3: Green
    [0.4, 0.0, 0.0],      # 4: Dark red (fifth)
    [1.0, 0.4, 0.0],      # 5: Orange
    [0.15, 0.0, 0.4],     # 6: Indigo
    [1.0, 0.95, 0.9],     # 7: Warm white
], dtype=np.float32)

# Movement speeds
move_speed = 0.05
rot_speed = 0.03

# -------------------------------------------------
# Input Handling
# -------------------------------------------------
def framebuffer_size_callback(window, width, height):
    """Handle window resize"""
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = width, height
    ctx.viewport = (0, 0, width, height)
    if 'prog' in globals():
        prog["u_resolution"].value = (width, height)

glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

def print_status():
    """Print current state variables when F1 is pressed"""
    print("\n" + "=" * 60)
    print("FRACTAL STATE")
    print("=" * 60)
    print(f"  Camera Position: ({cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f})")
    print(f"  Rotation:        Yaw={yaw:.4f}, Pitch={pitch:.4f}")
    print(f"  Scale:           {scale:.4f}")
    print(f"  Step Size:       {step_size:.4f}")
    print(f"  Max Distance:    {max_dist:.1f}")
    print(f"  Iterations:      Min={min_iter:.1f}, Max={iter_max:.1f}")
    print(f"  Escape Radius:   {escape_radius:.2f}")
    print(f"  Z-Slice Mode:    {'ON' if slice_mode else 'OFF'}")
    print(f"  Anti-Aliasing:   {'4x SSAA' if aa_samples > 1 else 'OFF'}")
    print(f"  Light Position:  ({light_pos[0]:.2f}, {light_pos[1]:.2f}, {light_pos[2]:.2f})")
    print(f"  Offset:          ({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})")
    print()
    print("  Ambient Occlusion:")
    print(f"    AO Strength:       {ao_strength:.2f}")
    print()
    print("  Volumetric Fog:")
    print(f"    Fog Density:       {fog_density:.2f}")
    print(f"    Light Scattering:  {volumetric_light:.2f}")
    print("=" * 60)

def key_callback(window, key, scancode, action, mods):
    global yaw, pitch, scale, iter_max, min_iter, step_size, max_dist, slice_mode, aa_samples, light_pos, colors
    global ao_strength, fog_density, volumetric_light
    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    # F1 - Print status
    if key == glfw.KEY_F1:
        print_status()
        return

    # Rotation (A/D keys for yaw, Page Up/Down for pitch)
    if key == glfw.KEY_A: yaw += rot_speed
    if key == glfw.KEY_D: yaw -= rot_speed
    if key == glfw.KEY_PAGE_UP: pitch += rot_speed
    if key == glfw.KEY_PAGE_DOWN: pitch -= rot_speed

    # Scale
    if key == glfw.KEY_Q: scale *= 0.95
    if key == glfw.KEY_E: scale *= 1.05

    # Step size
    if key == glfw.KEY_R: step_size *= 0.9
    if key == glfw.KEY_F: step_size *= 1.1

    # Max iterations (fractional for smooth transitions)
    if key == glfw.KEY_T: iter_max += 0.5
    if key == glfw.KEY_G: iter_max = max(5.0, iter_max - 0.5)

    # Min iterations (fractional for smooth transitions)
    if key == glfw.KEY_Y: min_iter += 0.25
    if key == glfw.KEY_H: min_iter = max(0.0, min_iter - 0.25)

    # Slice toggle
    if key == glfw.KEY_SPACE:
        slice_mode = 1 - slice_mode

    # Anti-aliasing toggle
    if key == glfw.KEY_X:
        aa_samples = 1 if aa_samples > 1 else 4

    # Light
    if key == glfw.KEY_I: light_pos[1] += 0.5
    if key == glfw.KEY_K: light_pos[1] -= 0.5
    if key == glfw.KEY_J: light_pos[0] -= 0.5
    if key == glfw.KEY_L: light_pos[0] += 0.5

    # Ambient Occlusion (V/B keys)
    if key == glfw.KEY_V:
        ao_strength = min(5.0, ao_strength + 0.25)
    if key == glfw.KEY_B:
        ao_strength = max(0.0, ao_strength - 0.25)

    # Fog Density (N/M keys)
    if key == glfw.KEY_N:
        fog_density = min(2.0, fog_density + 0.1)
    if key == glfw.KEY_M:
        fog_density = max(0.0, fog_density - 0.1)

    # Volumetric Light Scattering (comma/period keys)
    if key == glfw.KEY_COMMA:
        volumetric_light = min(3.0, volumetric_light + 0.25)
    if key == glfw.KEY_PERIOD:
        volumetric_light = max(0.0, volumetric_light - 0.25)

    # Randomize colors - 8-color complementary palette
    if key == glfw.KEY_C:
        import random
        import colorsys
        base_hue = random.random()
        def hsv_to_rgb(h, s, v):
            r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
            return [r, g, b]
        # Create 8-color complementary palette
        # Start dark, progress through hue wheel, end at white
        colors[0][:] = hsv_to_rgb(base_hue, 0.9, 0.05)  # Very dark
        colors[1][:] = hsv_to_rgb(base_hue + 0.05, 0.85, 0.2)
        colors[2][:] = hsv_to_rgb(base_hue + 0.15, 0.8, 0.4)
        colors[3][:] = hsv_to_rgb(base_hue + 0.25, 0.75, 0.6)
        colors[4][:] = hsv_to_rgb(base_hue + 0.4, 0.7, 0.75)
        colors[5][:] = hsv_to_rgb(base_hue + 0.55, 0.8, 0.9)
        colors[6][:] = hsv_to_rgb(base_hue + 0.7, 0.6, 0.95)
        colors[7][:] = [1.0, 1.0, 1.0]  # Always white at top


    # Max ray distance
    if key == glfw.KEY_U: max_dist *= 1.2
    if key == glfw.KEY_J: max_dist *= 0.8

    # Reset
    if key == glfw.KEY_0:
        reset_camera()

    # Clamp
    pitch = max(-1.5, min(1.5, pitch))
    step_size = max(0.0001, min(0.1, step_size))
    max_dist = max(10.0, min(500.0, max_dist))
    min_iter = min(min_iter, iter_max - 5.0)

    # U / J for max distance (check for shift to distinguish from J for light)
    # Note: Using mods to check for shift if needed, but we handle U/J above

def reset_camera():
    global cam_pos, yaw, pitch, scale, step_size, iter_max, min_iter, slice_mode, light_pos, max_dist, colors, aa_samples
    global ao_strength, fog_density, volumetric_light
    cam_pos[:] = [-0.300, 0.000, 1.150]
    yaw = 0.0
    pitch = 0.0
    scale = 1.0
    step_size = 0.005
    max_dist = 100.0
    iter_max = 100.0
    min_iter = 6.0
    slice_mode = 1
    aa_samples = 4
    light_pos[:] = [1.20, 0.50, 1.15]  # Right and above camera
    ao_strength = 2.0
    fog_density = 0.5
    volumetric_light = 1.0
    # Reset to default cosmic sunset palette
    colors[0][:] = [0.0, 0.2, 0.6]      # Deep blue (first)
    colors[1][:] = [1.0, 0.9, 0.0]      # Yellow (second)
    colors[2][:] = [0.0, 0.6, 0.7]      # Cyan/Teal
    colors[3][:] = [0.2, 0.8, 0.3]      # Green
    colors[4][:] = [0.4, 0.0, 0.0]      # Dark red (fifth)
    colors[5][:] = [1.0, 0.4, 0.0]      # Orange
    colors[6][:] = [0.15, 0.0, 0.4]     # Indigo
    colors[7][:] = [1.0, 0.95, 0.9]     # Warm white

glfw.set_key_callback(window, key_callback)

# -------------------------------------------------
# Build Camera Rotation Matrix
# -------------------------------------------------
def get_rotation_matrix(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    return np.array([
        [cy, sy*sp, -sy*cp],
        [0, cp, sp],
        [sy, -cy*sp, cy*cp]
    ], dtype=np.float32)

# -------------------------------------------------
# Print Instructions
# -------------------------------------------------
print("\n" + "=" * 60)
print("3D Mandelbrot - Iteration Shells")
print("=" * 60)
print(f"Resolution: {WIDTH}x{HEIGHT}")
print()
print("Controls:")
print("  W/S          - Move forward/backward")
print("  Arrow Keys   - Pan view (left/right/up/down)")
print("  A/D          - Rotate left/right (yaw)")
print("  Page Up/Down - Rotate up/down (pitch)")
print("  Q / E        - Scale")
print("  R / F        - Step size")
print("  U / J        - Max ray distance")
print("  T / G        - Max iterations (+/- 0.5)")
print("  Y / H        - Min iterations (+/- 0.25)")
print("  SPACE        - Toggle Z-slice")
print("  X            - Toggle anti-aliasing (4x SSAA)")
print("  I/K/J/L      - Move light")
print("  V / B        - Ambient occlusion strength")
print("  N / M        - Fog density")
print("  , / .        - Volumetric light scattering")
print("  C            - Randomize colors")
print("  F1           - Print current state")
print("  0            - Reset")
print("  ESC          - Exit")
print()
print("Shell Structure:")
print(f"  Min Iter: {min_iter:.2f} (hollow below this)")
print(f"  Max Iter: {iter_max:.2f} (hollow at/beyond this)")
print(f"  Showing: Smooth iteration range with soft boundaries")
print()
print("Current:")
print(f"  Step: {step_size}")
print(f"  Slice: {'ON' if slice_mode else 'OFF'}")
print(f"  AA: {'4x SSAA' if aa_samples > 1 else 'OFF'}")
print()
print("Visual Effects:")
print(f"  AO Strength: {ao_strength:.2f}")
print(f"  Fog Density: {fog_density:.2f}")
print(f"  Vol. Light:  {volumetric_light:.2f}")
print("=" * 60)

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
frame = 0
fps = 0.0
last_time = glfw.get_time()

while not glfw.window_should_close(window):
    glfw.poll_events()

    # Ensure viewport matches window (fix for black bars)
    fb_width, fb_height = glfw.get_framebuffer_size(window)
    if fb_width != WIDTH or fb_height != HEIGHT:
        WIDTH, HEIGHT = fb_width, fb_height
        ctx.viewport = (0, 0, WIDTH, HEIGHT)
        prog["u_resolution"].value = (WIDTH, HEIGHT)

    # Camera movement
    rot = get_rotation_matrix(yaw, pitch)
    forward = rot[:, 2]
    right = rot[:, 0]

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        cam_pos -= forward * move_speed
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        cam_pos += forward * move_speed
    # Arrow keys for panning (move camera up/down/left/right)
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        cam_pos[1] += move_speed  # Pan up (Y axis)
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        cam_pos[1] -= move_speed  # Pan down (Y axis)
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        cam_pos += right * move_speed  # Pan left
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        cam_pos -= right * move_speed  # Pan right

    # Update uniforms
    prog["u_resolution"].value = (WIDTH, HEIGHT)
    prog["u_cam_pos"].value = tuple(cam_pos)
    prog["u_cam_rot"].write(rot.T.tobytes())
    prog["u_scale"].value = scale
    prog["u_offset"].value = tuple(offset)
    prog["u_iter_max"].value = int(iter_max)
    prog["u_min_iter"].value = float(min_iter)
    prog["u_max_iter_float"].value = iter_max
    prog["u_escape"].value = escape_radius
    prog["u_step_size"].value = step_size
    prog["u_max_dist"].value = max_dist
    prog["u_slice_mode"].value = slice_mode
    prog["u_aa_samples"].value = aa_samples
    prog["u_light_pos"].value = tuple(light_pos)
    prog["u_colors"].write(colors.tobytes())
    prog["u_ao_strength"].value = ao_strength
    prog["u_fog_density"].value = fog_density
    prog["u_volumetric_light"].value = volumetric_light

    # Render
    ctx.clear(0, 0, 0)
    vao.render(moderngl.TRIANGLES)
    glfw.swap_buffers(window)

    # FPS counter (internal only, no console output)
    frame += 1
    if frame % 60 == 0:
        current_time = glfw.get_time()
        fps = 60.0 / (current_time - last_time)
        last_time = current_time

    # Update window title with stats (every 10 frames for smoothness)
    if frame % 10 == 0:
        aa_str = 'AA' if aa_samples > 1 else 'NOAA'
        # Calculate camera distance from origin
        orig_dist = np.sqrt(np.sum(cam_pos**2))
        title = f"3D Mandelbrot | FPS:{fps:.0f} | Pos:({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f}) | Dist:{orig_dist:.2f} | Step:{step_size:.4f} | Scale:{scale:.2f} | {aa_str}"
        glfw.set_window_title(window, title)

glfw.terminate()
print("\n\nExited.")
