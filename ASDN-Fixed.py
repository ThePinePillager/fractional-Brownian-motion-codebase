import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x, y = sp.symbols('x y')        # Sympy symbols for auto-differentiation

num_calls_to_class = 0

class Simplex_Noise:
    """
    2D simplex noise implemented in Python. 
    ----------------------------------------
    Class Parameters (in order):

    frequency: How close together the valleys and peaks of the noise. \n
    amplitude: The maximum height between valleys and peaks. \n
    start_x: Where along the x axis the noise function starts. \n
    stop_x: Where along the x axis the noise function stops. \n
    start_y: Where along the y axis the noise function starts. \n
    stop_y: Where along the y axis the noise function stops. \n
    res: The resolution. An inputted value of 1 gives 1 noise value per integer squared (along both x and y). A value of 10 gives 100 noise values per integer squared. \n
    seed: The seed of the random number generator used to assign vectors to simplex points (internal). \n
    warp_func_x: A sympy function that warps each x value by an amount dictated by the function, which can take in both x and y positions. \n
    warp_func_y: A sympy function that warps each y value by an amount dictated by the function, which can take in both x and y positions. \n
    warp_noise_amplitude: Determines the amplitude of an additional noise function that causes x and y warping. \n
    warp_noise_freq: Determines frequency of an additional noise function that causes x and y warping. \n
    warp_noise_seed: Determines seed of an additional noise function that causes x and y warping. \n
    warp_noise_octaves: HEAVILY IMPACTS PERFORMANCE - Determines octave count of an additional noise function that causes x and y warping. \n
    function: Bias function for the final noise graph. Allows one to make various kinds of noise, like Ridged or Billowed noise. Works with differentiable functions in terms of x. \n
    rotation_radians: Rotates the whole function by this many radians.
    
    ----------------------------------------
    Functions:

    .get_NoiseMap() returns the basic noise graph at a given resolution. \n
    .get_GradientMap() returns the analytical gradient of the noise graph at each resolution unit. \n
    .get_NormalMap() returns the normal vector of each resolution unit of noise (good for lighting). \n
    .get_Permutation_Table() returns RNG permutation table with seed.
    
    """
    def __init__(self, frequency, amplitude, start_x, stop_x, start_y, stop_y, res, seed, 
                warp_func_x=None, warp_func_y=None, warp_noise_amplitude=0, warp_noise_freq=0.01, warp_noise_seed=0, warp_noise_octaves=0, function=None, rotation_radians=0):
        
        global num_calls_to_class
        num_calls_to_class += 1
        print("initializing call ", num_calls_to_class)
        
        self.frequency = frequency
        self.amplitude = amplitude
        self.start_x = start_x
        self.stop_x = stop_x
        self.start_y = start_y
        self.stop_y = stop_y
        self.res = res
        self.seed = seed
        self.rotation_radians = rotation_radians

        self.x, self.y = sp.symbols('x y')
        self.warp_x = warp_func_x if warp_func_x is not None else self.x
        self.warp_y = warp_func_y if warp_func_y is not None else self.y
        self.warp_func_x = sp.lambdify((self.x, self.y), self.warp_x, 'cupy')
        self.warp_func_y = sp.lambdify((self.x, self.y), self.warp_y, 'cupy')
        self.warp_noise_amp = float(warp_noise_amplitude)
        self.warp_noise_freq = float(warp_noise_freq)
        self.warp_noise_seed = warp_noise_seed

        self.function = function if function is not None else self.x
        self.lambda_function = sp.lambdify((self.x), self.function, 'cupy')

        # Individual class specific attributes

        self.c_x = cp.linspace(start_x, stop_x, res)
        self.c_y = cp.linspace(start_y, stop_y, res)
        X_init, Y_init = cp.meshgrid(self.c_x, self.c_y)

        self.X = X_init * self.frequency
        self.Y = Y_init * self.frequency
        X_rot = self.X
        Y_rot = self.Y
        self.X = X_rot * cp.cos(rotation_radians) - Y_rot * cp.sin(rotation_radians)
        self.Y = X_rot * cp.sin(rotation_radians) + Y_rot * cp.cos(rotation_radians)
        self.X_orig = self.X
        self.Y_orig = self.Y
        self.X = self.X_orig + self.warp_func_x(self.X_orig, self.Y_orig)
        self.Y = self.Y_orig + self.warp_func_y(self.X_orig, self.Y_orig)

        if self.warp_noise_amp > 0.0:
            self.Wu = Simplex_Noise(self.warp_noise_freq, self.warp_noise_amp, self.start_x, self.stop_x, self.start_y, self.stop_y, self.res, self.warp_noise_seed, self.warp_x, self.warp_y)
            self.Wv = Simplex_Noise(self.warp_noise_freq, self.warp_noise_amp, self.start_x, self.stop_x, self.start_y, self.stop_y, self.res, self.warp_noise_seed + 1, self.warp_x, self.warp_y)
            self.Wu.fBm(octaves=warp_noise_octaves)
            self.Wv.fBm(octaves=warp_noise_octaves)
            self.X += self.Wu.get_NoiseMap()
            self.Y += self.Wv.get_NoiseMap()
        self.warp_noise_octaves = warp_noise_octaves

        self.F2 = (cp.sqrt(3) - 1) / 2                                      # Skew constant
        self.G2 = self.F2 / (2 * self.F2 + 1)                               # Unskew constant
        self.s = (self.X + self.Y) * self.F2                                # Skew variable

        self.i = cp.floor(self.X + self.s)                                  # Skewed coordinates
        self.j = cp.floor(self.Y + self.s)                                  

        self.X0 = self.X - (self.i - (self.i + self.j) * self.G2)           # Unskewed first triangle coordinates
        self.Y0 = self.Y - (self.j - (self.i + self.j) * self.G2)

        self.i1 = cp.where(self.X0 > self.Y0, 1, 0)                         # Conditional triangle coordinates (splitting the skew parallelogram into 2 triangles)
        self.j1 = cp.where(self.X0 > self.Y0, 0, 1)

        self.X1 = self.X0 - self.i1 + self.G2                               # Second, Conditional triangle coordinates
        self.Y1 = self.Y0 - self.j1 + self.G2
        self.X2 = self.X0 - 1 + 2 * self.G2                                 # Third, unconditional triangle coordinates
        self.Y2 = self.Y0 - 1 + 2 * self.G2

        self.perm_array = None                                              # Permutation table
        
            # Maps
        self.unbiased_noisemap = None
        self.noisemap = None
        self.gradientmap = None
        self.normalmap = None


    def get_NoiseMap(self):                                                 # Simplex noise map
        if self.noisemap is None:
            self.noisemap = self.noise_Calc()
        return self.noisemap
    
    def get_GradientMap(self):                                              # Analytical gradient map
        if self.gradientmap is None and self.warp_noise_amp <= 0.0:
            self.gradientmap = self.gradient_Calc()
        elif self.gradientmap is None:
            self.gradientmap = self.noisy_Gradient_Calc()
        return self.gradientmap
    
    def get_NormalMap(self):                                                # 3D Normal vector map
        if self.normalmap is None:
            self.normalmap = self.normal_Vector_Map()
        return self.normalmap
    
    def get_Permutation_Table(self):
        if self.perm_array is None:
            self.perm_array = self.generate_permutation(self.seed)
        return self.perm_array
    
    def gradient_Magnitude_Map(self):                                       # Gradient magnitude map
        gradientMap = self.get_GradientMap()
        dx = gradientMap[:,:,0]
        dy = gradientMap[:,:,1]
        return cp.sqrt(cp.square(dx) + cp.square(dy))

    
    def __add__(self, other, gradient_add_toggle=False):                    # Overwriting the addition operator to make octave logic easier
        self.noisemap = self.get_NoiseMap()
        other.noisemap = other.get_NoiseMap()
        self.noisemap = self.noisemap + other.noisemap
        if gradient_add_toggle:
            self_gradientmap = self.get_GradientMap()
            other_gradientmap = other.get_GradientMap()
            dx = self_gradientmap[:,:,0] + other_gradientmap[:,:,0]
            dy = self_gradientmap[:,:,1] + other_gradientmap[:,:,1]
            self.gradientmap = cp.stack([dx, dy], axis = -1)
        return self

    def generate_permutation(self, seed):
        rng = np.random.default_rng(seed)
        p = cp.asarray(rng.permutation(256))
        return cp.concatenate([p, p])  # 512 entries
    
    def grad_map(self, i, j):                                               # Creates random vectors
        """
        i, j: 2D arrays of lattice indices
        returns: 3D array of gradients with shape (res, res, 2)
        """
        grad_array = cp.array([
            [1,1], [-1,1], [1,-1], [-1,-1],
            [1,0], [-1,0], [0,1], [0,-1]
        ])

        i_int = i.astype(int)
        j_int = j.astype(int)

        self.perm_array = self.get_Permutation_Table()
        hash_vals = self.perm_array[(i_int + self.perm_array[j_int % 256] % 256) % 256] % 8

        gradients = grad_array[hash_vals]

        return gradients
    
    def unlerp_noise(self, min=None, max=None):
        noise = self.get_NoiseMap()
        gradient = self.get_GradientMap()
        b = cp.max(noise)
        a = cp.min(noise)
        self.noisemap = (((noise - a) / (b - a)) * (max - min)) + min
        gradient[:,:,0] = (gradient[:,:,0] * (max - min)) / (b - a)
        gradient[:,:,1] = (gradient[:,:,1] * (max - min)) / (b - a)

    def noise_Calc(self):

        distance1 = cp.maximum(0, 0.5 - cp.square(self.X0) - cp.square(self.Y0))
        distance2 = cp.maximum(0, 0.5 - cp.square(self.X1) - cp.square(self.Y1))
        distance3 = cp.maximum(0, 0.5 - cp.square(self.X2) - cp.square(self.Y2))

        vec0 = cp.stack([self.X0, self.Y0], axis=-1)  # shape (res,res,2)
        vec1 = cp.stack([self.X1, self.Y1], axis=-1)
        vec2 = cp.stack([self.X2, self.Y2], axis=-1)

        contribution1 = cp.power(distance1, 4) * cp.sum(self.grad_map(self.i, self.j) * vec0, axis=-1)
        contribution2 = cp.power(distance2, 4) * cp.sum(self.grad_map(self.i + self.i1, self.j + self.j1) * vec1, axis=-1)
        contribution3 = cp.power(distance3, 4) * cp.sum(self.grad_map(self.i + 1, self.j + 1) * vec2, axis=-1)

        simplex_noise_unfiltered = self.amplitude * (contribution1 + contribution2 + contribution3)
        self.unbiased_noisemap = simplex_noise_unfiltered
        simplex_noise = self.lambda_function(simplex_noise_unfiltered)
        return simplex_noise

    def gradient_Calc(self, magnitude_switch=None):
        g1 = self.grad_map(self.i, self.j)
        g2 = self.grad_map(self.i + self.i1, self.j + self.j1)
        g3 = self.grad_map(self.i + 1, self.j + 1)

        distance1 = cp.maximum(0.5 - self.X0**2 - self.Y0**2, 0)
        distance2 = cp.maximum(0.5 - self.X1**2 - self.Y1**2, 0)
        distance3 = cp.maximum(0.5 - self.X2**2 - self.Y2**2, 0)

        dot0 = g1[:,:,0] * self.X0 + g1[:,:,1] * self.Y0
        dot1 = g2[:,:,0] * self.X1 + g2[:,:,1] * self.Y1
        dot2 = g3[:,:,0] * self.X2 + g3[:,:,1] * self.Y2

        d_df = sp.diff(self.function, self.x)

        dU_dx = sp.diff(self.warp_x, self.x)
        dU_dy = sp.diff(self.warp_x, self.y)
        dV_dx = sp.diff(self.warp_y, self.x)
        dV_dy = sp.diff(self.warp_y, self.y)

        d_df_func = sp.lambdify(self.x, d_df, 'cupy')

        dU_dx_func = sp.lambdify((self.x, self.y), dU_dx, 'cupy')
        dU_dy_func = sp.lambdify((self.x, self.y), dU_dy, 'cupy')
        dV_dx_func = sp.lambdify((self.x, self.y), dV_dx, 'cupy')
        dV_dy_func = sp.lambdify((self.x, self.y), dV_dy, 'cupy')

        d_df_vals = d_df_func(self.amplitude * self.unbiased_noisemap)
        self.bias_derivative_map = d_df_vals

        dU_dx_vals = dU_dx_func(self.X_orig, self.Y_orig)  # X, Y are cupy arrays
        dU_dy_vals = dU_dy_func(self.X_orig, self.Y_orig)
        dV_dx_vals = dV_dx_func(self.X_orig, self.Y_orig)
        dV_dy_vals = dV_dy_func(self.X_orig, self.Y_orig)

        mask1 = distance1 > 0
        mask2 = distance2 > 0
        mask3 = distance3 > 0

        dx0 = (4*(distance1**3)*(-2*self.X0)*dot0 + (distance1**4)*g1[:,:,0]) * mask1
        dy0 = (4*(distance1**3)*(-2*self.Y0)*dot0 + (distance1**4)*g1[:,:,1]) * mask1

        dx1 = (4*(distance2**3)*(-2*self.X1)*dot1 + (distance2**4)*g2[:,:,0]) * mask2
        dy1 = (4*(distance2**3)*(-2*self.Y1)*dot1 + (distance2**4)*g2[:,:,1]) * mask2

        dx2 = (4*(distance3**3)*(-2*self.X2)*dot2 + (distance3**4)*g3[:,:,0]) * mask3
        dy2 = (4*(distance3**3)*(-2*self.Y2)*dot2 + (distance3**4)*g3[:,:,1]) * mask3

        dNx = dx0 + dx1 + dx2
        dNy = dy0 + dy1 + dy2

        dNx_corr = (1 - self.F2) * dNx - self.F2 * dNy
        dNy_corr = -self.F2 * dNx + (1 - self.F2) * dNy

        dNx, dNy = dNx_corr, dNy_corr

        dz_dx = dNx * (1 + dU_dx_vals) + dNy * dV_dx_vals
        dz_dy = dNy * (1 + dV_dy_vals) + dNx * dU_dy_vals
        dz_dx *= d_df_vals
        dz_dy *= d_df_vals


        if magnitude_switch:
            magnitude = cp.sqrt(cp.square(dz_dx) + cp.square(dz_dy))
            return magnitude

        return cp.stack([dz_dx, dz_dy], axis = -1)
    
    def noisy_Gradient_Calc(self):

        g1 = self.grad_map(self.i, self.j)
        g2 = self.grad_map(self.i + self.i1, self.j + self.j1)
        g3 = self.grad_map(self.i + 1, self.j + 1)

        distance1 = cp.maximum(0.5 - self.X0**2 - self.Y0**2, 0)
        distance2 = cp.maximum(0.5 - self.X1**2 - self.Y1**2, 0)
        distance3 = cp.maximum(0.5 - self.X2**2 - self.Y2**2, 0)

        dot0 = g1[:,:,0] * self.X0 + g1[:,:,1] * self.Y0
        dot1 = g2[:,:,0] * self.X1 + g2[:,:,1] * self.Y1
        dot2 = g3[:,:,0] * self.X2 + g3[:,:,1] * self.Y2

        d_df = sp.diff(self.function, self.x)

        dU_dx = sp.diff(self.warp_x, self.x)
        dU_dy = sp.diff(self.warp_x, self.y)
        dV_dx = sp.diff(self.warp_y, self.x)
        dV_dy = sp.diff(self.warp_y, self.y)

        d_df_func = sp.lambdify(self.x, d_df, 'cupy')

        dU_dx_func = sp.lambdify((self.x, self.y), dU_dx, 'cupy')
        dU_dy_func = sp.lambdify((self.x, self.y), dU_dy, 'cupy')
        dV_dx_func = sp.lambdify((self.x, self.y), dV_dx, 'cupy')
        dV_dy_func = sp.lambdify((self.x, self.y), dV_dy, 'cupy')

        d_df_vals = d_df_func(self.amplitude * self.unbiased_noisemap)

        dU_dx_vals = dU_dx_func(self.X_orig, self.Y_orig) 
        dU_dy_vals = dU_dy_func(self.X_orig, self.Y_orig)
        dV_dx_vals = dV_dx_func(self.X_orig, self.Y_orig)
        dV_dy_vals = dV_dy_func(self.X_orig, self.Y_orig)

        dx0 = 4*(distance1**3)*(-2*self.X0)*dot0 + (distance1**4)*g1[:,:,0]
        dy0 = 4*(distance1**3)*(-2*self.Y0)*dot0 + (distance1**4)*g1[:,:,1]

        dx1 = 4*(distance2**3)*(-2*self.X1)*dot1 + (distance2**4)*g2[:,:,0]
        dy1 = 4*(distance2**3)*(-2*self.Y1)*dot1 + (distance2**4)*g2[:,:,1]

        dx2 = 4*(distance3**3)*(-2*self.X2)*dot2 + (distance3**4)*g3[:,:,0]
        dy2 = 4*(distance3**3)*(-2*self.Y2)*dot2 + (distance3**4)*g3[:,:,1]

        dNx = dx0 + dx1 + dx2
        dNy = dy0 + dy1 + dy2

        dNx_corr = (1 - self.F2) * dNx - self.F2 * dNy
        dNy_corr = -self.F2 * dNx + (1 - self.F2) * dNy

        dNx, dNy = dNx_corr, dNy_corr

        Wu_gradient = self.Wu.get_GradientMap()
        Wv_gradient = self.Wv.get_GradientMap()
        Wu_dx = Wu_gradient[:,:,0]
        Wu_dy = Wu_gradient[:,:,1]
        Wv_dx = Wv_gradient[:,:,0]
        Wv_dy = Wv_gradient[:,:,1]

        dNx_dWx = dNx * (Wu_dx + 1) + dNy * Wv_dx
        dNy_dWy = dNy * Wv_dy + dNx * (Wu_dy + 1)

        dNx_dWx_dx = dNx_dWx * (dU_dx_vals + 1) + dNy_dWy * dV_dx_vals
        dNy_dWy_dy = dNy_dWy * (dV_dy_vals + 1) + dNy_dWy * dU_dy_vals

        dNx_dWx_dx *= d_df_vals
        dNy_dWy_dy *= d_df_vals

        return cp.stack([dNx_dWx_dx, dNy_dWy_dy], axis = -1)

    def normal_Vector_Map(self):
        gradientMap = self.get_GradientMap()
        dx = gradientMap[:,:,0]
        dy = gradientMap[:,:,1]
        x_vec = cp.stack([cp.ones_like(dx), cp.zeros_like(dx), dx], axis=-1)
        y_vec = cp.stack([cp.zeros_like(dy), cp.ones_like(dy), dy], axis=-1)
        normal_vector = cp.cross(x_vec, y_vec)
        normal_vector = normal_vector / cp.linalg.norm(normal_vector, axis=-1, keepdims=True)
        self.normalmap = normal_vector

    
    def fBm(self, octaves=4, lacunarity=2, persistence=0.5, octave_amplitude_array = None, fBm_bias_function=None):      # Fractional Brownian Motion for multiple octaves
        """
            :param octaves: HEAVILY IMPACTS PERFORMANCE - The number of smaller noise sublayers of the function. \n
            :param lacunarity: Multiplicative increase in frequency over octave layers. \n
            :param persistence: Multiplicative decrease in amplitude over octave layers. \n
            :param octave_amplitude_array: Allows one to submit weights for each octave that dictates how much individual octaves impact the final function value. Must be an int array with as many elements as there are octaves.  \n
            :param fBm_bias_function: Allows the bias function to be different than the initial bias function (the function parameter). \n
        """
        if fBm_bias_function is None:
            fBm_bias_function = self.function
        if octave_amplitude_array is None:
            octave_amplitude_array = [1] * octaves
        if len(octave_amplitude_array) != octaves:
            raise ValueError("Weight Vector must have length equal to octaves")
        for n in range(1, octaves + 1):
            if octave_amplitude_array[n-1] == 0:
                continue
            S_freq = self.frequency * (lacunarity ** n)
            S_ampl = self.amplitude * (persistence ** n) * octave_amplitude_array[n - 1]
            S_warp_freq = self.warp_noise_freq * (lacunarity ** n)
            S_warp_ampl = self.warp_noise_amp * (persistence ** n) * octave_amplitude_array[n - 1]
            S = Simplex_Noise(S_freq, S_ampl, self.start_x, self.stop_x, self.start_y, self.stop_y, self.res, self.seed + n, self.warp_x, self.warp_y, warp_noise_amplitude=S_warp_ampl, warp_noise_freq=S_warp_freq, warp_noise_seed=self.warp_noise_seed, 
                    warp_noise_octaves=self.warp_noise_octaves, function=fBm_bias_function, rotation_radians=self.rotation_radians + 2.7)
            self.__add__(S,gradient_add_toggle=True)
        return self    

def basic_fBm():
    """Prebuilt basic noise function"""
    S = Simplex_Noise(1/400, 100, -500, 500, -500, 500, 1000, 58008, x, rotation_radians=1)
    S.fBm(True, 12)
    return S

def basic_Ridged_fBm():
    """Prebuilt ridge noise function"""
    S = Simplex_Noise(1/400, 100, -500, 500, -500, 500, 1000, 58008, sp.sqrt(x ** 2) * -1, rotation_radians=1)
    S.fBm(True, 12)
    return S

def basic_Billowed_fBm():
    """Prebuilt ridge noise function"""
    S = Simplex_Noise(1/400, 100, -500, 500, -500, 500, 1000, 58008, sp.sqrt(x ** 2), rotation_radians=1)
    S.fBm(True, 12)
    return S

def ridged_Noise_Function():
    """Place this as the 'function' parameter in the Simplex class for ridged noise"""
    return sp.sqrt(x ** 2 + 0.0001) * -1

def billowed_Noise_Function():
    """Place this as the 'function' parameter in the Simplex class for billowed noise"""
    return sp.sqrt(x ** 2 + 0.0001)

def lambertian_Lighting(S, lighting_vector):
    if len(lighting_vector) != 3:
        raise ValueError("lighting vector must have 3 elements, 1 for each dimension.")
    lx, ly, lz = lighting_vector
    S.normal_Vector_Map()
    cp_light_vector = cp.array([lx, ly, lz], dtype=cp.float32)
    cp_light_vector /= cp.linalg.norm(cp_light_vector, axis=-1, keepdims=True)
    light_map = cp.tensordot(S.normalmap, cp_light_vector, axes=([-1], [0]))
    L = cp.maximum(light_map, 0)
    min = L.min()
    max = L.max()
    if min != max:
        L = (L - L.min()) / (L.max() - L.min())
    return L

def shadow_map(heightmap, light_vector, step_size=1.0, max_steps=50, smoothness=0.5):
    """
    Compute soft shadows based on distance to first occlusion.
    
    heightmap: (N,N) CuPy array
    light_vector: (lx, ly, lz)
    step_size: step along ray
    max_steps: max steps along light ray
    smoothness: controls softness of shadow edges, 0 to 1
    """
    lx, ly, lz = light_vector 
    norm = cp.sqrt(lx**2 + ly**2 + lz**2)
    lx, ly, lz = lx / norm, ly / norm, lz / norm

    N, M = heightmap.shape
    visibility = cp.ones_like(heightmap, dtype=cp.float32)
    first_occlusion = cp.full_like(heightmap, max_steps + 1, dtype=cp.float32)

    X, Y = cp.meshgrid(cp.arange(M), cp.arange(N))
    x_pos = X.astype(cp.float32)
    y_pos = Y.astype(cp.float32)
    h_line = heightmap.copy()

    dx = lx / lz * step_size
    dy = ly / lz * step_size

    for step in range(1, max_steps + 1):
        x_pos += dx
        y_pos += dy
        h_line += step_size

        xi = cp.clip(cp.floor(x_pos).astype(cp.int32), 0, M - 1)
        yi = cp.clip(cp.floor(y_pos).astype(cp.int32), 0, N - 1)
        h_ray = heightmap[yi, xi]

        delta = h_line - h_ray
        occluded = delta <= 0

        first_occlusion = cp.where((occluded) & (first_occlusion == max_steps + 1), step, first_occlusion)

    vis = first_occlusion / (max_steps + 1)
    vis = cp.clip(vis, 0, 1)

    visibility = vis**2 * (3 - 2*vis) # Smoothstep for soft edges

    return visibility

def Simple_Wrapper(seed, function, octaves, allow_light=True):
    res = 2             # Resolution. Heavily impacts performance
    freq = 400
    start_x = -500      # Graph size
    stop_x = 500
    start_y = -500
    stop_y = 500
    amplitude = 100

    area = (stop_x - start_x) * (stop_y - start_y)
    resolution = int(area ** 0.5) * int(res)

    noise_func = Simplex_Noise(1 / freq, amplitude, start_x, stop_x, start_y, stop_y, resolution, seed, function=function, rotation_radians=1)
    noise_func.fBm(octaves, lacunarity=1.5, persistence=0.75, fBm_bias_function=function)
    noise_func.unlerp_noise(-1000, 1000)

    Nmp = noise_func.get_NoiseMap().get()
    Gmp = noise_func.gradient_Magnitude_Map().get()

    plt.imshow(Nmp, extent=(start_x, stop_x, start_y, stop_y), origin="lower", cmap="viridis")
    plt.colorbar(label="f(x, y)")
    plt.title("Example Function")
    plt.title('Simple Wrapper Noise Map')

    if allow_light:
        lighting_vector = [1, 1, 1]
        L = (shadow_map(noise_func.noisemap, lighting_vector, 1 / res, 100 * res, 0.5 / res) + lambertian_Lighting(noise_func, lighting_vector) / 5 + 0.2).get()
        L = (L - cp.min(L)) / (cp.max(L) - cp.min(L))

        plt.imshow(L, extent=(start_x, stop_x, start_y, stop_y), origin='lower',cmap='Greys_r', alpha=0.5)  # alpha controls transparency, making the gradient overlay visible but not overpowering
        plt.colorbar(label='Light Map')
        plt.title('Simple Wrapper Noise Map with Lighting')

    plt.show()


def Complex_Wrapper (seed, function, octaves, fBm_bias_func, lacunarity=1.5, persistence=0.75, warp_func_x=None, warp_func_y=None, 
                    warp_noise_amplitude=0, warp_noise_freq=0.01, warp_noise_seed=0, warp_noise_octaves=0, rotation_radians=0, unlerp_strength=1000, light_vector = [1, 1, 1]):
    """
    :param seed: The seed of the random number generator used to assign vectors to simplex points (internal). \n
    :param function: Bias function for the final noise graph. Allows one to make various kinds of noise, like Ridged or Billowed noise. Works with differentiable functions in terms of x. \n
    :param octaves: HEAVILY IMPACTS PERFORMANCE - The number of smaller noise sublayers of the function. \n
    :param fBm_bias_func: Allows the bias function to be different than the initial bias function (the function parameter). \n
    :param lacunarity: Multiplicative increase in frequency over octave layers. \n
    :param persistence: Multiplicative decrease in amplitude over octave layers. \n
    :param warp_func_x: A sympy function that warps each x value by an amount dictated by the function, which can take in both x and y positions. \n
    :param warp_func_y: A sympy function that warps each y value by an amount dictated by the function, which can take in both x and y positions. \n
    :param warp_noise_amplitude: Determines the amplitude of an additional noise function that causes x and y warping. \n
    :param warp_noise_freq: Determines frequency of an additional noise function that causes x and y warping. \n
    :param warp_noise_seed: Determines seed of an additional noise function that causes x and y warping. \n
    :param warp_noise_octaves: HEAVILY IMPACTS PERFORMANCE - Determines octave count of an additional noise function that causes x and y warping. \n
    :param rotation_radians: Rotates the whole function by this many radians.
    :param unlerp_strength: The range the noise function will adhere to. This will affect the lighting engine.
    :param light_vector: The source of the light. The vector "points to where the sun is." Works with negative z values too.
    """
    
    res = 2             # Resolution. Heavily impacts performance
    freq = 400
    start_x = -500      # Graph size
    stop_x = 500
    start_y = -500
    stop_y = 500
    amplitude = 100

    area = (stop_x - start_x) * (stop_y - start_y)
    resolution = int(area ** 0.5) * int(res)

    noise_func = Simplex_Noise(1 / freq, amplitude, start_x, stop_x, start_y, stop_y, resolution, seed, warp_func_x, warp_func_y, 
                                warp_noise_amplitude, warp_noise_freq, warp_noise_seed, warp_noise_octaves, function, rotation_radians)
    noise_func.fBm(octaves, lacunarity, persistence, fBm_bias_function=fBm_bias_func)
    noise_func.unlerp_noise(-unlerp_strength, unlerp_strength)

    Nmp = noise_func.get_NoiseMap().get()
    Gmp = noise_func.gradient_Magnitude_Map().get()
    plt.imshow(Nmp, extent=(start_x, stop_x, start_y, stop_y), origin="lower", cmap="terrain")
    plt.colorbar(label="f(x, y)")
    plt.title("Example Function")
    plt.title('Complex Wrapper Noise Map')

    if light_vector != [0, 0, 0]:
        L = (shadow_map(noise_func.noisemap, light_vector, 1 / res, 100 * res, 0.5 / res) + lambertian_Lighting(noise_func, light_vector) / 5 + 0.2).get()
        L = (L - cp.min(L)) / (cp.max(L) - cp.min(L))

        plt.imshow(L, extent=(start_x, stop_x, start_y, stop_y), origin='lower',cmap='Greys_r', alpha=0.5)  # alpha controls transparency, making the gradient overlay visible but not overpowering
        plt.colorbar(label='Light Map')
        plt.title('Complex Wrapper Noise Map with Lighting')

    plt.show()


if __name__ == "__main__":

    seed = 58008                      # The RNG seed for noise generation
    octaves = 0                    # The number of sublayers called in the noise function
    bias_function = sp.sin(x)              # Can be any Sympy compatible differentiable function in terms of x
                                    # Billowed noise: sp.sqrt(x**2 + 0.000001)
                                    # Ridged noise: sp.sqrt(x**2 + 0.000001) * -1
                                    # Something cool: sp.sin(x**3 - 5 * x**2 + x)
    lighting = True                 # Toggles the lighting engine only for the simple wrapper. Analytical differentiation makes this faster and more accurate.

    """<<--------------------------------------------------->>"""

    # Extra values for the more complicated wrapper. Explanation found in wrapper documentation. To disable lighting engine, set light vector to [0, 0, 0].

    fBm_bias_func = sp.sqrt(x**2 + 0.000001)
    lacunarity = 1.5
    persistence = 0.75
    warp_func_x = 4*x + y              #sp.sin(x + 2 * y)
    warp_func_y = sp.cos(y) + y             #sp.sqrt(y**2 + 0.00001) * -1
    warp_noise_amplitude = 0
    warp_noise_frequency = 10
    warp_noise_seed = 50
    warp_noise_octaves = 1
    rotation_radians = 0
    noise_function_range = 1000             # -val to val
    light_vector = [1, 1, 1]             # When z is large relative to x and y, very few shadows are cast, meaning all pixels have a similar light level. 
                                            # Matplotlib decides to color it dark though, and I don't know how to fix it right now.


simple = False       # Toggles simple wrapper vs complex wrapper, so they don't need to be separately commented out everytime you want to switch.


if simple == True:
    Simple_Wrapper(seed, bias_function, octaves, lighting)
else:
    Complex_Wrapper(seed, bias_function, octaves, fBm_bias_func, lacunarity, persistence, warp_func_x, warp_func_y, warp_noise_amplitude, warp_noise_frequency, 
                    warp_noise_seed, warp_noise_octaves, rotation_radians, noise_function_range, light_vector)
