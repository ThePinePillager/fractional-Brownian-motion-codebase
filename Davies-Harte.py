import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#step 0: get valid inputs
#inputs are H, T, N
# H is in (0,1)
# T is most commonly 1 for 1 year(in financial simulations using fBm)
# N is how many substeps in T, the higher the number the more points the graph will have

def retrieve_H():
    H = float(input('Enter Hurst Parameter: '))
    while not 0 <= H <= 1:
        print('Hurst Parameter not in range 0-1')
        H = float(input('Enter Hurst Parameter: '))
    return H
 
#actual implementating in finished code
N = int(input('Enter Time Steps(N): '))
T = int(input('Enter Total Time(T): '))
H = retrieve_H()

#for testing defaults for debugging code
# N = 10000
# T = 1
# H = 0.5

#step 1: time grid
#t_k = kT/N, where T is total time, N is time steps

t = np.linspace(0,T,N+1)

#step 2: define autocovariance function
#gamma(k) = 1/2(abs(k+1)^2H - 2abs(k)^2H + abs(k-1)^2H)
#gamma(k) = 0 for k > N

gamma = lambda k: 0.5*(abs(k+1)**(2*H) -2*abs(k)**(2*H) +abs(k-1)**(2*H))

#step 3: form circulant vector of autocovariances
#c = [gamma(0),gamma(1),[...],gamma(N-1), gamma(N), gamma(N-1),[...],gamma(2),gamma(1)]

c = np.concatenate([np.array([gamma(k) for k in range(N+1)]), np.array([gamma(k) for k in range(N-1,0,-1)])])

#step 4: compute real positive eigenvalues usind FFT
#lambda = Re(FFT(c))

l = np.fft.fft(c).real

if not np.allclose(np.fft.fft(c).imag, 0, atol = 1e-10):
    raise ValueError('FFT has significant imaginary components, check input vector.')

if np.any(l<0):
    raise ValueError('Negative eigenvalues found -- invalid circulant embedding')

#step 5: generate complex gaussian vector
# Z ~ C^2N, Z_0 ~ N(0,l_0), Z_N ~ N(0,l_N)
# for 1<= k < N; Z_k = sqrt(l_k/2)(X_k + iY_k), where K_k,Y_k ~ N(0,1)
# define Z_2N-k = conjugate Z_k

# Generate standard normals
Z = np.zeros(2*N, dtype=np.complex128)

# Real parts
Z[0] = np.sqrt(l[0]) * np.random.normal()
Z[N] = np.sqrt(l[N]) * np.random.normal()

# For 1 ≤ k < N
X = np.random.normal(0, 1, N - 1)
Y = np.random.normal(0, 1, N - 1)

for k in range(1, N):
    Z[k] = np.sqrt(l[k] / 2) * (X[k-1] + 1j * Y[k-1])
    Z[2*N - k] = np.conj(Z[k])

#step 6: inverse FFT to get fractional Gaussian noise
#fGn = Re(InvFFT(Z))
#note: impementation of InvFFT in python needs scaling to get to usable terms
#scale by sqrt(2N) and (T/N) ^ H

fGn = np.fft.ifft(Z).real[:N] * (T / N) ** H * np.sqrt(2*N)

#step 7: obtain fBm
#B_h(t_k) = sum from j = 0 to k of fGn_j

fbm = np.concatenate([np.array([0]), np.cumsum(fGn)])

plt.figure(figsize=(10, 6))
plt.plot(t, fbm, linewidth=2, color='#2E86C1', alpha=0.8)
plt.title(f'Fractional Brownian Motion Sample Path: H = {H}', fontsize=12)
plt.xlabel('Time', fontsize=10)
plt.ylabel('B(t)', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

