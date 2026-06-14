import numpy as np
import os
import pickle


######################################################
##      WINDING NUMBER BY PARALLEL TRANSPORT        ##
######################################################
def compromise(a, b):
    """Given a pair of n-dimensional vectors `a`, `b`, this function
    finds the (n x 2) matrix `Y` whose columns are orthonormal and such that
    the Frobenius norm of `(Y - [a b])` is minimized. This is the "best orthonormal
    approximation to `[a b]`

    Args:
        a (Numpy array): First vector
        b (Numpy array): Second vector

    Returns:
        list: A two-element list of orthonormal vectors
    """
    X = np.array([a,b])
    u, _, vh = np.linalg.svd(X, full_matrices=False)
    Y = u @ vh
    return [*Y]

def compute_winding(structure, smoothing=20):
    """
    Computes the normal bundle framing and cumulative winding number
    for a protein structure
    
    each protein structure stored in the `structures` dictionary.
    The backbone, normal bundle, "flattened" curve (projection to the
    normal bundle), and cumulative winding number are stored to
    the respective member variables: `backbones`, `normal_bundles`, `flattened`,
    and `winding`.

    Parameters
    ----------
    structure: ndarray(n, 3)
        Coordinates of the residue sequence in 3D
    smoothing: int (optional): 
        Amount of smoothing to apply when computing the backbone curve. Defaults to 20.

    Returns
    -------
    {
        winding: ndarray(n)
            The winding number at each residue,
        backbone: ndarray(n, 3)
            The smoothed backbone structure
        normal_bundle: ndarray(n, 2, 3)
            The normal bundle at each residue
        flattened: ndarray(n, 2)
            Residues projected onto the normal bundle
    }
    """
    from scipy.ndimage import gaussian_filter1d as gf1d
    X = gf1d(structure,  sigma=1, axis=0) # smoothed out structure
    Y = gf1d(X, sigma=smoothing, axis=0) # backbone
    dY = gf1d(X, sigma=smoothing, axis=0, order=1) # tangent of backbone
    dZ = dY / np.sqrt(np.sum(dY ** 2, axis=1))[:, np.newaxis] # normalized tangent

    # parallel transport along backbone
    # V[i] is an orthonormal basis for the orthogonal complement of dZ[i]
    V = np.zeros((len(dZ), 2, 3)) 
    V[0] = np.random.rand(2, 3)
    for i, z in enumerate(dZ):
        if i: V[i] = V[i-1]

        # remove projection onto z, the current tangent vector,
        # then enforce orthonormality
        V[i] -= np.outer(V[i] @ z, z)
        V[i] = compromise(*V[i])

    s = np.array([x @ v for x, v in zip(X - Y, V[:,0,:])])
    c = np.array([x @ w for x, w in zip(X - Y, V[:,1,:])])

    summand = np.arctan((c[:-1] * s[1:] - s[:-1] * c[1:]) / (s[:-1] * s[1:] + c[:-1] * c[1:]))

    winding = np.cumsum(summand) / (2 * np.pi)
    winding *= np.sign(winding[-1] - winding[0])

    return dict(
        winding = winding,
        backbone=Y,
        normal_bundle=V,
        flattened = np.array([s, c]).T
    )


######################################################
##             B FACTOR PERIOD LOCATIONS            ##
######################################################
def compute_bfactor_periods(bfactor, period=25):
    """
    Given the b-factor, or the displacement of the atoms
    from their mean, compute locations of the period boundaries.
    This serves as a baseline for other periodicity analysis
    techniques
    
    Parameters
    ----------
    bfactor: ndarray(N)
        The b-factor at each residue
    period: float
        Approximate period of each residue, used for tuning
        the bandpass filter
    
    Returns
    -------
    locations: list of int
        Locations of periods
    """
    ## Step 1: Bandpass filter the b factor to hone in on periodicities
    from scipy import signal
    sos = signal.butter(10, [0.5/period, 2/period], 'bandpass', output='sos')
    bff = signal.sosfiltfilt(sos, bfactor)
    ## Step 2: Find and return all local maxes
    idx = np.arange(1, bff.size-1)
    idx = idx[(bff[idx] > bff[idx-1])*(bff[idx] > bff[idx+1])]
    return idx


######################################################
##          LAPLACIAN CIRCULAR COORDINATES          ##
######################################################

def get_csm(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    Parameters
    ----------
    X: ndarray(M, d)
        A point cloud with M points in d dimensions
    Y: ndarray(N, d)
        A point cloud with N points in d dimensions
    Returns
    -------
    D: ndarray(M, N)
        An MxN Euclidean cross-similarity matrix
    """
    if len(X.shape) == 1:
        X = X[:, None]
    if len(Y.shape) == 1:
        Y = Y[:, None]
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def csm_to_binary(D, kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix, using partitions instead of
    nearest neighbors for speed
    Parameters
    ----------
    D: ndarray(M, N)
        M x N cross-similarity matrix
    kappa: float
        If kappa = 0, take all neighbors
        If kappa < 1 it is the fraction of neighbors to consider
        Otherwise kappa is the number of neighbors to consider
    Returns
    -------
    B: ndarray(M, N)
        MxN binary cross-similarity matrix
    """
    from scipy import sparse
    N = D.shape[0]
    M = D.shape[1]
    if kappa == 0:
        return np.ones_like(D)
    elif kappa < 1:
        NNeighbs = int(np.round(kappa*M))
    else:
        NNeighbs = kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M), dtype=np.uint8)
    return ret.toarray()

def csm_to_binary_mutual(D, kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix, where 
    an entry (i, j) is a 1 if and only if it is within the both the nearest
    neighbor set of i and the nearest neighbor set of j
    Parameters
    ----------
    D: ndarray(M, N)
        M x N cross-similarity matrix
    kappa: float
        If kappa = 0, take all neighbors
        If kappa < 1 it is the fraction of mutual neighbors to consider
        Otherwise kappa is the number of mutual neighbors to consider
    Returns
    -------
    B: ndarray(M, N)
        MxN binary cross-similarity matrix
    """
    return csm_to_binary(D, kappa)*(csm_to_binary(D.T, kappa).T)

def sliding_window(D, win):
    """
    Average down diagonals to simulate the effect of a sliding window
    
    Parameters
    ----------
    D: ndarray(N, N)
        Distance matrix
    win: int
        Sliding window length

    Returns
    -------
    ndarray(N-win+1, N-win+1)
        Sliding window distance matrix
    """
    N = D.shape[0]
    D_stack = np.zeros((N-win+1, N-win+1))
    for i in range(0, win):
        D_stack += D[i:i+N-win+1, i:i+N-win+1]
    for i in range(N-win+1):
        D_stack[i, i] = 0
    return D_stack

def get_unweighted_laplacian_eigs_dense(W):
    """
    Get eigenvectors of the unweighted Laplacian
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere

    Returns
    -------
    v: ndarray(N, N)
        A matrix of eigenvectors
    """
    from scipy import sparse
    import numpy.linalg as linalg
    D = sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    L = D - W
    try:
        _, v = linalg.eigh(L)
    except:
        return np.zeros_like(W)
    return v

def get_most_circular_pair(v, period, hop=10):
    """
    Compute the most circular pair of adjacent eigenvectors
    by computing persistent homology of alpha complexes in
    small blocks
    
    v: ndarray(n-period+1, n-period+1)
        Eigenvectors of graph Laplacian
    period: int
        Approximate period
    hop: int
        Hop length between blocks in which circularity is tested
    
    Returns
    -------
    scores: ndarray(n-period)
        Scores of each pair of eigenvalues
    """
    from gudhi import AlphaComplex
    block = period*2
    all_scores = []
    for i in range(v.shape[1]-1):
        scores = 0
        vi = v[:, i:i+2]
        for idx in range(0, v.shape[0]-block+1, hop):
            ac = AlphaComplex(points=vi[idx:idx+block, :])
            stree = ac.create_simplex_tree()
            stree.compute_persistence()
            I = stree.persistence_intervals_in_dimension(1)
            if I.size > 0:
                scores += np.max(I[:, 1]-I[:, 0])
        all_scores.append(scores)
    return np.array(all_scores)

def compute_laplacian_circular_coords(structure, sigma=1, period=25, kappa=50):
    """
    Parameters
    ----------
    structure: ndarray(n, 3)
        Coordinates of the residue sequence in 3D
    sigma: float (optional)
        Amount by which to smooth curve when computing velocity.
        Default 1
    period: int (optional)
        The amount by which to do a sliding window, or roughly the expected
        period of each solenoidal region
        Default 25
    kappa: float (optional)
        Nearest neighbor proportion to take when computing graph Laplacian
        If kappa = 0, take all neighbors
        If kappa < 1 it is the fraction of mutual neighbors to consider
        Otherwise kappa is the number of mutual neighbors to consider
        Default 50
    
    Returns
    -------    
    {
        D: ndarray(n-period+1, n-period+1)
            Sliding window distance matrix
        B: ndarray(n-period+1, n-period+1)
            Binarized sliding window distance matrix
        v: ndarray(n-period+1, n-period+1)
            Eigenvectors of graph Laplacian
        theta: ndarray(n-period+1)
            Estimated circular coordinates
        idx: int
            Index in v of the first pair of eigenvector to use
    }
    """
    from scipy.ndimage import gaussian_filter1d as gf1d
    X = gf1d(structure, sigma=sigma, order=1, axis=0) # smoothed out structure
    D = get_csm(X, X)
    D = sliding_window(D, period)
    B = csm_to_binary_mutual(D, kappa)
    v = get_unweighted_laplacian_eigs_dense(1-B)
    scores = get_most_circular_pair(v[:, 0:3], period)
    idx = np.argmax(scores)
    theta = np.arctan2(v[:, idx+1], v[:, idx])
    theta = np.unwrap(theta)/(2*np.pi)
    if theta[-1] < 0:
        theta *= -1
    return dict(D=D, B=B, v=v, theta=theta, idx=idx)


def compute_lrr_winding_laplacian(structure, breakpoints, period=25):
    """
    Compute LRR windings of a structure within each LRR region, 
    as determined by breakpoints

    Parameters
    ---------- 
    structure: ndarray(n, 3)
        Coordinates of the residue sequence in 3D
    breakpoints: ndarray(int)
        Residue locations of the breakpoints
    period: int (optional)
        Approximate period of each winding
        Default 25
    
    Returns
    -------
    lwinding: ndarray(n)
        Winding number in annotation regions
    """
    n = structure.shape[0]
    lwinding = np.zeros(n)
    last_theta = 0
    a = breakpoints[0]
    b = breakpoints[-1]
    res = compute_laplacian_circular_coords(structure[a:b+period-1, :], period=period)
    theta = res["theta"] + last_theta
    last_theta = theta[-1]
    if theta.size < b-a: # Pad if too close to the end
        theta = np.concatenate((theta, (b-a-theta.size)*[last_theta]))
    lwinding[a:b] = theta
    lwinding[b:] = last_theta
    return lwinding


######################################################
##                REGION ANNOTATION                ##
######################################################

def median_slope(data, small = 100, big = 250):
    """Computes the distribution of slopes of secant lines
    over a data curve (e.g. cumulative winding number)

    Args:
        data (Numpy array): Curve on which to compute slopes of secant lines
        small (int): Lower bound for run. Defaults to 100.
        big (int): Upper bound for run. Defaults to 250. 

    Returns:
        list: A two-element list consisting of the median slope, and `scores`, 
        the histogram of secant line slopes
    """
    slopes = []
    weights = []
    for i in range(len(data) - small):
        for j in range(i + small, min(i + big, len(data))):
            s = (data[j]-data[i])/(j-i)
            slopes.append(s)
            reg = data[i:j] - s * np.arange(i,j)
            reg -= np.mean(reg)
            # weights.append(np.sqrt(j - i))
            weights.append((j - i) / (1 + np.sum(reg ** 2)))
    
    n_bins = int(np.sqrt(len(slopes)))
    scores = [0 for i in range(n_bins)]
    a = min(slopes)
    b = max(slopes) + 0.01

    for s, weight in zip(slopes, weights):
        bin_index = int(n_bins * (s - a) / (b - a))
        scores[bin_index] += weight

    return a + (np.argmax(scores) / n_bins) * (b - a), scores

def multi_loss(winding, breakpoints, slope, penalties):
    """Computes loss associated with a particular piecewise-linear
    regression of `winding`.
    Args:
        winding (_type_): Cumulative winding number (signal to be regressed)
        breakpoints (_type_): Breakpoint locations
        slope (_type_): Slope inferred by median-secant-line computation
        penalties (_type_): Weights allowed deviations in the coiling and non-coiling regions
    Returns:
        float: Loss value
    """
    cost = 0
    boundaries = [0] + list(breakpoints.astype('int')) + [len(winding)]

    for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        b = min(b, winding.size)
        if b > a:
            linear = (i % 2) * slope * (np.arange(a, b) - (a + b - 1) / 2)
            cost += penalties[i % 2] * np.sum((winding[a:b] - linear - np.mean(winding[a:b])) ** 2)

    return cost

def compute_regression(winding, n_breakpoints=2, penalties=[1, 1.5], learning_rate=0.01, iterations=10000, initial_guess=[]):
    """
    Computes piecewise-linear regressions (constant - slope = m - constant) over
    all cumulative winding curves stored in the `winding` dictionary. Writes the parameters
    of these regressions to the `parameters` and `slopes` dictionaries.

    Parameters
    ----------
    winding: ndarray(n)
        The winding number at each residue
    n_breakpoints: int (optional)
        How many breakpoints to use
    penalties: list[float, float] (optional)
        Two-element list describing the relative penalties, in the loss function, of deviation. 
        The first component refers to the non-coiling regions; the second to the coiling region.
        Defaults to [1, 1.5].
    learning_rate: float (optional)
        Scalar for gradient descent in parameter optimization. Defaults to 0.01.
        iterations (int, optional): Iterations of gradient descent. Defaults to 10000.
    iterations: int (optional)
        Number of iterations in gradient descent.  Defaults to 10000
    initial_guess: list of float (optional)
        An initial guess of the breakpoint locations.  If not specified, initial conditions
        will be taken as equally spaced.
    
    Returns
    -------
    {
        slope: float
            Estimated slope in each winding segment
        breakpoints: ndarray(int)
            Residue locations of the breakpoints
        loss: float
            Final loss from the regression
    }
    """
    n = len(winding)

    if len(initial_guess) > 0:
        breakpoints = np.array(initial_guess)
    else:
        # best-guess initialization
        breakpoints = n * (1 + np.arange(n_breakpoints)) / (n_breakpoints + 1)
    gradient = np.zeros(n_breakpoints)
    delta = [*np.identity(n_breakpoints)]

    m, _ = median_slope(winding)

    for _ in range(iterations):
        present = multi_loss(winding, breakpoints, m, penalties)
        # Compute a finite difference approximation of the gradient
        gradient = np.array([multi_loss(winding, breakpoints + d, m, penalties) - present for d in delta])
        breakpoints = breakpoints - learning_rate * gradient
        # Safeguards
        breakpoints[breakpoints > winding.size] = winding.size
        breakpoints = np.sort(breakpoints)

    # if breakpoints[-1] > 0.9 * n:
    #     breakpoints[-1] = len(winding)

    return dict(
        slope=m,
        breakpoints=np.array(breakpoints, dtype=int),
        loss=present
    )

def compute_split_coil(winding, learning_rate=0.001, epochs=1000, batch_size=32):
    # Initialize parameters
    t0 = len(winding) // 2  # Initial guess for the transition point
    n = len(winding)
    t = np.arange(n)
    
    def piecewise_function(t, t0, m0, m1):
        left_segment = m0 * t[t <= t0]
        right_segment = m0 * t0 + m1 * (t[t > t0] - t0)
        return np.concatenate([left_segment, right_segment])
    
    def compute_loss(winding, t, t0, m0, m1):
        predictions = piecewise_function(t, t0, m0, m1)
        return np.sum((winding - predictions) ** 2)
    
    def compute_gradients(winding, t, t0, m0, m1):
        predictions = piecewise_function(t, t0, m0, m1)
        residuals = predictions - winding
        
        gradient_m0 = np.sum(residuals * np.where(t <= t0, t, t0))
        gradient_m1 = np.sum(residuals * np.where(t > t0, t - t0, 0))
        gradient_t0 = np.sum(residuals * ((m1 - m0) * np.where(t > t0, 1, 0)))
        
        return gradient_m0, gradient_m1, gradient_t0
    
    # Adaptive learning rate
    base_lr = learning_rate
    for epoch in range(epochs):
        # Compute median slopes for left and right segments
        m0, _ = median_slope(winding[:int(t0) + 1])
        m1, _ = median_slope(winding[int(t0) + 1:])
        
        # Compute gradients
        gradient_m0, gradient_m1, gradient_t0 = compute_gradients(winding, t, t0, m0, m1)
        
        # Update parameters
        t0 -= learning_rate * gradient_t0 / n
        t0 = np.clip(t0, 1, n-2)  # Ensure t0 remains within valid range
        
        # Adjust learning rate
        if epoch % 100 == 0:
            learning_rate = base_lr / (1 + epoch / 100)
        
        # Compute and print loss for monitoring
        loss = compute_loss(winding, t, t0, m0, m1)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}, t0: {t0:.2f}, m0: {m0:.4f}, m1: {m1:.4f}')
    
    return m0, m1, t0




######################################################
##              STATISTICS ON RESULTS               ##
######################################################

def compute_lrr_discrepancy_arithmetic(winding, locs, a, b):
    """
    Compute the discrepancy of a winding number when assuming
    certain period boundary locations
    
    Parameters
    ----------
    winding: ndarray(n)
        The winding number at each residue]
    locs: list of int
        Period boundary locations
    a: int
        Left endpoint of LRR region, inclusive
    b: int
        One beyond right endpoint of LRR region
    
    Returns
    -------
    discrepancy: float
        Root mean discrepancy over all period locations
    """
    locs = np.array([l for l in locs if l >= a and l < b], dtype=int)
    lrr_heights = winding[locs]

    k = len(lrr_heights)
    u = 1 + np.zeros(k)
    v = np.arange(0.0, k)
    v -= (u @ v) / (u @ u) * u
    z = (lrr_heights @ u) / (u @ u) * u + (lrr_heights @ v) / (v @ v) * v
    diff = np.sign(np.mean(z[1:] - z[:-1]))
    projected = diff * v + (z @ u) / (u @ u) * u

    return np.sqrt(np.mean((projected - lrr_heights) ** 2))


def compute_lrr_discrepancy(winding, locs, a, b):
    """
    Compute the discrepancy of a winding number when assuming
    certain period boundary locations
    
    Parameters
    ----------
    winding: ndarray(n)
        The winding number at each residue]
    locs: list of int
        Period boundary locations
    a: int
        Left endpoint of LRR region, inclusive
    b: int
        One beyond right endpoint of LRR region
    
    Returns
    -------
    discrepancy: float
        Root mean discrepancy over all period locations
    """
    locs = np.array([l for l in locs if l >= a and l < b], dtype=int)
    lrr_heights = winding[locs]
    return np.sqrt(np.sum((lrr_heights[1:] - lrr_heights[:-1] - 1) ** 2))


def compute_lrr_std(winding, breakpoints, slope):
    """
    Compute the standard deviation of the difference
    between the linear estimates and the actual winding
    number over all LRR segments

    Parameters
    ----------
    winding: ndarray(n)
        The winding number at each residue
    breakpoints: ndarray(int)
        Residue locations of the breakpoints
    slope: float
        Estimated slope in each winding segment
    """
    winding_seg = np.array([])
    y_seg = np.array([])
    for i in range(0, len(breakpoints), 2):
        if i+1 < len(breakpoints):
            [a, b] = breakpoints[i:i+2]
            winding_seg = np.concatenate((winding_seg, winding[a:b]))
            linear = slope * (np.arange(a, b) - (a + b - 1) / 2)
            y = linear + np.mean(winding[a:b])
            y_seg = np.concatenate((y_seg, y))
    return np.std(winding_seg-y_seg)



######################################################
##                 BATCH PROCESSOR                  ##
######################################################

class Analyzer:
    def __init__(self):
        self.structures = {}
        self.bfactors = {}
        self.backbones = {}
        self.normal_bundles = {}
        self.flattened = {}
        self.windings = {}
        self.slopes = {}
        self.breakpoints = {}
        self.lwindings = {}
        self.losses = {}
        self.stds = {}

    def load_structures(self, structures):
        """Updates internal dictionary of three-dimensional protein structures,
        loaded, e.g., by a Loader object.

        Args:
            structures (dict): Dictionary of protein structures
        """
        self.structures.update(structures)
    
    def load_bfactors(self, bfactors):
        """Updates internal dictionary of b-factors,
        loaded, e.g., by a Loader object.

        Args:
            structures (dict): Dictionary of b-factors
        """
        self.bfactors.update(bfactors)

    def compute_windings(self, smoothing=20, progress=True):
        """Computes the normal bundle framing and cumulative winding number
        for each protein structure stored in the `structures` dictionary.
        The backbone, normal bundle, "flattened" curve (projection to the
        normal bundle), and cumulative winding number are stored to
        the respective member variables: `backbones`, `normal_bundles`, `flattened`,
        and `winding`.

        Parameters
        ----------
        smoothing: int (optional): 
            Amount of smoothing to apply when computing the backbone curve. Defaults to 20.
        progress: bool (optional):
            Whether to show a progress bar (default True)
        """
        from tqdm import tqdm
        for key, structure in (tqdm(self.structures.items(), desc = 'Computing windings') if progress else self.structures.items()):
            res = compute_winding(structure, smoothing=smoothing)
            self.windings[key] = res["winding"]
            self.backbones[key] = res["backbone"]
            self.normal_bundles[key] = res["normal_bundle"]
            self.flattened[key] = res["flattened"]
            

    def compute_regressions(self, penalties=[1, 1.5], learning_rate=0.01, iterations=10000, std_cutoff=1, progress=True):
        """Computes piecewise-linear regressions (constant - slope = m - constant) over
        all cumulative winding curves stored in the `winding` dictionary. 
        Start by assuming 2 breakpoints, and then if the standard deviation exceeds
        Writes the breakpoints
        of these regressions to the `breakpoints` and `slopes` dictionaries.

        Parameters
        ----------
        penalties: list[float, float] (optional)
            Two-element list describing the relative penalties, in the loss function, of deviation. 
            The first component refers to the non-coiling regions; the second to the coiling region.
            Defaults to [1, 1.5].
        learning_rate: float (optional)
            Scalar for gradient descent in parameter optimization. Defaults to 0.01.
            iterations (int, optional): Iterations of gradient descent. Defaults to 10000.
        iterations: int (optional)
            Number of iterations in gradient descent.  Defaults to 10000
        std_cutoff: float (optional)
            The standard deviation amount beyond which to subdivide LRR region    
        progress: bool (optional):
            Whether to show a progress bar (default True)
        """
        from tqdm import tqdm
        for key, winding in (tqdm(self.windings.items(), desc = 'Computing regressions') if progress else self.windings.items()):
            ## Step 1: Compute breakpoints
            res = compute_regression(winding, n_breakpoints=2, penalties=penalties, learning_rate=learning_rate, iterations=iterations)
            std = compute_lrr_std(winding, res["breakpoints"], res["slope"])
            self.stds[key] = std
            [a, b] = res["breakpoints"]
            if std > std_cutoff:
                breakpoints = [a, a + (b-a)/2, a + (b-a)/2 + 1, b]
                res = compute_regression(winding, n_breakpoints=4, initial_guess=breakpoints, penalties=penalties, learning_rate=learning_rate, iterations=iterations)
            self.slopes[key] = res["slope"]
            self.breakpoints[key] = res["breakpoints"]
            self.losses[key] = res["loss"]
            
    def compute_lrr_windings_laplacian(self, period=25, progress=True):
        """
        Parameters
        ---------- 
        period: int
            Approximate period of each winding
        progress: bool (optional):
            Whether to show a progress bar (default True)
        """
        from tqdm import tqdm
        for key in (tqdm(self.structures, desc = 'Computing Laplacian windings') if progress else self.windings.items()):
            self.lwindings[key] = compute_lrr_winding_laplacian(self.structures[key], self.breakpoints[key], period=period)

    def cache_geometry(self, directory, prefix = ''):
        with open(os.path.join(directory, prefix + 'backbones.pickle'), 'wb') as handle:
            pickle.dump(self.windings, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(directory, prefix + 'normal_bundles.pickle'), 'wb') as handle:
            pickle.dump(self.normal_bundles, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(directory, prefix + 'flattened.pickle'), 'wb') as handle:
            pickle.dump(self.flattened, handle, protocol = pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, prefix + 'windings.pickle'), 'wb') as handle:
            pickle.dump(self.windings, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def retrieve_geometry(self, directory, prefix = ''):
        with open(os.path.join(directory, prefix + 'backbones.pickle'), 'rb') as handle:
            self.backbones.update(pickle.load(handle))
        
        with open(os.path.join(directory, prefix + 'normal_bundles.pickle'), 'rb') as handle:
            self.normal_bundles.update(pickle.load(handle))

        with open(os.path.join(directory, prefix + 'flattened.pickle'), 'rb') as handle:
            self.flattened.update(pickle.load(handle))
        
        with open(os.path.join(directory, prefix + 'windings.pickle'), 'rb') as handle:
            self.windings.update(pickle.load(handle))

    def cache_regressions(self, directory, prefix = ''):
        with open(os.path.join(directory, prefix + 'slopes.pickle'), 'wb') as handle:
            pickle.dump(self.slopes, handle, protocol = pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(directory, prefix + 'regressions.pickle'), 'wb') as handle:
            pickle.dump(self.regressions, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def retrieve_regressions(self, directory, prefix = ''):
        with open(os.path.join(directory, prefix + 'slopes.pickle'), 'rb') as handle:
            self.slopes.update(pickle.load(handle))

        with open(os.path.join(directory, prefix + 'regressions.pickle'), 'rb') as handle:
            self.regressions.update(pickle.load(handle))
