"""
Helper functions adapted from: https://github.com/facebookresearch/PoincareMaps

"""

#from model.poincare_loss import poincare_translation
from ContrastivePoincareMaps.model.poincare_loss import poincare_translation
#plt.switch_backend('agg')
from sklearn.decomposition import PCA

from .visualize import *
from ContrastivePoincareMaps.helpers.visualize import *
#from model import *
import seaborn as sns; sns.set()
import torch as th


class PoincareMaps:
    def __init__(self, coordinates, cpalette=None):
        self.coordinates = coordinates
        self.distances = None       
        self.radius = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
        self.iroot = np.argmin(self.radius)
        self.labels_pos = None
        if cpalette is None:
            self.colors_palette = colors_palette
        else:
            self.colors_palette = cpalette
        
    def find_iroot(self, labels, head_name):
        head_idx = np.where(labels == head_name)[0]
        if len(head_idx) > 1:
            D = self.distances[head_idx, :][head_idx]
            self.iroot = head_idx[np.argmin(D.sum(axis=1))]
        else:
            self.iroot = head_idx[0]            

    def get_distances(self):
        self.distances = poincare_distance(th.DoubleTensor(self.coordinates)).numpy()

    def get_angles(self):
        self.angles = angular_sep(th.DoubleTensor(self.coordinates)).numpy()

    def rotate(self):
        self.coordinates_rotated = poincare_translation(-self.coordinates[self.iroot, :], self.coordinates)     

    def plot(self, pm_type='ori', labels=None, 
        labels_name='labels', print_labels=None, labels_text=None,
        labels_order=None, coldict=None, file_name=None, title_name=None, alpha=1.0,
        zoom=None, show=True, d1=4.5, d2=4.0, fs=9, ms=20, bbox=(1.3, 0.7), u=None, v=None, leg=True, ft='pdf'):                            
        if pm_type == 'ori':
            coordinates = self.coordinates
        
        elif pm_type == 'rot':
            coordinates = self.coordinates_rotated

        if labels_order is None:
            labels_order = np.unique(labels)

        if not (zoom is None):
            if zoom == 1:
                coordinates = np.array(linear_scale(coordinates))
            else:           
                radius = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
                idx_zoom = np.where(radius <= 1/zoom)[0]
                coordinates = coordinates[idx_zoom, :]
                coordinates = np.array(linear_scale(coordinates))    
                coordinates[np.isnan(coordinates)] = 0
                labels = labels[idx_zoom]

        
        self.labels_pos = plot_poincare_disc(coordinates, title_name=title_name, 
            print_labels=print_labels, labels_text=labels_text,
            labels=labels, labels_name=labels_name, labels_order=labels_order, labels_pos = self.labels_pos,
                       file_name=file_name, coldict=coldict, u=u, v=v, alpha=alpha,
                       d1=d1, d2=d2, fs=fs, ms=ms, col_palette=self.colors_palette, bbox=bbox, leg=leg, ft=ft)

def get_geodesic_parameters(u, v, eps=1e-10):
    if all(u) == 0:
        u = np.array([eps, eps])
    if all(v) == 0:
        v = np.array([eps, eps])

    nu = u[0]**2 + u[1]**2
    nv = v[0]**2 + v[1]**2
    a = (u[1]*nv - v[1]*nu + u[1] - v[1]) / (u[0]*v[1] - u[1]*v[0])
    b = (v[0]*nu - u[0]*nv + v[0] - u[0]) / (u[0]*v[1] - u[1]*v[0])
    return a, b

def intermediates(p1, p2, nb_points=20):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return np.array([[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)])
    
    
def poincare_linspace(u, v, n_points=175):            
    if (np.sum(u**2) == 0):
        x = np.linspace(u[0], v[0], num=n_points)
        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] = x 
        if v[0] != 0:
            k = v[1]/v[0]
            interpolated[:, 1] = k*interpolated[:, 0]
        else:
            interpolated[:, 1] = np.linspace(0, v[1], num=n_points)
    elif (np.sum(v**2) == 0):
        x = np.linspace(u[0], v[0], num=n_points)
        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] = x 
        if u[0] != 0:
            k = u[1]/u[0]
            interpolated[:, 1] = k*interpolated[:, 0]
        else:
            interpolated[:, 1] = np.linspace(0, u[1], num=n_points)
                
    else:
        a, b = get_geodesic_parameters(u, v)

        x = np.linspace(u[0], v[0], num=n_points)

        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] =x 

        r = a**2/4 + b**2/4 - 1
        y_1 = -b/2 + np.sqrt(r - (x+a/2)**2)
        y_2 = -b/2 - np.sqrt(r - (x+a/2)**2)

        if max(x**2 + y_1**2) > 1:
            interpolated[:, 1] = y_2 
        elif max(x**2 + y_2**2) > 1:
            interpolated[:, 1] = y_1
        elif (np.mean(y_1) <= max(u[1], v[1])) and (np.mean(y_1) >= min(u[1], v[1])):
            interpolated[:, 1] = y_1
        else:
            interpolated[:, 1] = y_2

    return interpolated

def read_data(fin, with_labels=True, normalize=False, n_pca=20):
    """
    Reads a dataset in CSV format from the ones in datasets/
    """
    df = pd.read_csv(fin + '.csv', sep=',')
    n = len(df.columns)

    if with_labels:
        x = np.double(df.values[:, 0:n - 1])
        labels = df.values[:, (n - 1)]
        labels = labels.astype(str)
        colnames = df.columns[0:n - 1]
    else:
        x = np.double(df.values)
        labels = ['unknown'] * np.size(x, 0)
        colnames = df.columns

    if "Hematopoiesis" in fin:
        x = np.double(x[:,0:-1])
        colnames = colnames[0:-1]

    n = len(colnames)

    idx = np.where(np.std(x, axis=0) != 0)[0]
    x = x[:, idx]

    if normalize:
        s = np.std(x, axis=0)
        s[s == 0] = 1
        x = (x - np.mean(x, axis=0)) / s

    if n_pca:
        if n_pca == 1:
            n_pca = n

        nc = min(n_pca, n)
        pca = PCA(n_components=nc)
        x = pca.fit_transform(x)
        colnames = [f'PC{i+1}' for i in range(n_pca)]

    labels = np.array([str(s) for s in labels])

    return x, labels, colnames

def euclidean_distance(x):
    th.set_default_tensor_type('torch.DoubleTensor')
    # print('computing euclidean distance...')
    nx = x.size(0)
    x = x.contiguous()
    
    x = x.view(nx, -1)

    norm_x = th.sum(x ** 2, 1, keepdim=True).t()
    ones_x = th.ones(nx, 1)

    xTx = th.mm(ones_x, norm_x)
    xTy = th.mm(x, x.t())
    
    d = (xTx.t() + xTx - 2 * xTy)
    d[d < 0] = 0

    return d

def poincare_distance(x):
    # print('computing poincare distance...')
    eps = 1e-5
    boundary = 1 - eps
    
    nx = x.size(0)
    x = x.contiguous()
    x = x.view(nx, -1)
    
    norm_x = th.sum(x ** 2, 1, keepdim=True)
    sqdist = euclidean_distance(x) * 2    
    squnorm = 1 - th.clamp(norm_x, 0, boundary)

    x = (sqdist / th.mm(squnorm, squnorm.t())) + 1
    z = th.sqrt(th.pow(x, 2) - 1)
    
    return th.log(x + z)

def angular_sep(x, eps: float = 1e-8):

    # flatten in case x has extra dims
    x = x.view(x.size(0), -1).contiguous()

    # pairwise dot products
    dot = th.mm(x, x.t())

    # norms as a column vector, then outer product
    norms = th.linalg.norm(x, dim=1, keepdim=True)  # (N,1)
    denom = th.mm(norms, norms.t()) + eps          # (N,N)

    cos = th.clamp(dot / denom, -1.0 + eps, 1.0 - eps)
    return th.acos(cos)

def linear_scale(embeddings):
    # embeddings = np.transpose(embeddings)
    sqnorm = np.sum(embeddings ** 2, axis=1, keepdims=True)    
    dist = np.arccosh(1 + 2 * sqnorm / (1 - sqnorm))
    dist = np.sqrt(dist)
    dist /= dist.max()

    sqnorm[sqnorm==0] = 1
    embeddings = dist * embeddings / np.sqrt(sqnorm)
    # embeddings[abs(embeddings) > 1] = 1
    # embeddings = embeddings / np.sum(embeddings ** 2, axis=1, keepdims=True).max()
    return embeddings