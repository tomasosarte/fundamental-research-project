
import torch
import numpy as np

class Rn:

    name = 'R^2'
    n = 2
    e = torch.tensor([0., 0.], dtype=torch.float32)


class H:
    # Label for the group
    name = 'O(2)'
    # Dimension of the sub-group H
    n = 2  # Each element consists of 2 parameter
    # The identify element
    e = torch.tensor([0., 1.], dtype=torch.float32) # \theta, m --> \theta = 0 and m = 1
    # Haar measure
    haar_meas = None

    ## Essential for constructing the group G = R^n \rtimes H
    # Define how H acts transitively on R^n
    ## TODO: So far just for multiples of 90 degrees. No interpolation required
    def left_representation_on_Rn(h, fx):
        if not False in (h == H.e):
            return fx
        else:
            # Fist rotate, then mirror
            Lgfx = fx
            # Rotate:
            Lgfx = torch.rot90(Lgfx, k=int(torch.round((1. / (np.pi / 2.) * h[0])).item()), dims=[-2, -1])
            # Mirror
            if h[-1] == -1:
                Lgfx = torch.flip(Lgfx, dims=[-2])      # mirrorings on the x axis.
            # Return Lgfx
            return Lgfx

    def left_representation_on_G(h, fx):
        if not False in (h == H.e):
            return fx
        else:
            shape = fx.shape
            # First rotate then mirror
            Lgfx = H.left_representation_on_Rn(h, fx)
            # Now permute the axes
            Lgfx = torch.reshape(Lgfx, [shape[0], shape[1], 2, 4, shape[-2], shape[-1]])
            # First permutation on rotate, then on mirror
            # They rotate in opposite directions
            if h[0] != 0:
                Lgfx[:, :, 0, :, :, :] = torch.roll(Lgfx[:, :, 0, :, :, :], shifts=int(torch.round((1. / (np.pi / 2.) * h[0])).item()), dims=2)
                Lgfx[:, :, 1, :, :, :] = torch.roll(Lgfx[:, :, 1, :, :, :], shifts=-int(torch.round((1. / (np.pi / 2.) * h[0])).item()), dims=2)
            if h[-1] == -1:
                # Then on the m axis
                Lgfx = torch.roll(Lgfx, shifts=1, dims=2)
            # Reshape
            Lgfx = torch.reshape(Lgfx, shape)
            # Return Lgfx
            return Lgfx

    ## Essential in the group convolutions
    # Define the determinant (of the matrix representation) of the group element
    def absdet(h):
        return 1.   # It actually needs to return the absdet.

    ## Grid class
    class grid_global:  # For a global grid
        # Should a least contain:
        #	N     - specifies the number of grid points
        #	scale - specifies the (approximate) distance between points, this will be used to scale the B-splines
        # 	grid  - the actual grid
        #	args  - such that we always know how the grid was constructed
        # Construct the grid
        def __init__(self, N):
            # This rembembers the arguments used to construct the grid (this is to make it a bit more future proof, you may want to define a grid using specific parameters and later in the code construct a similar grid with the same parameters, but with N changed for example)
            self.args = locals().copy()
            self.args.pop('self')
            # Store N
            self.N = N
            N_m = int(N / 2)  # Recall that the mirroring group consists of 2 elements
            # Define the scale (the spacing between points)
            self.scale = [2 * np.pi / N_m]
            # Generate the grid
            if self.N == 0:
                h_list = torch.tensor([], dtype=torch.float32)
            else:
                h_list = np.array([np.linspace(0, 2 * np.pi - 2 * np.pi / N_m, N_m)], dtype=np.float32).transpose()
                h_list_m = np.stack(((np.concatenate((h_list, h_list), axis=0)).squeeze(),                                                  # 2 times rotation
                                     np.concatenate((np.ones(N_m, dtype=np.float32), -1 * np.ones(N_m, dtype=np.float32))).transpose()),    # [1, ..., -1, ...]
                                    axis=1)
                h_list_m = torch.from_numpy(h_list_m)
            self.grid = h_list_m
            # -------------------
            # Update haar measure
            H.haar_meas = 2 * (2 * np.pi / N_m)


## The derived group G = R^n \rtimes H.
# The above translation group and the defined group H together define the group G
# The following is automatically constructed and should not be changed unless
# you may have some speed improvements, or you may want to add some functions such
# as the logarithmic and exponential map.
# A group element in G should always be a vector of length Rn.n + H.n
class G:
    # Label for the group G
    name = 'E(2)'
    # Dimension of the group G
    n = Rn.n + H.n
    # The identity element
    e = torch.cat([Rn.e, H.e], dim=-1)

    # Function that returns the classes for R^n and H
    @staticmethod
    def Rn_H():
        return Rn, H
