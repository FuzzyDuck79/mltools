#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class SOM(object):
    """Self-Organising Map (SOM) training and analysis on a rectangular grid.

    Attributes
    ----------
    n_rows : int
        Number of rows of neurons.
    n_cols : int
        Number of columns of neurons.
    neighbourhood : str, optional
        Form of neighbourhood function on SOM. `gaussian` by default,
        with options `exponential`, `linear`, `gaussian`.
    bmus : ndarray
        1D array of Best Matching Units (BMUs) for each training instance.
    wts: ndarray
        SOM weights (codebook). Rows correspond to neurons, columns to
        weights in feature space.
    """
    
    
    def __init__(self, n_rows, n_cols, neighbourhood='gaussian'):
        """Class constructor.
        
        Parameters
        ----------
        n_rows : int
            Number of rows in SOM.
        n_cols : int
            Number of columns in SOM.
        neighbourhood : str, optional
            Form of neighbourhood function on SOM. Options available:
            `gaussian` (default), `exponential`, `linear`, `bubble`.
        """
        
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.neighbourhood = neighbourhood
        self.bmus = None
        self.wts = None
        
        # Calculate distance**2 matrix for neuron array
        ixs = np.arange(n_rows*n_cols)       
        rows, cols = ixs % n_cols, ixs // n_cols
        self.d2mat = (rows[:,None] - rows[None,:])**2 + (cols[:,None] - cols[None,:])**2

    
    def calc_BMUs(self, X):
        """Calculate Best-Matching Units (BMUs) for training data array X.
        
        Parameters
        ----------
            X : ndarray
                Training data, with rows as instances, columns as features.
        """

        return ((X[:,None] - self.wts)**2).sum(axis=2).argmin(axis=1)

    
    def make_kernels(self):
        """Generate kernels for all epochs. 
        
        User-defined kernel weights at the maximum radius Rmax for the
        first epoch, and unit radius R1 at the final epoch.
        """ 
        
        # Define kernels based on neighbourhood function
        if self.neighbourhood == 'bubble':
            sigs = np.linspace(self.Rmax, 0.5, self.n_epochs)
            self.kernels = np.where(np.sqrt(self.d2mat)[None,:,:]<=sigs[:,None,None], 1, 0)
        elif self.neighbourhood == 'linear':
            sig_Rmax = (1 - self.h0_Rmax)/self.Rmax
            sig_R1 = (1 - self.hN_R1)
            sigs = np.linspace(sig_Rmax, sig_R1, self.n_epochs)
            self.kernels = np.clip(1 - np.sqrt(self.d2mat[None,:,:])*sigs[:,None,None], 0, 1)
        elif self.neighbourhood == 'exponential':
            sig_Rmax = -self.Rmax/np.log(self.h0_Rmax)
            sig_R1 = -1/np.log(self.hN_R1)
            sigs = np.linspace(sig_Rmax, sig_R1, self.n_epochs)
            self.kernels = np.exp(-np.sqrt(self.d2mat[None,:,:])/sigs[:,None,None])
        elif self.neighbourhood == 'gaussian':
            sig_Rmax = np.sqrt(-self.Rmax**2/np.log(self.h0_Rmax))
            sig_R1 = np.sqrt(-1/np.log(self.hN_R1))
            sigs = np.linspace(sig_Rmax, sig_R1, self.n_epochs)
            self.kernels = np.exp(-self.d2mat[None,:,:]/sigs[:,None,None])
        else:
            print('Invalid neighbourhood')
            return None

    
    def fit(self, X, labels=None, n_epochs=10, h0_Rmax=0.5, hN_R1=0.01, Rmax=None, 
            initial='random'):
        """Train SOM on input data array X using the batch algorithm.
        
        Input array X should be in the standard format, i.e.
        rows (axis 0) are instances, columns (axis 1) are features.

        Parameters
        ----------
            X : ndarray
                Training data, with rows as instances, columns as features.
            labels : ndarray or list
                Labels of training data for supervised training.
            n_epochs : int, optional
                Number of training epochs.
            h0_Rmax : float, optional
                Initial kernel weight at Rmax.
            hN_R1 : float, optional
                Final kernel weight at unit radius (distance to nearest neighbours).
            Rmax : float, optional
                Maximum radius of neighbourhood kernel.
            initial : str, optional
                Weights initialisation method. One of `random` or `pca`.
        """
        
        self.n_epochs = n_epochs
        self.h0_Rmax = h0_Rmax
        self.hN_R1 = hN_R1
        self.Rmax = Rmax
        
        # Set initial neighbourhood radius to be the largest distance in the SOM
        if Rmax is None:
            self.Rmax = np.sqrt((self.n_cols-1)**2 + (self.n_rows-1)**2)
        else:
            self.Rmax = Rmax
        
        # Initialise SOM weights as a random array or using PCA
        n_samp, n_feat = X.shape
        if initial == 'random':
            self.wts = np.random.random(size=(self.n_rows*self.n_cols, n_feat))
        elif initial == 'pca':
            X_mean = X.mean(axis=0)
            X_zm = X - X_mean
            covmat = (X_zm.T @ X_zm)/n_samp
            eigvals, eigvecs = np.linalg.eigh(covmat)

            # Variance explained by PCs beyond the first two
            resid_variance = eigvals[:-2].sum()
            
            # Generate Gaussian noise to make up variance
            noise = np.random.normal(loc=0, scale=np.sqrt(resid_variance), size=(self.n_rows, self.n_cols, n_feat))
            
            # Ranges of row PCs (for EOF1) and column PCs (for EOF2) over SOM
            row_facs = np.linspace(-eigvals[-1], eigvals[-1], self.n_rows)
            col_facs = np.linspace(-eigvals[-2], eigvals[-2], self.n_cols)
            col_facs, row_facs = np.meshgrid(col_facs, row_facs)
            
            self.wts = ((row_facs[:,:,None] * eigvecs[:,-1]) + 
                        (col_facs[:,:,None] * eigvecs[:,-2]) + 
                        noise + X_mean
                       ).reshape((self.n_rows*self.n_cols, -1))
        else:
            print('initial must be random or pca')
            return None

        # Define kernels based on neighbourhood function
        self.make_kernels()

        for i in range(n_epochs):
            # Calculate BMUs for all training vectors
            bmus = self.calc_BMUs(X)
            
            # Calculate numerator (BMU kernel-weighted sum of training data)
            num = (X[:,None] * self.kernels[i][bmus][:,:,None]).sum(axis=0)
            
            # Calculate denominator (sum of BMU weights for training data) and update weights
            denom = self.kernels[i][bmus].sum(axis=0)
            self.wts = num/denom[:,None]
            
        # Update BMUs
        self.bmus = self.calc_BMUs(X)
        
        # Calculate distance matrix of neurons in feature space
        self.dmat = np.sqrt(((self.wts[:,None] - self.wts)**2).sum(axis=2))
        
        # Supervised learning, if labels supplied
        labelmap = np.empty(self.n_cols*self.n_rows)
        labelmap[:] = np.nan
        
        if labels is not None:
            labels = np.array(labels)
            
            # Define mapping from neurons to classes using majority vote
            for i in range(self.n_cols*self.n_rows):
                labels_i = labels[self.bmus==i]
                values, counts = np.unique(labels_i, return_counts=True)
                if counts.size > 0:
                    labelmap[i] = values[np.argmax(counts)]
            
            # If any values in labelmap are nan, backfill with the closest non-nan neuron
            # Fill all nan neurons columns in feature distance matrix with large number
            dmat_nonan = np.where(np.isnan(labelmap), 1e9, self.dmat)
            
            # Get column indices of nearest neurons for all rows
            nearest_nonan = dmat_nonan.argsort(axis=1)[:,0]
            
            # Backfill nan values
            labelmap = np.where(np.isnan(labelmap), labelmap[nearest_nonan], labelmap)

        self.labelmap = labelmap


    def predict(self, X):
        """Predict classes of data.
        
        Makes use of supervised classification if labelled 
        training data are available.
        
        Parameters
        ----------
            X : ndarray
                Training data, with rows as instances, columns as features.
                
        Returns
        -------
            predicted : ndarray
                Predicted labels.
        """
        
        if self.wts is None:
            print('Train SOM before classifying')
            return None
        
        bmus = self.calc_BMUs(X)

        if np.isnan(self.labelmap).all():
            return bmus
        else:
            return np.array([self.labelmap[bmu] for bmu in bmus])


    def umatrix(self, figsize=(6,6)):
        """Plot U-matrix."""
   
        # Create empty U-matrix of appropriate dimensions
        umat = np.zeros((self.n_rows*2-1, self.n_cols*2-1))
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Loop over neurons
        for i in range(self.n_rows * self.n_cols):
            row, col = i // self.n_cols, i % self.n_cols
            
            # Calculate means of distances in feature space to immediate neighbours
            dmat_neighbours = np.where((self.d2mat>0) & (self.d2mat<=2), self.dmat, np.nan
                                      )[i].reshape((self.n_rows, self.n_cols))
            dmat_neighbours_mean = np.nanmean(dmat_neighbours)
            umat[row*2, col*2] = dmat_neighbours_mean
            
            # Plot each neuron
            ax.plot([col*2], [row*2], marker='o', color='k', markersize=2)

            # Subset row, column coordinates of neighbours and calculate offset
            rows_alt, cols_alt = np.where(~np.isnan(dmat_neighbours))
            rows_offset, cols_offset = rows_alt - row, cols_alt - col
            
            # Fill between-neuron distances in U-matrix
            umat[row*2+rows_offset, col*2+cols_offset] = dmat_neighbours[rows_alt, cols_alt]
        
        # Save and plot U-matrix
        self.umat = umat
        ax.imshow(umat, cmap='Reds')
        ax.tick_params(labelbottom=False,labeltop=True)
        plt.tight_layout()
        
        
    def component_planes(self, i=None, cmap='viridis_r', figsize=(6,6)):
        """Plot component planes.
        
        Parameters
        ----------
            i : int, list or ndarray
                Index or indices of neuron weights to visualise.
            cmap : str
                Matplotlib colourmap.
        """
        
        if i is None:
            to_loop = range(self.wts.shape[1])
        elif isinstance(i, int):
            to_loop = [i]
        else:
            to_loop = i
            
        # Loop over dimensions of codebook
        for i in to_loop:
            arr = self.wts[:,i].reshape((self.n_rows, self.n_cols))
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(arr, cmap=cmap)