"""Other models to compare GRAE."""

import os
import numpy as np
import scipy
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from torch.autograd import grad as torch_grad
from pydiffmap import diffusion_map as dm

from grae.models.grae_models import AE
from grae.models.external_tools.topological_loss import TopoAELoss, compute_distance_matrix
from grae.data.base_dataset import DEVICE

from grae.prop_topo_tools.connectivity_topo_regularizer import TopologicalZeroOrderLoss

class PropTopoAE(AE):
    """Topological Autoencoder.

    From the paper of the same name. See https://arxiv.org/abs/1906.00722.

    See external_tools/topological_loss.py for the loss definition. Adapted from their source code.
    """

    def __init__(self, *, lam=100,importance_strat = "min",take_top_p_scales=1.0,  **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.topo_loss = TopologicalZeroOrderLoss(method="deep",timeout=10,multithreading=True,importance_calculation_strat=importance_strat, take_top_p_scales=take_top_p_scales)
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True).to(DEVICE)
        self._last_calculated_loss = -0.0000000000001
    def compute_loss(self, x, x_hat, z, idx):
        """Compute topological loss over a batch.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        x_distances = compute_distance_matrix(x)

        dimensions = x.size()
        if len(dimensions) == 4:
            # If we have an image dataset, normalize using theoretical maximum
            batch_size, ch, b, w = dimensions
            # Compute the maximum distance we could get in the data space (this
            # is only valid for images wich are normalized between -1 and 1)
            max_distance = (2 ** 2 * ch * b * w) ** 0.5
            x_distances = x_distances / max_distance
        else:
            # Else just take the max distance we got in the batch
            x_distances = x_distances / x_distances.max()

        latent_distances = compute_distance_matrix(z)
        latent_distances = latent_distances / self.latent_norm
        topo_loss = self.topo_loss(x_distances, latent_distances)[0]
        loss = self.criterion(x, x_hat) + self.lam * topo_loss
        self._last_calculated_loss = topo_loss.detach().cpu().item()
        loss.backward()
        
        
    def log_metrics_val(self, epoch):
        """Compute validation metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Validation loss
        if self.val_loader is not None:
            val_mse = self.eval_MSE(self.val_loader)

            if self.comet_exp is not None:
                with self.comet_exp.validate():
                    self.comet_exp.log_metric('MSE_loss', val_mse, epoch=epoch)
                    self.comet_exp.log_metric('topo_loss', self._last_calculated_loss, epoch=epoch)

            if val_mse < self.current_loss_min:
                # If new min, update attributes and checkpoint model
                self.current_loss_min = val_mse
                self.early_stopping_count = 0
                self.save(os.path.join(self.write_path, 'checkpoint.pt'))
            else:
                self.early_stopping_count += 1

    def log_metrics_train(self, epoch):
        """Log train metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Train loss
        if self.comet_exp is not None:
            train_mse = self.eval_MSE(self.loader)
            with self.comet_exp.train():
                self.comet_exp.log_metric('MSE_loss', train_mse, epoch=epoch)
                self.comet_exp.log_metric('topo_loss', self._last_calculated_loss, epoch=epoch)

class TopoAE(AE):
    """Topological Autoencoder.

    From the paper of the same name. See https://arxiv.org/abs/1906.00722.

    See external_tools/topological_loss.py for the loss definition. Adapted from their source code.
    """

    def __init__(self, *, lam=100, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.topo_loss = TopoAELoss()
        self._last_calculated_loss = -0.0000000000001

    def compute_loss(self, x, x_hat, z, idx):
        """Compute topological loss over a batch.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        topo_loss = self.topo_loss(x, z)
        loss = self.criterion(x, x_hat) + self.lam * topo_loss
        self._last_calculated_loss = topo_loss.detach().cpu().item()
        loss.backward()
        
    def log_metrics_val(self, epoch):
        """Compute validation metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Validation loss
        if self.val_loader is not None:
            val_mse = self.eval_MSE(self.val_loader)

            if self.comet_exp is not None:
                with self.comet_exp.validate():
                    self.comet_exp.log_metric('MSE_loss', val_mse, epoch=epoch)
                    self.comet_exp.log_metric('topo_loss', self._last_calculated_loss, epoch=epoch)

            if val_mse < self.current_loss_min:
                # If new min, update attributes and checkpoint model
                self.current_loss_min = val_mse
                self.early_stopping_count = 0
                self.save(os.path.join(self.write_path, 'checkpoint.pt'))
            else:
                self.early_stopping_count += 1

    def log_metrics_train(self, epoch):
        """Log train metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Train loss
        if self.comet_exp is not None:
            train_mse = self.eval_MSE(self.loader)
            with self.comet_exp.train():
                self.comet_exp.log_metric('MSE_loss', train_mse, epoch=epoch)
                self.comet_exp.log_metric('topo_loss', self._last_calculated_loss, epoch=epoch)


class EAERMargin(AE):
    """AE with margin-based regularization in the latent space.

    As presented in the EAER paper. See https://link.springer.com/chapter/10.1007/978-3-642-40994-3_14

    Note : The algorithm was adapted to support mini-batch training and SGD.
    """

    def __init__(self, *, lam=100, n_neighbors=10, margin=1, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            n_neighbors(int): The size of local neighborhood used to build the neighborhood graph.
            margin(float):  Margin used for the max-margin loss.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.margin = margin
        self.knn_graph = None  # Will store the neighborhood graph of the data

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        x_np, _ = x.numpy()

        # Determine neighborhood parameters
        x_np, _ = x.numpy()
        if x_np.shape[1] > 100:
            print('Computing PCA before knn search...')
            x_np = PCA(n_components=100).fit_transform(x_np)

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto').fit(x_np)

        self.knn_graph = nbrs.kneighbors_graph()

        super().fit(x)

    def compute_loss(self, x, x_hat, z, idx):
        """Compute max-margin loss over a batch.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        if self.lam > 0:
            batch_d = compute_distance_matrix(z)
            is_nb = torch.from_numpy(self.knn_graph[np.ix_(idx, idx)].toarray()).to(DEVICE)

            clipped_dist = torch.clamp(input=self.margin - batch_d, min=0)

            d = is_nb * batch_d + (1 - is_nb) * clipped_dist ** 2

            margin_loss = torch.sum(d)

            loss = self.criterion(x, x_hat) + self.lam * margin_loss
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()


class DiffusionNet(AE):
    """Diffusion nets.

    As presented in https://arxiv.org/abs/1506.07840

    Note: Subsampling was required to run this model on our benchmarks.

    """

    def __init__(self, *, lam=100, eta=100, n_neighbors=100, alpha=1, epsilon='bgh_generous', subsample=None, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor for the coordinate constraint.
            eta(float): Regularization factor for the EV constraint.
            n_neighbors(int): The size of local neighborhood used to build the neighborhood graph.
            alpha(float): Exponent to be used for the left normalization in constructing the diffusion map.
            epsilon(Any):  Method for choosing the epsilon. See scikit-learn NearestNeighbors class for details.
            subsample(int): Number of points to sample from the dataset before fitting.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.epsilon = epsilon
        self.subsample = subsample
        self.eta = eta

    def fit(self, x):
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        # DiffusionNet do not support mini-batches. Subsample data if needed to fit in memory
        if self.subsample is not None:
            x = x.random_subset(self.subsample, random_state=self.random_state)

        # Use whole dataset (after possible subsampling) as batch, as in the paper
        self.batch_size = len(x)

        x_np, _ = x.numpy()

        # Reduce dimensionality for faster kernel computations. We do the same with PHATE and UMAP.
        if x_np.shape[1] > 100 and x_np.shape[0] > 1000:
            print('Computing PCA before running DM...')
            x_np = PCA(n_components=100).fit_transform(x_np)

        neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=self.n_components,
                                              alpha=self.alpha,
                                              epsilon=self.epsilon,
                                              k=self.n_neighbors,
                                              neighbor_params=neighbor_params)
        dmap = mydmap.fit_transform(x_np)

        self.z = torch.tensor(dmap).float().to(DEVICE)

        self.Evectors = torch.from_numpy(mydmap.evecs).float().to(DEVICE)
        self.Evalues = torch.from_numpy(mydmap.evals).float().to(DEVICE)

        # Potential matrix sparse form
        P = scipy.sparse.coo_matrix(mydmap.L.todense())
        values = P.data
        indices = np.vstack((P.row, P.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        self.P = torch.sparse.FloatTensor(i, v).float().to(DEVICE)

        # Identity matrix sparse 
        I_n = scipy.sparse.coo_matrix(np.eye(self.batch_size))
        values = I_n.data
        indices = np.vstack((I_n.row, I_n.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        self.I_t = torch.sparse.FloatTensor(i, v).float().to(DEVICE)
        super().fit(x)

    def compute_loss(self, x, x_hat, z, idx):
        """Compute diffusion-based loss.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        rec_loss = self.criterion(x, x_hat)
        coord_loss = self.criterion(z, self.z[idx])
        Ev_loss = (torch.mean(torch.pow(torch.mm((self.P.to_dense() - self.Evalues[0] *
                                                  self.I_t.to_dense()),
                                                 z[:, 0].view(self.batch_size, 1)),
                                        2)) + torch.mean(
            torch.pow(torch.mm((self.P.to_dense() - self.Evalues[1] *
                                self.I_t.to_dense()),
                               z[:, 1].view(self.batch_size, 1)), 2)))

        loss = rec_loss + self.lam * coord_loss + self.eta * Ev_loss

        loss.backward()


class VAE(AE):
    """Variational Autoencoder class."""

    def __init__(self, *, beta=1, loss='MSE', **kwargs):
        """Init.

        Args:
            beta(float): Regularization factor for KL divergence.
            loss(str): 'BCE', 'MSE' or 'auto' for the reconstruction loss, depending on the desired p(x|z).
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.loss = loss

        if self.loss not in ('BCE', 'MSE'):
            raise ValueError(f'loss should either be "BCE" or "MSE", not {loss}')

    def init_torch_module(self, data_shape):
        super().init_torch_module(data_shape, vae=True, sigmoid=self.loss == 'BCE')

        # Also initialize criterion
        self.criterion = torch.nn.MSELoss(reduction='mean') if self.loss == 'MSE' else torch.nn.BCELoss(
            reduction='mean')

    def transform(self, x):
        """Transform data.

        Args:
            x(BaseDataset): Dataset to transform.
        Returns:
            ndarray: Embedding of x.

        """
        self.torch_module.eval()
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
        # Same as AE but slice only mu, ignore logvar
        z = [self.torch_module.encoder(batch.to(DEVICE))[:, :self.n_components].cpu().detach().numpy() for batch, _, _
             in loader]
        return np.concatenate(z)

    def compute_loss(self, x, x_hat, z, idx):
        """Apply VAE loss.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        mu, logvar = z.chunk(2, dim=-1)

        # From pytorch repo
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if self.beta > 0:
            # MSE and BCE are averaged. Do the same for KL
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            KLD = 0

        loss = self.criterion(x_hat, x) + self.beta * KLD

        loss.backward()


class DAE(AE):
    """Denoising Autoencoder.

    Supports both masking and gaussian noise.

    See Stacked Denoising Autoencoders: Learning Useful Representations in
    a Deep Network with a Local Denoising Criterion by Vincent et al."""

    def __init__(self, *, mask_p=0, sigma=0, clip=0, **kwargs):
        """Init.

        Args:
            mask_p(float): Input features will be set to 0 with probability p.
            sigma(float): Standard deviation of the isotropic gaussian noise added to the input.
            clip(int): 0: no clip.  1 : clip values between 0 and 1. 2 : clip values between 0 and +inf.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)

        if sigma < 0:
            raise ValueError(f'sigma should be a positive number.')

        if mask_p < 0 or mask_p >= 1:
            raise ValueError(f'mask_p should be in [0, 1).')

        self.mask_p = mask_p
        self.sigma = sigma
        self.clip = clip

    def train_body(self, batch):
        """Called in main training loop to update torch_module parameters.

        Add corruption to input.

        Args:
            batch(tuple[torch.Tensor]): Training batch.

        """
        data, _, idx = batch  # No need for labels. Training is unsupervised
        data = data.to(DEVICE)

        data_corrupted = data.clone()

        if self.sigma > 0:
            data_corrupted += self.sigma * torch.randn_like(data_corrupted, device=DEVICE)
            if self.clip == 1:
                data_corrupted = torch.clip(data_corrupted, 0, 1)
            elif self.clip == 2:
                data_corrupted = torch.clip(data_corrupted, 0, None)

        if self.mask_p > 0:
            if len(data.shape) == 4:
                # Broadcast noise across RGB channel
                n, _, h, w = data.shape
                u = torch.rand((n, 1, h, w),
                               dtype=data_corrupted.dtype,
                               layout=data_corrupted.layout,
                               device=data_corrupted.device)
                # View sample
            else:
                u = torch.rand_like(data_corrupted, device=DEVICE)

            data_corrupted *= u > self.mask_p

            # View samples for debugging
            # for i in range(3):
            #     sample = data_input[i].cpu().numpy()
            #     sample = np.transpose(sample, (1, 2, 0)).squeeze()
            #     import matplotlib.pyplot as plt
            #     plt.imshow(sample)
            #     plt.show()
            # exit()

        x_hat, z = self.torch_module(data_corrupted)  # Forward pass

        # Compute loss using original uncorrupted data
        self.compute_loss(data, x_hat, z, idx)


class CAE(AE):
    """Contractive Autoencoders.

    Add Frobenius norm of the Jacobian of the embedding with respect to the input.
    From Contractive Auto-Encoders : Explicit Invariance during Feature Extraction
    by Rifai, Vincent et al.
    """

    def __init__(self, *, lam=1, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor for Frobenius norm of encoder Jacobian.
            **kwargs: All other keyword arguments are passed to the AE parent class.
        """
        super().__init__(**kwargs)
        self.lam = lam

    def train_body(self, batch):
        # Tell torch to compute gradient w.r. to input
        batch[0].requires_grad = True
        super().train_body(batch)

    def compute_loss(self, x, x_hat, z, idx):
        """Apply loss to update parameters following a forward pass.

        Regularize loss with the Frobenius norm of the Jacobian of the embedding with respect to the input.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        # Retain input gradient
        x.retain_grad()

        # Note : Test code for future review. See actual code below.
        # # Method found online.
        # # See https://stackoverflow.com/questions/58249160/how-to-implement-contractive-autoencoder-in-pytorch
        # z.backward(torch.ones_like(z), retain_graph=True)
        # grads_1 = x.grad  # Not a jacobian...
        # frob_1 = (grads_1 ** 2).sum()/x.shape[0]
        # print(grads_1.shape)
        # print(frob_1)
        # x.grad = None
        #
        # # The above does not compute the actual Jacobian, but rather the gradient w.r. to the sum of
        # # the latent space dimensions
        # s = z.sum()
        # s.backward(retain_graph=True)
        # grads_2 = x.grad
        # frob_2 = (grads_2 ** 2).sum()/x.shape[0]
        # print(grads_2.shape)
        # print(frob_2)
        # print(torch.allclose(grads_1, grads_2))
        # x.grad = None

        # Naive loopy way to compute the actual encoder Jacobian by stacking gradients.
        # Get one matrix by element in the batch. This can expensive if latent space is high dimensional!
        # Note : This could be improved. See https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa. The
        # trick requires to embed multiple copies of the batch however.
        if self.lam > 0:
            # Stack gradients to compute jacobian
            grads = list()
            for i in range(z.shape[1]):
                g = torch_grad(inputs=x, outputs=z[:, i],
                               grad_outputs=torch.ones_like(z[:, 0]), retain_graph=True,
                               create_graph=True)[0]
                grads.append(g)
                x.grad = None  # Reset input gradient
            jaco = torch.stack(grads, dim=1)

            # Reduction by mean over the batch
            frob_squared = (jaco ** 2).sum() / x.shape[0]
        else:
            frob_squared = 0

        rec = self.criterion(x_hat, x)
        loss = rec + self.lam * frob_squared
        loss.backward()
