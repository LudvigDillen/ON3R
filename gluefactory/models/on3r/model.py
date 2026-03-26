import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np

from . import geometry as geo


class ON3R(nn.Module):
    """
    Maps normalized 2D query keypoints to 3D.
    """

    def __init__(self, pred_mask, intrinsics_query, intrinsics, extrinsics, ref_img_dims, ref_kpts,
                 max_epochs, hidden_dim=512, positional_encoding_frequencies=5, n_layers=5,
                 dev='cuda', dt=torch.float32, bundle_adjust=True, cauchy_scaler=None,
                 kpt_depths_refs=None, moge2_depth_loss_scaler=0.1, s_min_depth=50.0,
                 s_scaler_depth=5.0, s_max=100.0, query_kpts=None, new_cam_center_db=None,
                 bias_scale=10.0):
        """
        A simple MLP for the warp_query function.

        Parameters:
        pred_mask (Tensor): Boolean mask of shape (N, T)
            indicating which keypoints to use in the loss.
        intrinsics_query (Tensor): Intrinsics matrix of shape (3, 3). (normalized to pixel space)
        intrinsics (Tensor): Intrinsics matrix of shape (T, 3, 3). (normalized to pixel space)
        extrinsics (Tensor): Extrinsics matrix of shape (T, 3, 4). (world to camera)
        ref_img_dims (Tensor): Reference image dimensions of shape (T, 2).
        ref_kpts (Tensor): Reference keypoints of shape (N, T, 2).
        max_epochs (int): Maximum number of epochs for training.
        hidden_dim (int): Hidden layer size of the MLP.
        positional_encoding_frequencies (int): Number of frequencies to use in positional encoding.
        n_layers (int): Number of hidden layers in the MLP.
        dev (str): Device to put tensors on.
        dt (torch.dtype): Data type for the tensors.
        bundle_adjust (bool): Whether to use bundle adjustment.
        cauchy_scaler (float): Cauchy scaler for the loss.
        kpt_depths_refs (Tensor): Depths of the reference keypoints list of depths. Shape (N, T, 1).
        moge2_depth_loss_scaler (float): Scaling factor for the MoGe-2 depth loss.
        s_min_depth (float): Minimum value for the depth scaling factor.
        s_scaler_depth (float): Scaling factor for the depth scaling factor.
        s_max (float): Maximum value for the cauchy scaler.
        query_kpts (Tensor): Query keypoints of shape (N, 2).
        new_cam_center_db (Tensor): New camera center for the database cameras of shape (T, 3).
        bias_scale (float): scaling factor from median camera center to initial bias. If <= 0, no scaling is applied.
        """
        super(ON3R, self).__init__()
        self.pred_mask = pred_mask
        self.tuple_length = pred_mask.shape[1]

        if pred_mask is not None:
            self.n_train = pred_mask.size(0)

        self.input_dim = 2
        self.use_positional_encoding = positional_encoding_frequencies > 0
        self.positional_encoding_frequencies = positional_encoding_frequencies

        if self.use_positional_encoding:
            self.frequencies = 2.0 ** torch.arange(
                self.positional_encoding_frequencies, device=dev, dtype=dt)

        # If positional encoding is used:
        #   each input dimension is expanded by (2 * freq_count) for sin/cos + original 1
        # => (positional_dim * positional_encoding_frequencies + 1) * input_dim
        in_dim = (2*positional_encoding_frequencies + 1)*self.input_dim

        # Define the MLP
        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())

        # Combine all layers in a sequential block
        self.mlp = nn.Sequential(*layers)

        # Separate head for coordinate predictions (x, y, z)
        head_3d = []
        head_3d.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
        head_3d.append(nn.LayerNorm(hidden_dim))
        head_3d.append(nn.GELU())
        head_3d.append(nn.Linear(hidden_dim, 3))
        self.head_3d = nn.Sequential(*head_3d)

        # Allocate stuff for the loss
        self.dtype = dt
        self.device = dev

        self.intrinsics = intrinsics  # (T, 3, 3)  (normalized to pixel space)
        self.inv_intrinsics = intrinsics.inverse()  # (T, 3, 3)  (pixel space to normalized)
        self.extrinsics = extrinsics  # (T, 3, 4)  (world to camera)
        self.R = extrinsics[:, :, :3]  # (T, 3, 3)
        self.t = extrinsics[:, :, 3]   # (T, 3)
        self.new_cam_center_db = new_cam_center_db

        self.intrinsics_query = intrinsics_query  # (3, 3) (normalized to pixel space)
        self.inv_intrinsics_query = intrinsics_query.inverse()  # (3, 3) (pixel space to normalized)
        self.R_est_query = None  # (3, 3)
        self.t_est_query = None  # (3)  (expected to be parameter later)
        self.quats_est_query = None  # (4)  (expected to be parameter later)
        self.P_est_query = None  # (3, 4)  (expected to be parameter later)
        self.poselib_pose_est_query = None
        self.bundle_adjust = bundle_adjust

        self.camera_centers = -torch.einsum("tij,ti->tj", self.R, self.t)  # (T, 3)
        self.ones_h = torch.ones((self.n_train, 1), device=dev, dtype=dt)  # (N, 1)
        self.ref_img_dims = ref_img_dims  # (T, 2)

        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.tol_reproj = cauchy_scaler  # px
        # TODO: It is actually wrong to divide by intrinsics_query[0, 0] sinec the projections in
        #       the different cameras might be with different focal lengths. Maybe fix this.
        self.tol_reproj_norm = self.tol_reproj / self.intrinsics_query[0, 0]  # (1)
        self.s_max = s_max / self.intrinsics_query[0, 0]
        self.s_min = self.tol_reproj_norm  # (1)
        self.s_min_depth = s_min_depth / self.intrinsics_query[0, 0]
        self.s_scaler_depth = s_scaler_depth
        self.set_s(self.s_max)
        self.set_s_depth(self.s_scaler_depth*self.s_max)
        self.lambda_val = 1.0 / self.intrinsics_query[0, 0]  # (1)

        # Init last bias of head_3d
        self.prinicipal_axes = self.R[:, 2]  # (T, 3)

        bias = self.init_xyz_bias(ref_kpts, verbose=False)

        # Add varying noise levels to the bias to reproduce the bias sensitivity study
        # means = torch.zeros_like(bias)
        # sigmas = torch.full_like(bias, 16.0)
        # bias = bias + torch.normal(mean=means, std=sigmas)

        if bias_scale > 0:
            denom = self.pred_mask.sum(0).clamp(min=1)     # (T)
            weights = denom / denom.sum()  # (T)
            self.scale_cams_with_bias(bias_scale, bias, weights)

        with torch.no_grad():
           self.head_3d[-1].bias.data = bias  # (3)

        self.cauchy_scaler = cauchy_scaler
        self.use_moge2_depth = kpt_depths_refs is not None
        self.query_kpts = query_kpts  # (N, 2)

        if self.use_moge2_depth:
            assert kpt_depths_refs is not None, "kpt_depths_refs must be provided."
            self.kpt_depths_refs = kpt_depths_refs  # (n_errors_refs)
            self.valid_ref_depths = ~(torch.isinf(self.kpt_depths_refs) | torch.isnan(self.kpt_depths_refs) | (self.kpt_depths_refs <= 0))  # (n_errors_refs)
            self.kpt_depths_refs_valid = self.kpt_depths_refs[self.valid_ref_depths]  # (n_errors_refs_valid)
            assert torch.isnan(self.kpt_depths_refs_valid).sum() == 0, "Cannot have NaN depths in kpt_depths_refs_valid."
            assert (self.kpt_depths_refs_valid <= 0).sum() == 0, "All depths in kpt_depths_refs_valid must be positive."
            self.moge2_depth_loss_scaler = moge2_depth_loss_scaler
        self.to(device=dev, dtype=dt)

    def forward(self, XQ):
        """
        Forward pass for the warp_query function.

        Parameters:
        XQ (Tensor): Input tensor of shape (K, 2).

        Returns:
        Tensor Output of shape (K, 3/4), representing predicted (x, y, z).
        """
        if self.use_positional_encoding:
            XQ = self.apply_positional_encoding(XQ)  # (K, 2 + 2*2*positional_encoding_frequencies)
        mlp_output = self.mlp(XQ)  # (K, D)
        output_3d = self.head_3d(mlp_output)  # (K, 3)
        return output_3d

    def apply_positional_encoding(self, X):
        """
        Apply sine-based positional encoding to the input coordinates.

        Parameters:
        X (Tensor): Input tensor of shape (K, 2).

        Returns:
        Tensor: Positional encoded tensor of shape (K, 2 + 2*2*positional_encoding_frequencies).
        """
        # Generate frequencies dynamically based on self.positional_encoding_frequencies
        X_unsq = X.unsqueeze(-1)  # (K, 2, 1)
        sin_encoding = torch.sin(X_unsq * self.frequencies)  # (K, 2, freq_count)
        cos_encoding = torch.cos(X_unsq * self.frequencies)  # (K, 2, freq_count)
        # (K, 2, 2*freq_count)
        pos_enc = torch.cat([sin_encoding, cos_encoding], dim=-1).view(X.shape[0], -1)
        return torch.cat([X, pos_enc], dim=-1)

    def loss(self, output, r_in_mask, epoch, depth_only=False):
        """
        Parameters:
        output : Tensor
            Shape (N, 3). Predicted 3D keypoints
        ref_kpts : Tensor
            Shape (n_errors, 2). Reference keypoints in normalized pixel space.
        Returns:
            Tensor: Computed loss (scalar).
        """
        ones_h = self.ones_h  # (N, 1)
        X3D = output[..., :3]  # (N, 3)

        # Project 3D points to image plane (but keep in 3D space)
        proj_pts = geo.project_points(X3D, self.extrinsics, ones_h, keep_in_3D=True)  # (N, T, 3)
        proj_pts_filt = proj_pts[self.pred_mask]  # (n_errors, 3)
        n_errors = proj_pts_filt.shape[0]
        z_r = proj_pts_filt[:, 2]  # (n_errors)
        z_mask = z_r > 0
        normalized_proj_pts = proj_pts_filt[:, :2] / torch.clamp(z_r[:, None], min=1e-6)  # (n_errors, 2)
        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if self.use_moge2_depth:
            z_valid = z_r[self.valid_ref_depths] # [valid_z_mask]
            kdrv = self.kpt_depths_refs_valid # [valid_z_mask]
            n_valid = z_valid.shape[0]
            if n_valid > 0:
                # NOTE: If the scene is metric scale, we need not use gammas at all. But, we do it anyway for simplicity.
                #       The results should be about the same.
                proj = normalized_proj_pts[self.valid_ref_depths].detach()
                corr = r_in_mask[self.valid_ref_depths]
                squared_errors = torch.sum((proj - corr)**2, dim=-1)  # (N)
                w = 1 / (1 + (torch.clamp(squared_errors, max=(self.s_scaler_depth*self.s_max)**2) / (self.s**2 + 1e-9)))  # (N)
                kdrv_w = kdrv * w
                z_valid_d = z_valid.detach()
                z_valid_wd = z_valid_d * w
                gamma = torch.dot(kdrv_w, z_valid_wd) / torch.clamp(torch.dot(z_valid_wd, z_valid_wd), min=1e-6)
                if gamma > 0:
                    depth_error = ((gamma * z_valid - kdrv) / torch.clamp(kdrv, min=1e-6))**2
                    depth_loss = self.moge2_depth_loss_scaler * self.cauchy_loss(depth_error, mode="residual_based_depth").sum() / n_errors
                    loss += depth_loss
                else:
                    z_neg_mask = ~z_mask
                    if z_neg_mask.any():
                        residuals_z = -z_r[z_neg_mask]
                        loss += residuals_z.sum() / n_errors
            else:
                gamma = -1.0
                z_neg_mask = ~z_mask
                if z_neg_mask.any():
                    residuals_z = -z_r[z_neg_mask]
                    loss += residuals_z.sum() / n_errors

        if not depth_only:
            residuals = torch.sum((normalized_proj_pts[z_mask] - r_in_mask[z_mask])**2, dim=-1)
            if not epoch % 10:
                self.set_residual_metric(torch.sqrt(residuals).detach())
                self.set_s(self.residual_metric)
            residuals_scaled = self.cauchy_loss(residuals)  # (n_errors)

            loss += residuals_scaled.sum() / n_errors

        if torch.isnan(loss) or torch.isnan(z_r).any():
            print("NaN in loss or z_r!")

        return loss

    def get_residual_metric(self, residuals):
        if len(residuals) == 0:
            return self.s_max
        mean_diag = self.ref_img_dims.float().norm(dim=-1).mean()
        mask = residuals < mean_diag / self.intrinsics_query[0, 0].item()
        if mask.any():
            mean = residuals[mask].mean()
        else:
            return self.s_max
        weight = 0.70
        residual_metric = weight*(weight*residuals.median() + (1-weight)*mean)
        return residual_metric

    def set_residual_metric(self, residuals):
        self.residual_metric = self.get_residual_metric(residuals)
        return None

    def set_residual_depth_metric(self, residuals):
        self.residual_depth_metric = self.get_residual_metric(residuals)
        return None

    def set_s(self, s):
        self.s = max(s, self.s_min)

    def set_s_depth(self, s):
        self.s_depth = max(s, self.s_min_depth)

    def get_s(self, mode="init"):
        if mode == "init" or mode == "depth":
            t_relative = self.current_epoch / self.max_epochs
            w_t = torch.sqrt(torch.tensor(1 - t_relative**2))
            #w_t = torch.tensor(1 - t_relative**(1/2))
            s = self.s_max * w_t + self.s_min
            if mode == "depth":
                s = max(self.s_scaler_depth*s, self.s_min_depth)  # Increasing s for depth mode as the depth map is noisy
        elif mode == "residual_based" or mode == "residual_based_depth":
            s = self.s
            if mode == "residual_based_depth":
                s = max(self.s_scaler_depth*s, self.s_min_depth)  # Increasing s for depth mode as the depth map is noisy
                #s = self.s_depth
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'init' or 'depth' or 'residual_based' or 'residual_based_depth'.")
        # s_print = self.intrinsics_query[0, 0] * s
        # print(f"s ({mode}): {s_print:.1f}px at epoch {self.current_epoch}")

        # if mode == "residual_based_depth":
        #     t_relative = self.current_epoch / self.max_epochs
        #     w_t = torch.sqrt(torch.tensor(1 - t_relative**2))
        #     #w_t = torch.tensor(1 - t_relative**(1/2))
        #     s_init = self.s_max * w_t + self.s_min
        #     s_depth = max(self.s_scaler_depth*s_init, self.s_min_depth)

        #     print(f"s (residual_based_depth/depth): {self.intrinsics_query[0, 0] * s:.1f} vs {s_depth * self.intrinsics_query[0, 0]:.1f}px at epoch {self.current_epoch}")
        #     pass
        return s

    def cauchy_loss(self, x, y=None, w=None, mode="residual_based"):
        """
        Cauchy loss function.
        Args:
            x: estimated values of shape (..., D) where D is the dimension of the data.
            y: ground truth values of shape (..., D) where D is the dimension of the data.
            weights: optional weights of shape (..., 1) to scale the loss.
        Returns:
            loss: Scalar tensor representing the Cauchy loss.
        """
        if y is not None:
            if w is not None:
                squared_errors = torch.sum((w * (x - y))**2, dim=-1)  # (N)
            else:
                squared_errors = torch.sum((x - y)**2, dim=-1)  # (N)
        else:
            squared_errors = x
        s = self.get_s(mode)
        # loss = s**2 * torch.log(1 + squared_errors / s**2)
        # This line below leads to the loss gradient having the max magnitude of 1
        # instead of s leading to that we do not have to adjust the learning rate with s.
        # It does not matter that it is not the same as in the BA since this just affets
        # the rate of convergence which is controlled by the optimizer and the learning rate
        # which differs anyway. And here we change s, but in the BA we do not.
        loss = s * torch.log(1 + squared_errors / (s**2))
        return loss

    def get_weight_3d(self, X, pose, data=None, batch_ind=None):
        """
        Compute weights for each 3D point based on the ratio of distances in 2D and 3D space.

        Args:
            X : Tensor
                Shape (N, 3). 3D keypoints.
            pose : Tensor
                Shape (3, 4). Camera pose (rotation and translation) (assumes w2cam).
        Returns:
            w: Tensor
                Shape (N). Weights based on the ratio of distances in 2D and 3D space.
        """
        rot = pose[:3, :3]  # (3, 3)
        trans = pose[:, 3]  # (3)

        X_cam = torch.einsum("ij,nj->ni", rot, X) + trans[None]
        depths = X_cam[:, 2]  # (N)
        x_cam = X_cam[:, :2] / depths[:, None]  # (N, 2)

        y = torch.ones((1, 2), device=x_cam.device, dtype=x_cam.dtype)  # (1, 2)
        y_normalized = y / torch.norm(y, dim=1, keepdim=True)  # (1, 2)
        z_cam = x_cam + self.lambda_val*y_normalized  # (N, 2)
        z_cam_h = torch.cat((z_cam, torch.ones_like(z_cam[:, :1])), dim=1)  # (N, 3)
        numerator = self.lambda_val  # ||x - z||_2 = lambda_val

        Z_cam = depths[:, None] * z_cam_h
        denominator = torch.norm(X_cam - Z_cam, dim=1)  # (N)

        w = numerator / (denominator + 1e-6)  # (N)

        if data is not None and batch_ind is not None:
            plot = False
            if plot:
                import rerun as rr
                import numpy as np
                img_query = data["query_to_ref_0"]['view0']['image'][batch_ind].permute(1, 2, 0).cpu().numpy()
                name_cam = "world/cam"
                rr.init("Scene", spawn=True)  # spawn=True will open the Rerun Viewer automatically.

                image_clipped = np.clip(img_query, 0, 1)
                rr.log(f'{name_cam}/image', rr.Image(image_clipped))

                rr.log(
                    f"world/3d_points",
                    rr.Points3D(X_cam.detach().cpu().numpy(), colors=[0, 0, 255], radii=0.01)
                )

                rr.log(
                    f"world/Z",
                    rr.Points3D(Z_cam.detach().cpu().numpy(), colors=[0, 255, 0], radii=0.01)
                )


                #x_world = torch.einsum("ij,nj->ni", rot.T, x_h - trans[None])  # (N, 3)
                x_cam_h = torch.cat((x_cam, torch.ones_like(x_cam[:, :1])), dim=1)  # (N, 3)

                rr.log(
                    f"world/keypoints",
                    rr.Points3D(x_cam_h.detach().cpu().numpy(), colors=[255, 0, 0], radii=0.003)
                )

                #z_world = torch.einsum("ij,nj->ni", rot.T, z_h - trans[None])  # (N, 3)
                rr.log(
                    f"world/z",
                    rr.Points3D(z_cam_h.detach().cpu().numpy(), colors=[255, 255, 0], radii=0.003)
                )

                rr.log(
                    "world/correspondences",
                    rr.LineStrips3D(torch.stack((X_cam, x_cam_h), dim=1).detach().cpu().numpy(),
                    colors=[0, 255, 255], radii=0.0003,
                    ),
                )
                trans_ident_np = np.array([0, 0, 0], dtype=np.float32)
                rot_ident_np = np.eye(3, dtype=np.float32)
                rr.log("world/cam", rr.Transform3D(
                    translation=trans_ident_np, mat3x3=rot_ident_np, from_parent=True))
                rr.log(
                    "world/cam",
                    rr.Pinhole(
                        focal_length=[self.intrinsics_query[0, 0].item(), self.intrinsics_query[1, 1].item()],
                        principal_point=self.intrinsics_query[0:2, 2].detach().cpu().numpy(),
                        width=(self.intrinsics_query[0, 2]*2).detach().cpu().numpy(), # can be wrong
                        height=(self.intrinsics_query[1, 2]*2).detach().cpu().numpy(), # can be wrong
                    ),
                )
        return w.detach()

    def get_pose(self):
        normed_q = self.quats_est_query / torch.norm(self.quats_est_query)
        rot = geo.quaternion_to_matrix(normed_q)  # (3, 3)  expects (QW, QX, QY, QZ)
        P_ret = torch.hstack((rot, self.t_est_query[:, None]))  # (3, 4)
        return P_ret

    def init_xyz_bias(self, ref_kpts, verbose=False):
        r_K_norm = geo.normalized_pixels_to_kpts(ref_kpts, self.inv_intrinsics)  # (N, T, 2)

        start = torch.tensor(0.1, device=self.device, dtype=self.dtype).log().item()
        end = torch.tensor(50, device=self.device, dtype=self.dtype).log().item()
        S = 1000
        ls_log_scale = torch.linspace(start, end, S, device=self.device, dtype=self.dtype).exp()  # (S)

        start_biases = self.get_start_biases(ls_log_scale)  # (S, 3)

        bias_loss = geo.bias_loss(
            bias=start_biases, extrinsics=self.extrinsics, r_K_norm=r_K_norm,
            pred_mask=self.pred_mask)  # (S)
        ind = bias_loss.argmin()
        init_bias = start_biases[ind]  # (3)

        if verbose:
            y = bias_loss.cpu()
            x = ls_log_scale.cpu()
            ax = plt.gca()
            ax.scatter(x, y, s=1.0)
            ax.set_yscale('log')
            ax.set_xscale('log')

            # Get the x and y values at the minimum index
            x_min = x[ind].item()  # convert tensor to Python scalar
            y_min = y[ind].item()

            # Mark the minimum point with a red circle
            ax.scatter(x_min, y_min, color='red', s=50, marker='o', label='Min value')

            # Annotate the minimum point with its (x, y) coordinates
            annotation_text = f"({x_min:.3f}, {y_min:.3f})"
            ax.annotate(annotation_text,
                        (x_min, y_min),
                        textcoords="offset points",
                        xytext=(10, 10),  # offset the text from the point (in points)
                        arrowprops=dict(arrowstyle="->", color='red'))

            plt.legend()
            plt.show()

        # Gradient-Based Optimization of the bias
        bias = init_bias.clone()
        bias.requires_grad = True

        # Create an optimizer
        optimizer = torch.optim.Adam([bias], lr=1e-1)
        num_iters = 100
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.25)
        ones_h = torch.ones(1, 1, device=bias.device, dtype=bias.dtype)

        loss_history = []
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = geo.bias_loss(bias[None], self.extrinsics, r_K_norm,
                                   self.pred_mask, ones_h=ones_h)[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose:
                loss_history.append(loss.item())
                if (i % 50 == 0):
                    print(f"Iteration {i}, Loss: {loss.item()}")
                    print(f"Bias: {bias.detach().cpu().numpy()}")

        if verbose:
            print(f"Start bias {bias.detach().cpu().numpy()}")
            print(f"Init bias {init_bias.cpu().numpy()}")
            print(f"Optimized bias: {bias.detach().cpu().numpy()}")

            plt.figure(figsize=(10, 5))
            plt.plot(loss_history)
            plt.title("Bias Optimization Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.yscale("log")  # optional, if your losses span multiple orders
            plt.show()
        return bias

    def get_start_biases(self, ls_log_scale):
        offset_samples = ls_log_scale[:, None, None] * self.prinicipal_axes[None]  # (S, T, 3)
        S_points_in_front_of_cameras = self.camera_centers[None] + offset_samples  # (S, T, 3)
        start_biases = S_points_in_front_of_cameras.quantile(q=0.5, dim=1)  # (S, 3)
        return start_biases

    def scale_cams_with_bias(self, bias_scale, bias, weights):
        cam_filt = weights > (0.5/len(weights))  # Don't want to let cams with super-few matches influence. Probably outlier retrieval
        filt_median_cam_center = self.camera_centers[cam_filt].quantile(0.5, dim=0)
        self.b = bias.clone().detach()
        dist_to_bias = torch.norm(filt_median_cam_center - self.b)
        self.sc = bias_scale / torch.clamp(dist_to_bias, min=0.1)
        # This is the same as applying P'_w2c = P_w2c@H_inv_world <=> P'_c2w = H_world@P_c2w
        # where H = move_back_bias@scale_with_s@move_bias_to_origo  (T(b)@S(s)@T(-b))
        t_new = self.sc * self.t - (1 - self.sc) * torch.einsum("nij,j->ni", self.R, self.b)
        self.extrinsics = torch.cat((self.R, t_new[:, :, None]), dim=2)  # (T, 3, 4)
        self.old_extrinsics = self.extrinsics.clone().detach()
        return None


def custom_layernorm_init(model):
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)
