import torch
import numpy as np
import torch.func

class SILoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            # New parameters
            time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
            time_mu=-0.4,                 # Mean parameter for logit_normal distribution
            time_sigma=1.0,               # Std parameter for logit_normal distribution
            ratio_r_not_equal_t=0.75,     # Ratio of samples where r≠t
            adaptive_p=1.0,               # Power param for adaptive weighting
            label_dropout_prob=0.1,       # Drop out label
            # CFG related params
            cfg_omega=1.0,                # CFG omega param, default 1.0 means no CFG
            cfg_kappa=0.0,                # CFG kappa param for mixing class-cond and uncond u
            cfg_min_t=0.0,                # Minium CFG trigger time 
            cfg_max_t=0.8,                # Maximum CFG trigger time
            ):
        self.weighting = weighting
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = label_dropout_prob
        # Adaptive weight config
        self.adaptive_p = adaptive_p
        
        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t


    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        
        return r, t
    
    def __call__(self, model, images, model_kwargs=None):
        """
        Compute MeanFlow loss function
        """
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        batch_size = images.shape[0]
        device = images.device

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()  
            batch_size = y.shape[0]
            if hasattr(model, 'module'):  # DDP
                num_classes = model.module.num_classes
            else:
                num_classes = model.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)

        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)

        # ------------------------------------------------------------------
        # New logic: split the batch based on whether r == t to skip costly
        # JVP computation when time_diff is zero.
        # ------------------------------------------------------------------
        eq_mask = (t == r)                    # Indices where time_diff == 0
        neq_mask = ~eq_mask                   # Indices where time_diff != 0

        # Allocate tensors for the full batch
        u_target = torch.zeros_like(v_t)
        u = torch.zeros_like(v_t)

        # -----------------------------
        # 1) Process r == t  (no JVP)  
        # -----------------------------
        if eq_mask.any():
            eq_idx = torch.where(eq_mask)[0]
            z_t_eq   = z_t[eq_idx]
            r_eq     = r[eq_idx]
            t_eq     = t[eq_idx]
            v_t_eq   = v_t[eq_idx]

            # Slice model kwargs for this subset (tensor values only)
            eq_kwargs = {}
            for k, v in model_kwargs.items():
                if isinstance(v, torch.Tensor):
                    eq_kwargs[k] = v[eq_idx]
                else:
                    eq_kwargs[k] = v

            # Single forward – no JVP needed
            u_eq = model(z_t_eq, r_eq, t_eq, **eq_kwargs)
            u_target_eq = v_t_eq  # time_diff = 0 => target is just velocity

            # Store results back
            u[eq_idx] = u_eq
            u_target[eq_idx] = u_target_eq

        # ---------------------------------
        # 2) Process r != t  (standard JVP)
        # ---------------------------------
        if neq_mask.any():
            neq_idx = torch.where(neq_mask)[0]

            # Gather subset tensors
            z_t_neq   = z_t[neq_idx]
            r_neq     = r[neq_idx]
            t_neq     = t[neq_idx]
            v_t_neq   = v_t[neq_idx]
            time_diff_neq = time_diff[neq_idx]
            unconditional_mask_neq = unconditional_mask[neq_idx]

            # Slice model kwargs for this subset (tensor values only)
            neq_kwargs = {}
            for k, v in model_kwargs.items():
                if isinstance(v, torch.Tensor):
                    neq_kwargs[k] = v[neq_idx]
                else:
                    neq_kwargs[k] = v

            # Helper for model call with current kwargs
            def fn_current_neq(z, cur_r, cur_t):
                return model(z, cur_r, cur_t, **neq_kwargs)

            # Determine which of these samples need CFG processing
            cfg_time_mask_neq = (t_neq >= self.cfg_min_t) & (t_neq <= self.cfg_max_t) & (~unconditional_mask_neq)

            if neq_kwargs.get('y') is not None and cfg_time_mask_neq.any():
                # -----------------------------
                # Split further into CFG / no-CFG
                # -----------------------------
                cfg_idx_local = torch.where(cfg_time_mask_neq)[0]
                no_cfg_idx_local = torch.where(~cfg_time_mask_neq)[0]

                # --- CFG branch (with improved guidance) ---
                if len(cfg_idx_local) > 0:
                    cfg_select = cfg_idx_local
                    cfg_z_t = z_t_neq[cfg_select]
                    cfg_v_t = v_t_neq[cfg_select]
                    cfg_r   = r_neq[cfg_select]
                    cfg_t   = t_neq[cfg_select]
                    cfg_time_diff = time_diff_neq[cfg_select]

                    cfg_kwargs = {k: (v[cfg_select] if isinstance(v, torch.Tensor) else v) for k, v in neq_kwargs.items()}

                    # Build conditional & unconditional batches to compute v_tilde
                    cfg_y = cfg_kwargs.get('y')
                    if hasattr(model, 'module'):
                        num_classes = model.module.num_classes
                    else:
                        num_classes = model.num_classes

                    cfg_z_t_batch = torch.cat([cfg_z_t, cfg_z_t], dim=0)
                    cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0)
                    cfg_t_end_batch = torch.cat([cfg_t, cfg_t], dim=0)
                    cfg_y_batch = torch.cat([cfg_y, torch.full_like(cfg_y, num_classes)], dim=0)

                    cfg_combined_kwargs = cfg_kwargs.copy()
                    cfg_combined_kwargs['y'] = cfg_y_batch

                    with torch.no_grad():
                        cfg_combined_u_at_t = model(cfg_z_t_batch, cfg_t_batch, cfg_t_end_batch, **cfg_combined_kwargs)
                        cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                        cfg_v_tilde = (
                            self.cfg_omega * cfg_v_t +
                            self.cfg_kappa * cfg_u_cond_at_t +
                            (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t
                        )

                    # Now compute JVP only for these cfg samples
                    def fn_current_cfg(z, cur_r, cur_t):
                        return model(z, cur_r, cur_t, **cfg_kwargs)

                    primals = (cfg_z_t, cfg_r, cfg_t)
                    tangents = (cfg_v_tilde, torch.zeros_like(cfg_r), torch.ones_like(cfg_t))
                    cfg_u_theta, cfg_dudt = torch.func.jvp(fn_current_cfg, primals, tangents)

                    cfg_u_target = cfg_v_tilde - cfg_time_diff * cfg_dudt

                    # Write back to main tensors (map local -> global indices)
                    global_cfg_idx = neq_idx[cfg_select]
                    u[global_cfg_idx] = cfg_u_theta
                    u_target[global_cfg_idx] = cfg_u_target

                # --- NO-CFG branch (standard JVP) ---
                if len(no_cfg_idx_local) > 0:
                    no_cfg_select = no_cfg_idx_local
                    no_cfg_z_t = z_t_neq[no_cfg_select]
                    no_cfg_v_t = v_t_neq[no_cfg_select]
                    no_cfg_r   = r_neq[no_cfg_select]
                    no_cfg_t   = t_neq[no_cfg_select]
                    no_cfg_time_diff = time_diff_neq[no_cfg_select]
                    no_cfg_kwargs = {k: (v[no_cfg_select] if isinstance(v, torch.Tensor) else v) for k, v in neq_kwargs.items()}

                    def fn_current_no_cfg(z, cur_r, cur_t):
                        return model(z, cur_r, cur_t, **no_cfg_kwargs)

                    primals = (no_cfg_z_t, no_cfg_r, no_cfg_t)
                    tangents = (no_cfg_v_t, torch.zeros_like(no_cfg_r), torch.ones_like(no_cfg_t))
                    no_cfg_u_theta, no_cfg_dudt = torch.func.jvp(fn_current_no_cfg, primals, tangents)

                    no_cfg_u_target = no_cfg_v_t - no_cfg_time_diff * no_cfg_dudt

                    global_no_cfg_idx = neq_idx[no_cfg_select]
                    u[global_no_cfg_idx] = no_cfg_u_theta
                    u_target[global_no_cfg_idx] = no_cfg_u_target
            else:
                # No CFG in this subset – standard JVP for all samples here
                primals = (z_t_neq, r_neq, t_neq)
                tangents = (v_t_neq, torch.zeros_like(r_neq), torch.ones_like(t_neq))
                u_neq, dudt_neq = torch.func.jvp(fn_current_neq, primals, tangents)
                u_target_neq = v_t_neq - time_diff_neq * dudt_neq

                u[neq_idx] = u_neq
                u_target[neq_idx] = u_target_neq

        # Detach the target to prevent gradient flow
        u_target = u_target.detach()

        error = u - u_target

        # Apply adaptive weighting based on configuration
        error_norm = torch.norm(error.reshape(error.shape[0], -1), dim=1)
        loss = error_norm ** 2

        if self.weighting == "adaptive":
            loss = torch.log(loss + 1e-8)

        return loss