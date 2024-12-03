import logging
from functools import partial

import timm
from transformers import AutoModel, RobertaModel

from models.losses import CLIP_Loss, CyCLIP_Loss, SogCLR_Loss, VICReg_Loss
from models.losses import iSogCLR_New_v2_Loss, iSogCLR_New_v1_Loss, onlineCLR_Loss, iSogCLR_New_Loss, HybridSogCLR_Loss

import torch
from torch import nn
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(self,               
                 image_encoder = None,
                 text_encoder = None,
                 embed_dim = 256,
                 init_model = True,
                 world_size = 8,
                 ita_type = 'clip',
                 sogclr_gamma = 0.9,
                 rho_I = 0.1,
                 rho_T = 0.1,
                 rho_init=6.0,
                 eta_init = 0.001,
                 tau_init = 0.01,
                 eta_sched = None,
                 eta_exp_gamma = 0.8,
                 beta_u = 0.9,
                 temp = 0.01,
                 learnable_temp = False,
                 personalized_tau = False,
                 bsz = 128,
                 vicreg_sim_coeff = 25.0, 
                 vicreg_std_coeff = 25.0,
                 use_temp_net = True,
                 alpha = 1.0,
                 # New hybrid-specific parameters
                 num_strategies=3,
                 neg_samples_per_strategy=256,
                 memory_bank_size=65536,
                 memory_bank_momentum=0.99,
                 hybrid_tau_min=0.005,
                 hybrid_tau_max=1.0,
                 distributed=True,
                 # Add new hybrid negative sampling parameters
                 hard_negative_temp=0.1,
                 semantic_threshold=0.7,
                 warmup_epochs=5,
                 hard_mining_epochs=10,
                 mining_strategy='hybrid',  # 'hard', 'semantic', or 'hybrid'
                 cross_modal_weight=0.3,
                 difficulty_weight=0.2,
                 ):

        super().__init__()

        self.temp = temp
        self.learnable_temp = learnable_temp
        self.personalized_tau = personalized_tau

        self.distributed = distributed

        if self.learnable_temp:
            if not personalized_tau:
                self.temp = nn.Parameter(torch.ones([]) * self.temp)
            else:
                self.image_temp = nn.Parameter(torch.ones(2900000) * self.temp)
                self.text_temp = nn.Parameter(torch.ones(2900000) * self.temp)
    
        self.visual_encoder = timm.create_model(image_encoder, pretrained=init_model)
        self.visual_encoder.reset_classifier(0)

        if text_encoder == 'roberta-large':
            self.text_encoder = RobertaModel.from_pretrained(text_encoder)
            self.text_proj = nn.Linear(1024, embed_dim)
        else:
            self.text_encoder = AutoModel.from_pretrained(text_encoder)
            self.text_proj = nn.Linear(768, embed_dim)

        if not init_model:
            self.text_encoder.init_weights()

        self.vision_proj = nn.Linear(self.visual_encoder.num_features, embed_dim)

        self.ita_type = ita_type

        if self.ita_type == 'clip':
            if not personalized_tau:
                self.criterion = CLIP_Loss(world_size=world_size, personalized_tau=personalized_tau, temperature=self.temp)
            else:
                self.criterion = CLIP_Loss(world_size=world_size, personalized_tau=personalized_tau, image_tau=self.image_temp, text_tau=self.text_temp)

        elif self.ita_type == 'cyclip':
            self.criterion = CyCLIP_Loss(world_size=world_size, temperature=self.temp)

        elif self.ita_type == 'vicreg':
            self.criterion = VICReg_Loss(world_size=world_size, dim_size=embed_dim, sim_coeff=vicreg_sim_coeff, std_coeff=vicreg_std_coeff)

        elif self.ita_type == 'sogclr':
            # self.criterion = SogCLR_Loss(world_size=world_size, gamma=sogclr_gamma, temperature=self.temp, bsz=bsz, enable_surrogate=enable_surrogate, 
            #                              surrogate_c=surrogate_c, lamda_rho=lamda_rho, lamda_init=lamda_init)
            self.criterion = SogCLR_Loss(world_size=world_size, gamma=sogclr_gamma, temperature=self.temp, bsz=bsz)

        # elif self.ita_type == 'sogclr_dro':
        #     self.criterion = SogCLR_DRO_Loss(world_size=world_size, gamma=sogclr_gamma, rho_init=rho_init, tau_init=tau_init, bsz=bsz,
        #                                      eta_init=eta_init, beta_u=beta_u, enable_surrogate=enable_surrogate)
        elif self.ita_type == 'isogclr_new_v2':
            self.criterion = iSogCLR_New_v2_Loss(world_size=world_size, gamma=sogclr_gamma, rho_init=rho_init, tau_init=tau_init, bsz=bsz,
                                                 eta_init=eta_init, beta_u=beta_u)
        elif self.ita_type == 'isogclr_new_v1':
            self.criterion = iSogCLR_New_v1_Loss(world_size=world_size, gamma=sogclr_gamma, rho_init=rho_init, bsz=bsz)
        elif self.ita_type == 'onlineclr':
            self.criterion = onlineCLR_Loss(world_size=world_size, temperature=self.temp, gamma=sogclr_gamma)

        elif self.ita_type == 'isogclr_new':
            self.criterion = iSogCLR_New_Loss(world_size=world_size, gamma=sogclr_gamma, rho_I=rho_I, rho_T=rho_T, tau_init=tau_init, bsz=bsz,
                                              use_temp_net=use_temp_net, feature_dim=embed_dim)

        elif self.ita_type == 'hybrid':
                    # Calculate safe memory bank size (should be power of 2)
                    safe_bank_size = min(memory_bank_size, 8192)

                    # Initialize criterion without explicitly moving to device
                    self.criterion = HybridSogCLR_Loss(
                        N=safe_bank_size,
                        gamma=sogclr_gamma,
                        temperature=tau_init,
                        bsz=min(bsz, safe_bank_size // 4),  # Ensure batch size doesn't exceed memory bank capacity
                        world_size=world_size,
                        eps=1e-8,
                        memory_momentum=memory_bank_momentum,
                        feature_dim=embed_dim,
                        cross_modal_weight=0.5,
                        difficulty_weight=0.1,
                        covariance_weight=0.1,
                        use_adaptive_temp=True,
                        min_bank_size=1024,
                        num_strategies=2
                    )

        else:
            raise NotImplementedError


    def forward(self, image, text, idx, text_idx, epoch, max_epoch):
        if self.learnable_temp:
            with torch.no_grad():
                if not self.personalized_tau:
                    self.temp.clamp_(0.001, 0.5)
                else:
                    self.image_temp.clamp_(0.001, 0.5)
                    self.text_temp.clamp_(0.001, 0.5)
        
        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds)
        image_feat = F.normalize(image_embeds, dim=-1)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, output_hidden_states=False)
        text_embeds = self.text_proj(text_output.last_hidden_state[:,0,:])
        text_feat = F.normalize(text_embeds, dim=-1)

        avg_image_tau = None
        avg_text_tau = None
        cur_eta = None
        grad_tau_image = None
        grad_tau_text = None
        b_I = None
        b_T = None

        info_dict = {}

        if self.ita_type in ['clip', 'cyclip']:
            if self.personalized_tau:
                if self.distributed:
                    image_ids = concat_all_gather(idx)
                    text_ids = concat_all_gather(text_idx)
                else:
                    image_ids, text_ids = idx, text_idx
                loss_ita = self.criterion(image_feat, text_feat, image_ids, text_ids)
                info_dict['avg_image_tau'] = self.criterion.image_tau[image_ids].mean()
                info_dict['avg_text_tau'] = self.criterion.text_tau[text_ids].mean()

            else:
                loss_ita = self.criterion(image_feat, text_feat)
                if not self.learnable_temp:
                    avg_tau = torch.tensor(self.temp)
                else:
                    avg_tau = self.temp
                info_dict['avg_image_tau'] = avg_tau
                info_dict['avg_text_tau'] = avg_tau

        elif self.ita_type == 'vicreg':
            loss_ita = self.criterion(image_embeds, text_embeds)
            info_dict['avg_image_tau'] = 0.0
            info_dict['avg_text_tau'] = 0.0

        elif self.ita_type == 'sogclr':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch)
            if not self.learnable_temp:
                avg_tau = torch.tensor(self.temp)
            else:
                avg_tau = self.temp
            info_dict['avg_text_tau'] = avg_text_tau
            info_dict['avg_image_tau'] = avg_image_tau
            info_dict['lamda'] = 0.0

        elif self.ita_type in ['sogclr_dro', 'isogclr_new']:
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau, cur_eta, grad_tau_image, grad_tau_text, b_I, b_T = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch, max_epoch)
            info_dict = {'avg_image_tau':avg_image_tau, 'avg_text_tau':avg_text_tau, 'cur_eta':cur_eta, 
                         'grad_tau_image':grad_tau_image, 'grad_tau_text':grad_tau_text, 'b_I':b_I, 'b_T':b_T}

        elif self.ita_type == 'isogclr_new_v2':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau, cur_eta, grad_tau_image, grad_tau_text, b_I, b_T, v, lamda = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch, max_epoch)
            info_dict = {'avg_image_tau':avg_image_tau, 'avg_text_tau':avg_text_tau, 'cur_eta':cur_eta, 
                         'grad_tau_image':grad_tau_image, 'grad_tau_text':grad_tau_text, 'b_I':b_I, 'b_T':b_T, 'v':v, 'lamda':lamda}

        elif self.ita_type == 'isogclr_new_v1':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch)
            info_dict['avg_text_tau'] = avg_text_tau
            info_dict['avg_image_tau'] = avg_image_tau

        elif self.ita_type == 'hybrid':
            try:
                # Handle distributed setup
                if self.distributed:
                    try:
                        image_ids = concat_all_gather(idx)
                        text_ids = concat_all_gather(text_idx)
                    except Exception as e:
                        logging.warning(f"Error in all_gather: {e}, falling back to local ids")
                        image_ids, text_ids = idx, text_idx
                else:
                    image_ids, text_ids = idx, text_idx

                # Ensure inputs are contiguous and on correct device
                image_feat = image_feat.contiguous()
                text_feat = text_feat.contiguous()

                # Call hybrid loss function with autocast for better stability
                with torch.cuda.amp.autocast(enabled=True):
                    loss_ita, loss_stats = self.criterion(
                        image_feat, text_feat, image_ids, text_ids, epoch
                    )

                # Verify loss_ita is a tensor
                if not isinstance(loss_ita, torch.Tensor):
                    raise ValueError("Loss must be a tensor")

                # Ensure loss_stats has all required keys with safe defaults
                default_stats = {
                    'temperature': self.criterion.temperature.item(),
                    'base_loss': loss_ita.item(),
                    'cov_loss': 0.0,
                    'diff_loss': 0.0,
                    'memory_stats': {
                        'image_norm': torch.norm(self.criterion.b_I, dim=1).mean().item(),
                        'text_norm': torch.norm(self.criterion.b_T, dim=1).mean().item()
                    },
                    'scales': {
                        'base': 1.0,
                        'cov': 0.0,
                        'diff': 0.0
                    },
                    'warnings': {
                        'memory_warnings': 0,
                        'gradient_warnings': 0,
                        'temperature_warnings': 0
                    }
                }

                # Safely update default_stats with actual values
                for key in default_stats:
                    if key in loss_stats:
                        if isinstance(default_stats[key], dict):
                            default_stats[key].update(loss_stats[key])
                        else:
                            default_stats[key] = loss_stats[key]

                # Create backward-compatible stats dictionary
                compatible_stats = {
                    'avg_image_tau': default_stats['temperature'],
                    'avg_text_tau': default_stats['temperature'],
                    'cur_eta': default_stats['temperature'],
                    'grad_tau_image': default_stats['base_loss'],
                    'grad_tau_text': default_stats['base_loss'],
                    'b_I': default_stats['memory_stats']['image_norm'],
                    'b_T': default_stats['memory_stats']['text_norm'],
                    'memory_bank_size': len(self.criterion.b_I),
                }

                # Update info dictionary with compatible stats
                info_dict.update(compatible_stats)

                # Add new metrics including the new loss components and scales
                info_dict.update({
                    'temperature': default_stats['temperature'],
                    'base_loss': default_stats['base_loss'],
                    'cov_loss': default_stats['cov_loss'],
                    'diff_loss': default_stats['diff_loss'],
                    'image_norm': default_stats['memory_stats']['image_norm'],
                    'text_norm': default_stats['memory_stats']['text_norm'],
                    'loss_scales': default_stats['scales'],
                    'warnings': default_stats['warnings']
                })

                # Monitor warnings if they exist
                if any(default_stats['warnings'].values()):
                    logging.warning(f"Loss warnings detected: {default_stats['warnings']}")

                return loss_ita, info_dict

            except Exception as e:
                logging.error(f"Error in hybrid forward pass: {str(e)}")
                # Return default values in case of error
                default_loss = torch.tensor(0.0, device=image_feat.device, requires_grad=True)
                default_info = {
                    'avg_image_tau': self.criterion.temperature.item(),
                    'avg_text_tau': self.criterion.temperature.item(),
                    'cur_eta': self.criterion.temperature.item(),
                    'grad_tau_image': 0.0,
                    'grad_tau_text': 0.0,
                    'b_I': 0.0,
                    'b_T': 0.0,
                    'memory_bank_size': len(self.criterion.b_I),
                    'temperature': self.criterion.temperature.item(),
                    'base_loss': 0.0,
                    'cov_loss': 0.0,
                    'diff_loss': 0.0,
                    'image_norm': 0.0,
                    'text_norm': 0.0,
                    'loss_scales': {'base': 1.0, 'cov': 0.0, 'diff': 0.0},
                    'warnings': {'memory': 0, 'gradient': 0, 'temperature': 0}
                }
                return default_loss, default_info

        elif self.ita_type == 'onlineclr':
            loss_ita = self.criterion(image_feat, text_feat)
            info_dict['avg_text_tau'] = 0.0
            info_dict['avg_image_tau'] = 0.0

        else:
            raise NotImplementedError

        return loss_ita, info_dict

    def get_training_statistics(self):
        """
        Get detailed statistics about the current training state.
        Only available when using hybrid loss (ita_type='hybrid').
        """
        if self.ita_type == 'hybrid':
            return self.criterion.get_training_statistics()
        else:
            return {
                'warning': 'Training statistics are only available with hybrid loss type',
                'current_loss_type': self.ita_type
            }


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output        

