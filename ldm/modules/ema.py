import math
import torch
from torch import nn


class LitEma(nn.Module):
    def __init__(
        self,
        model,
        decay=0.9995,
        use_num_upates=True,
        ema_on_cpu=True,
        update_every=100,
        verbose=True,
    ):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        if update_every:
            decay = math.pow(decay, update_every)

        self.verbose = verbose
        self.update_every = update_every
        self.m_name2s_name = {}
        self.register_buffer(
            'decay', torch.tensor(decay, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            'num_updates',
            torch.tensor(0, dtype=torch.int)
            if use_num_upates
            else torch.tensor(-1, dtype=torch.int),
        )

        self.cpu_shadow_params = [{}] if ema_on_cpu else None

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})

                cloned = p.detach().clone().data
                cloned.requires_grad = False
                if ema_on_cpu:
                    self.cpu_shadow_params[0][s_name] = cloned.cpu()
                else:
                    self.register_buffer(s_name, cloned)

        self.collected_params = []

        self.requires_grad_(requires_grad=False)

    @torch.no_grad()
    def forward(self, model):
        decay = float(self.decay.item())

        num_updates = self.num_updates.item()
        if num_updates >= 0:
            num_updates += 1
            self.num_updates += 1
            decay = min(decay, (1 + num_updates) / (10 + num_updates))

        update_every = self.update_every
        if update_every and (num_updates % update_every != 0):
            return

        if self.verbose:
            print(
                f'Updating EMA update_every={update_every:03d}'
                + f'  num_updates={num_updates:09d}  decay={decay:.05f}  \n'
            )

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = (
                self.cpu_shadow_params[0]
                if self.cpu_shadow_params
                else dict(self.named_buffers())
            )

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    param = m_param[key]
                    if self.cpu_shadow_params:
                        param = param.detach().cpu()
                    else:
                        param = param.detach().to(shadow_params[sname].device)
                        shadow_params[sname] = shadow_params[sname].type_as(
                            param
                        )
                    shadow_params[sname].sub_(
                        one_minus_decay * (shadow_params[sname] - param)
                    )
                else:
                    assert not key in self.m_name2s_name

    @torch.no_grad()
    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = (
            self.cpu_shadow_params[0]
            if self.cpu_shadow_params
            else dict(self.named_buffers())
        )
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(
                    shadow_params[self.m_name2s_name[key]].data
                )
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        if self.collected_params:
            raise ValueError(
                'Params already saved, forgot to call .restore after .store?'
            )
        self.collected_params = [
            param.detach().clone().cpu() for param in parameters
        ]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        if not self.collected_params:
            raise ValueError(
                'No params to restore, call .store before .restore'
            )
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        del c_param, param
        del self.collected_params
        torch.cuda.empty_cache()
        self.collected_params = []

    ### Hacks to allow saving and restoring to the same place in the state_dict
    ### both with and without 
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        pass_on_state_dict = dict(state_dict)
        if self.cpu_shadow_params:
            for name, param in self.cpu_shadow_params[0].items():
                input_param = state_dict[prefix + name]
                with torch.no_grad():
                    param.copy_(input_param)
                del pass_on_state_dict[prefix + name]
        # decay shouldn't be stored as we want this to be configurable
        # removing it is necessary to allow loading state from a checkpoint
        # with decay stored to work
        if f'{prefix}decay' in pass_on_state_dict:
            del pass_on_state_dict[f'{prefix}decay']
        super()._load_from_state_dict(
            pass_on_state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.cpu_shadow_params:
            for name, buf in self.cpu_shadow_params[0].items():
                if buf is not None:
                    destination[prefix + name] = (
                        buf if keep_vars else buf.detach()
                    )
