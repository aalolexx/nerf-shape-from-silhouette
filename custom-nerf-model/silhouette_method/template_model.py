"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model 
# from template_field import CustomHashMLPDensityField # custom 


@dataclass
class TemplateModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateModel)


class TemplateModel(NerfactoModel):
    """Template Model."""

    config: TemplateModelConfig

    def populate_modules(self):
        super().populate_modules()
        
        # # Override the default behavior to use the custom density field
        # self.density_fns = []
        # num_prop_nets = self.config.num_proposal_iterations

        # # Build the proposal network(s)
        # self.proposal_networks = torch.nn.ModuleList()
        # if self.config.use_same_proposal_network:
        #     assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
        #     prop_net_args = self.config.proposal_net_args_list[0]
        #     network = CustomHashMLPDensityField(
        #         self.scene_box.aabb,
        #         spatial_distortion=self.scene_contraction,
        #         **prop_net_args,
        #         average_init_density=self.config.average_init_density,
        #         implementation=self.config.implementation,
        #     )
        #     self.proposal_networks.append(network)
        #     self.density_fns.extend([network.get_density for _ in range(num_prop_nets)])
        # else:
        #     for i in range(num_prop_nets):
        #         prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
        #         network = CustomHashMLPDensityField(
        #             self.scene_box.aabb,
        #             spatial_distortion=self.scene_contraction,
        #             **prop_net_args,
        #             average_init_density=self.config.average_init_density,
        #             implementation=self.config.implementation,
        #         )
        #         self.proposal_networks.append(network)
        #     self.density_fns.extend([network.get_density for network in self.proposal_networks])

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
