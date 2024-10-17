from typing import Tuple
import matplotlib.pyplot as plt
import ipywidgets as w
import numpy as np
import torch
from src._static.common.helpers import setup_hardware_client, save_nightly_calibration
from src._static.tutorial.snn_yinyang_helpers import plot_data, plot_input_encoding, plot_training
setup_hardware_client()
from functools import partial
import hxtorch
import hxtorch.snn as hxsnn
import hxtorch.snn.functional as F
from hxtorch.snn.transforms import weight_transforms
from dlens_vx_v3 import halco

class SNN(torch.nn.Module):
    """ SNN with one hidden LIF layer and one readout LI layer """

    def __init__(
            self,
            n_in: int,
            n_hidden: int,
            n_out: int,
            mock: bool,
            dt: float,
            tau_mem: float,
            tau_syn: float,
            alpha: float,
            trace_shift_hidden: int,
            trace_shift_out: int,
            weight_init_hidden: Tuple[float, float],
            weight_init_output: Tuple[float, float],
            weight_scale: float,
            trace_scale: float,
            input_repetitions: int,
            device: torch.device):
        """
        :param n_in: Number of input units.
        :param n_hidden: Number of hidden units.
        :param n_out: Number of output units.
        :param mock: Indicating whether to train in software or on hardware.
        :param dt: Time-binning width.
        :param tau_mem: Membrane time constant.
        :param tau_syn: Synaptic time constant.
        :param trace_shift_hidden: Indicates how many indices the membrane
            trace of hidden layer is shifted to left along time axis.
        :param trace_shift_out: Indicates how many indices the membrane
            trace of readout layer is shifted to left along time axis.
        :param weight_init_hidden: Hidden layer weight initialization mean
            and std value.
        :param weight_init_output: Output layer weight initialization mean
            and std value.
        :param weight_scale: The factor with which the software weights are
            scaled when mapped to hardware.
        :param input_repetitions: Number of times to repeat input channels.
        :param device: The used PyTorch device used for tensor operations in
            software.
        """
        super().__init__()

        # Neuron parameters
        lif_params = F.CUBALIFParams(
            1. / tau_mem, 1. / tau_syn, alpha=alpha)
        li_params = F.CUBALIParams(1. / tau_mem, 1. / tau_syn)
        self.dt = dt

        # Instance to work on

        if not mock:
            save_nightly_calibration('spiking2_cocolist.pbin')
            self.experiment = hxsnn.Experiment(mock=mock, dt=dt) #,calib_path='spiking2_cocolist.pbin'
        else:
            self.experiment = hxsnn.Experiment(mock=mock, dt=dt)

        # Repeat input
        self.input_repetitions = input_repetitions

        # Input projection
        self.linear_h = hxsnn.Synapse(
            n_in * input_repetitions,
            n_hidden,
            experiment=self.experiment,
            transform=partial(
                weight_transforms.linear_saturating, scale=weight_scale))

        # Initialize weights
        if weight_init_hidden:
            w = torch.zeros(n_hidden, n_in)
            torch.nn.init.normal_(w, *weight_init_hidden)
            self.linear_h.weight.data = w.repeat(1, input_repetitions)

        # Hidden layer
        self.lif_h = hxsnn.Neuron(
            n_hidden,
            experiment=self.experiment,
            func=F.cuba_lif_integration,
            params=lif_params,
            trace_scale=trace_scale,
            cadc_time_shift=trace_shift_hidden,
            shift_cadc_to_first=True)

        # Output projection
        self.linear_o = hxsnn.Synapse(
            n_hidden,
            n_out,
            experiment=self.experiment,
            transform=partial(
                weight_transforms.linear_saturating, scale=weight_scale))

        # Readout layer
        self.li_readout = hxsnn.ReadoutNeuron(
            n_out,
            experiment=self.experiment,
            func=F.cuba_li_integration,
            params=li_params,
            trace_scale=trace_scale,
            cadc_time_shift=trace_shift_out,
            shift_cadc_to_first=True,
            placement_constraint=list(
                halco.LogicalNeuronOnDLS(
                    hxsnn.morphology.SingleCompartmentNeuron(1).compartments,
                    halco.AtomicNeuronOnDLS(
                        halco.NeuronRowOnDLS(1), halco.NeuronColumnOnDLS(nrn)))
                for nrn in range(n_out)))

        # Initialize weights
        if weight_init_output:
            torch.nn.init.normal_(self.linear_o.weight, *weight_init_output)

        # Device
        self.device = device
        self.to(device)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward path.
        :param spikes: NeuronHandle holding spikes as input.
        :return: Returns the output of the network, i.e. membrane traces of the
        readout neurons.
        """
        # Remember input spikes for plotting
        self.s_in = spikes
        # Increase synapse strength by repeating each input
        spikes = spikes.repeat(1, 1, self.input_repetitions)
        # Spike input handle
        spikes_handle = hxsnn.NeuronHandle(spikes)

        # Forward
        c_h = self.linear_h(spikes_handle)
        self.s_h = self.lif_h(c_h)  # Keep spikes for fire reg.
        c_o = self.linear_o(self.s_h)
        self.y_o = self.li_readout(c_o)

        # Execute on hardware
        hxtorch.snn.run(self.experiment, spikes.shape[0])

        return self.y_o.v_cadc