from typing import Optional, Tuple
from functools import partial
import torch

from dlens_vx_v3 import halco

import hxtorch.spiking as hxsnn
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.utils import calib_helper
import hxtorch.spiking.functional as F

import matplotlib.pyplot as plt
from snntorch import spikeplot as splt


class SNN(torch.nn.Module):
    """
    SNN with one hidden LIF layer and one readout LI layer
    """
    # pylint: disable=too-many-arguments, invalid-name

    def __init__(self, n_in: int, n_hidden: int, n_out: int, mock: bool,
                 calib_path: str, dt: float = 1.0e-6, tau_mem: float = 8e-6,
                 tau_syn: float = 8e-6, alpha: float = 50,
                 trace_shift_hidden: int = 0, trace_shift_out: int = 0,
                 weight_init_hidden: Optional[Tuple[float, float]] = None,
                 weight_init_output: Optional[Tuple[float, float]] = None,
                 weight_scale: float = 1., trace_scale: float = 1.,
                 input_repetitions: int = 1,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Initialize the SNN.

        :param n_in: Number of input units.
        :param n_hidden: Number of hidden units.
        :param n_out: Number of output units.
        :param mock: Indicating whether to train in software or on hardware.
        :param calib_path: Path to hardware calibration file.
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
            tau_mem_inv=1. / tau_mem, tau_syn_inv=1. / tau_syn, v_th=torch.tensor(0.005), alpha=alpha)
        li_params = F.CUBALIParams(
            tau_mem_inv=1. / tau_mem, tau_syn_inv=1. / tau_syn)

        # Experiment instance to work on
        self.exp = hxsnn.Experiment(
            mock=mock, dt=dt)
        if not mock:
            self.exp.default_execution_instance.load_calib(
                calib_path if calib_path else calib_helper.nightly_calib_path(
                    "spiking2"))

        # Repeat input
        self.input_repetitions = input_repetitions
        # Input projection
        self.linear_h = hxsnn.Synapse(
            n_in * input_repetitions, n_hidden, experiment=self.exp,
            transform=partial(
                weight_transforms.linear_saturating, scale=weight_scale))
        # Initialize weights
        if weight_init_hidden:
            w = torch.zeros(n_hidden, n_in)
            torch.nn.init.normal_(w, *weight_init_hidden)
            self.linear_h.weight.data = w.repeat(1, input_repetitions)

        # Hidden layer
        self.lif_h = hxsnn.Neuron(
            n_hidden, experiment=self.exp, func=F.cuba_lif_integration,
            params=lif_params, trace_scale=trace_scale,
            cadc_time_shift=trace_shift_hidden, shift_cadc_to_first=True)

        # Output projection
        self.linear_o = hxsnn.Synapse(
            n_hidden, n_out, experiment=self.exp,
            transform=partial(
                weight_transforms.linear_saturating, scale=weight_scale))

        # Readout layer
        self.li_readout = hxsnn.Neuron(
            n_out, experiment=self.exp, func=F.cuba_lif_integration,
            params=lif_params, trace_scale=trace_scale,
            cadc_time_shift=trace_shift_hidden, shift_cadc_to_first=True)
        
#         hxsnn.ReadoutNeuron(
#             n_out, experiment=self.exp, func=F.cuba_li_integration,
#             params=li_params, trace_scale=trace_scale,
#             cadc_time_shift=trace_shift_out, shift_cadc_to_first=True,
#             placement_constraint=list(
#                 halco.LogicalNeuronOnDLS(
#                     hxsnn.morphology.SingleCompartmentNeuron(1).compartments,
#                     halco.AtomicNeuronOnDLS(
#                         halco.NeuronRowOnDLS(1), halco.NeuronColumnOnDLS(nrn)))
#                 for nrn in range(n_out)))
        
        # Initialize weights
        if weight_init_output:
            torch.nn.init.normal_(self.linear_o.weight, *weight_init_output)

        # Device
        self.device = device
        self.to(device)

        # placeholder for hidden spikes
        self.s_h = None

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward path.

        :param spikes: NeuronHandle holding spikes as input.

        :return: Returns the output of the network, i.e. membrane traces of the
            readout neurons.
        """
        # print(spikes)
        # print(spikes.shape)
        # spikes = torch.flatten(spikes)
        # print(spikes)
        # print(spikes.shape)
        # Increase synapse strength by repeating each input
        # spikes = spikes.repeat(1, 1, self.input_repetitions)

        # print(spikes)
        # print(spikes.shape)
        
        # Spike input handle
        spikes_handle = hxsnn.NeuronHandle(spikes)
        # print(spikes_handle)
        # print(spikes_handle.spikes)

        # Forward
        c_h = self.linear_h(spikes_handle)
        self.s_h = self.lif_h(c_h)  # Keep spikes for fire reg.
        # print(self.s_h)
        c_o = self.linear_o(self.s_h)
        y_o = self.li_readout(c_o)

        # Execute on hardware
        hxsnn.run(self.exp, spikes.shape[0])
        # print("end of forward snn")
        # print(y_o.v_cadc)
        # print(type(c_o))
        # print(c_h)
        print("spikes input", spikes.sum((0,1)))
        print("spikes hid shape", self.s_h.spikes.shape) 
        print("hidden spike count", self.s_h.spikes.sum((0,1)))
        print("output spike count", y_o.spikes.sum((0,1)))
        plot_cur_mem_spk(self.s_h.v_cadc, self.s_h.v_cadc, self.s_h.spikes, thr_line=True, vline=False, title=False,
                     ylim_max1=1.25, ylim_max2=1.25, neuron_index=torch.argmax(self.s_h.spikes.sum(0)))

        plot_cur_mem_spk(y_o.v_cadc, y_o.v_cadc, y_o.spikes, thr_line=True, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25, neuron_index=torch.argmax(y_o.spikes.sum(0)))


        return y_o.spikes, y_o.v_cadc

def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False,
                     ylim_max1=1.25, ylim_max2=1.25, neuron_index=0):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})
    # Select data for the specified output neuron
    # print(mem.shape[2])
    # _,j,i = torch.unravel_index(neuron_index, spk.shape)
    # r = torch.remainder(neuron_index, spk.shape[2])
    j = neuron_index //spk.shape[2]
    i = neuron_index % spk.shape[2]

    cur = cur[:, j, i].detach()  
    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    ax[0].set_ylim([0, ylim_max1])
    ax[0].set_xlim([0, 200])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    if title:
        ax[0].set_title(title)


    # print(j)
    mem = mem[:, j, i].detach()  # Select neuron data from the first batch
    spk= spk[:, j, i].detach()  # Select neuron data from the first batch

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([0, ylim_max2])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    if thr_line:
        ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[2], s=400, c="black", marker="|")
    if vline:
        ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    plt.ylabel("Output spikes")
    plt.yticks([])

    plt.show()