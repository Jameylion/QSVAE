"""
Measure translation between hardware and software model
"""
from typing import Optional, Tuple, NamedTuple
import pylogging as logger
from tqdm import tqdm

import torch
from scipy.optimize import curve_fit

from dlens_vx_v3 import halco, lola
import hxtorch
from hxtorch import spiking
import hxtorch.spiking.functional as F

log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.INFO)


class Thresholds:

    def __init__(self, calib_path: Optional[str] = None, neuron_size: int = 1, 
                 batch_size: int = 100,
                 output_size: int = halco.AtomicNeuronOnDLS.size):
        self.calib_path = calib_path
        self.batch_size = batch_size
        self.time_length = 100
        self.output_size = int(output_size // neuron_size)
        self.neuron_size = neuron_size
        self.hw_neurons = None

    def build_model(self, inputs: torch.Tensor, exp: spiking.Experiment) \
            -> spiking.TensorHandle:
        """ Build model to map to hardware """
        # Layers
        synapse = spiking.Synapse(
            1, self.output_size, experiment=exp)
        self.neuron = spiking.Neuron(
            self.output_size, experiment=exp, func=spiking.functional.LIF,
            neuron_structure=spiking.morphology.SingleCompartmentNeuron(
                size=self.neuron_size, expand_horizontally=False),
            shift_cadc_to_first=False)
        # forward
        inputs = spiking.NeuronHandle(spikes=inputs)
        currents = synapse(inputs)
        traces = self.neuron(currents)
        return traces

    def run(self, dt: float = 1e-6):
        """ Execute forward """
        hxtorch.init_hardware()

        # Baseline
        exp = spiking.Experiment(mock=False, dt=dt, calib_path=self.calib_path)
        inputs = torch.zeros((self.time_length, self.batch_size, 1))
        baselines = self.build_model(inputs, exp)
        spiking.run(exp, self.time_length)

        # Instance
        exp = spiking.Experiment(mock=False, dt=dt, calib_path=self.calib_path)
        inputs = torch.zeros(
            (self.time_length, self.batch_size, 1))
        traces = self.build_model(inputs, exp)
        # Set initial config
        for nrn in halco.iter_all(halco.AtomicNeuronOnDLS):
            exp._chip.neuron_block.atomic_neurons[nrn] \
                .constant_current.i_offset = 1000
            exp._chip.neuron_block.atomic_neurons[nrn] \
                .constant_current.enable = True
        # run
        spiking.run(exp, self.time_length)
        hxtorch.release_hardware()

        return self.post_process(traces, baselines)

    def post_process(self, traces: torch.Tensor, baselines: torch.Tensor) \
            -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        baselines = baselines.v_cadc.detach().mean(0).mean(0)
        traces = traces.v_cadc.detach() - baselines
        thresholds = torch.max(traces, 0)[0].mean(0)

        return thresholds.mean()


class WeightScaling:
    """ Measure weight scaling """

    max_weight = lola.SynapseWeightMatrix.Value.max
    min_weight = -lola.SynapseWeightMatrix.Value.max

    def __init__(self, params: NamedTuple, calib_path: str,
                 neuron_size: int = 1, batch_size: int = 100,
                 trace_scale: float = 1.,
                 output_size: int = halco.AtomicNeuronOnDLS.size):
        self.calib_path = calib_path
        self.batch_size = batch_size
        self.time_length = 100
        self.output_size = int(output_size // neuron_size)
        self.hw_neurons = None
        self.neuron_size = neuron_size
        self.params = params
        self.trace_scale = trace_scale
        self.log = logger.get("HXSoace.WeightScaling")

    # pylint: disable=arguments-differ, attribute-defined-outside-init
    def execute(self, weight: int) -> torch.Tensor:
        # Instance
        inputs = torch.zeros(self.time_length, self.batch_size, 1)
        inputs[10, :, :] = 1
        self.synapse.weight.data.fill_(weight)

        # forward
        spikes = spiking.NeuronHandle(spikes=inputs)
        currents = self.synapse(spikes)
        traces = self.neuron(currents)

        spiking.run(self.exp, self.time_length)

        return self.post_process(traces, weight)

    def post_process(self, traces: torch.Tensor, weight: float) \
            -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.traces = traces.v_cadc.detach()
        # Get max/min PSP over time
        if weight >= 0:
            self.amp = self.traces.max(0)[0].mean(0)
        else:
            self.amp = self.traces.min(0)[0].mean(0)
        self.amp_mean = self.amp.mean()
        return self.amp_mean, self.amp, self.traces

    # pylint: disable=arguments-differ, too-many-arguments, too-many-locals
    def run(
            self, weight_step: int = 1) \
            -> Tuple[torch.Tensor, ...]:
        """ """
        # Measure thresholds
        self.log.INFO("Measure thresholds...")
        handler = Thresholds(self.calib_path)
        thresholds = handler.run()

        # Sweep weights
        self.log.INFO(f"Using weight step: {weight_step}")
        hw_weights = torch.linspace(
            self.min_weight, self.max_weight, weight_step, dtype=int)

        # Hardware amplitudes
        hw_amps = torch.zeros(hw_weights.shape[0], self.output_size)

        hxtorch.init_hardware()
        self.exp = spiking.Experiment(
            mock=False, dt=1e-6, calib_path=self.calib_path)
        self.synapse = spiking.Synapse(
            1, self.output_size, experiment=self.exp)
        self.neuron = spiking.ReadoutNeuron(
            self.output_size, experiment=self.exp,
            trace_scale=self.trace_scale,
            neuron_structure=spiking.morphology.SingleCompartmentNeuron(
                size=self.neuron_size, expand_horizontally=False),
            func=F.cuba_li_integration, shift_cadc_to_first=True,
            params=self.params)

        # Sweep
        self.log.INFO("Begin hardware weight sweep...")
        pbar = tqdm(total=hw_weights.shape[0])
        for i, weight in enumerate(hw_weights):
            # Measure
            _, hw_amps[i], _ = self.execute(weight)
            # Update
            pbar.set_postfix(
                weight=f"{weight}", mean_amp=float(hw_amps[i].mean()))
            pbar.update()
        pbar.close()
        self.log.INFO("Hardware weight sweep finished.")

        # Fit
        self.log.INFO("Fit hardware data...")
        hw_scales = torch.zeros(self.output_size)
        for nrn in range(self.output_size):
            popt, _ = curve_fit(
                f=lambda x, a: a * x, xdata=hw_weights.numpy(),
                ydata=hw_amps[:, nrn].numpy())
            hw_scales[nrn] = popt[0]

        # Mock values
        self.exp = spiking.Experiment(
            mock=True, dt=1e-6, calib_path=self.calib_path)
        self.synapse = spiking.Synapse(1, 1, experiment=self.exp)
        self.neuron = spiking.ReadoutNeuron(
            1, experiment=self.exp, trace_scale=self.trace_scale,
            func=F.cuba_li_integration, shift_cadc_to_first=True,
            params=self.params)

        self.log.INFO("Begin mock weight sweep...")
        self.output_size = 1
        sw_weights = torch.arange(-1, 1 + 0.1, 0.1)
        # Hardware amplitudes
        sw_amps = torch.zeros(sw_weights.shape[0], 1)
        pbar = tqdm(total=sw_weights.shape[0])
        for i, weight in enumerate(sw_weights):
            # Measure
            _, sw_amps[i], _ = self.execute(weight)
            pbar.set_postfix(
                weight=f"{weight}", mean_amp=float(sw_amps[i].mean()))
            pbar.update()
        pbar.close()

        # SW scale
        popt, _ = curve_fit(
            f=lambda x, a: a * x, xdata=sw_weights.numpy(),
            ydata=sw_amps.reshape(-1).numpy())
        sw_scales = popt[0]

        # Resulting scales
        scales = sw_scales / hw_scales

        self.log.INFO(
            f"Mock scale: {sw_scales}, HW scale: {hw_scales.mean()}"
            + f" +- {hw_scales.std()}")
        self.log.INFO(f"SW -> HW translation factor: {scales.mean()}")

        hxtorch.release_hardware()

        return scales.mean() * thresholds / self.params.v_th


def get_trace_scaling(sw_threshold, neuron_size, calib_path) -> float:
    thresholds = Thresholds(calib_path, neuron_size=neuron_size)
    hw_threshold = thresholds.run()
    return sw_threshold / hw_threshold.item()


def get_weight_scaling(
        tau_mem, tau_syn, neuron_size, calib_path, v_th_sw: float = 1.,
        v_leak_sw: float = 0., weight_step: int = 10) -> float:
    params = F.CUBALIFParams(
        tau_syn_inv=1 / tau_syn, tau_mem_inv=1 / tau_mem, v_leak=v_leak_sw, v_th=v_th_sw)
    scaling = WeightScaling(
        params, calib_path, neuron_size=neuron_size, batch_size=100,
        output_size=halco.AtomicNeuronOnDLS.size)
    return scaling.run(weight_step=weight_step).item()