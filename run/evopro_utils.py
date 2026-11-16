import os
import json
import pathlib
import typing
from typing import Sequence, Union, Callable, Tuple, Dict, Any
from functools import partial
import textwrap
import jax
import tokamax

from alphafold3.model.post_processing import post_process_inference_result
from alphafold3.model.config_factory import make_model_config
from alphafold3.model import model_config, model_registry, weights


from af3_utils import load_fold_inputs_from_path, get_af3_args
from run_af3 import ModelRunner, predict_structure, _resolve_jax_backend, JaxBackend, _precompile
from run_af3 import _HOME_DIR, _DEFAULT_MODEL_DIR, _DEFAULT_DB_DIR

from absl import flags

import logging
logging.getLogger('tokamax').setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore', message='.*int64.*')
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

def init_af3(proc_id: int, arg_file: str, lengths: Sequence[Union[int, Sequence[int]]]) -> Callable:
    # Work around for a known XLA issue:
    # https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#compilation-time-workaround-with-xla-flags
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
    
    args_dict = get_af3_args(arg_file)
    flags.FLAGS.mark_as_parsed()
    print(f"initializing AF3 with input dict {args_dict}")
    for flag_name, new_value in args_dict.items():
        if hasattr(flags.FLAGS, flag_name):
            setattr(flags.FLAGS, flag_name, new_value)
        else:
            print(f"Warning: Flag '{flag_name}' is not registered.")

    jax_backend = _resolve_jax_backend() if flags.FLAGS.run_inference else None
    if flags.FLAGS.run_inference and flags.FLAGS.flash_attention_implementation == 'auto':
      if jax_backend == JaxBackend.CPU:
        flags.FLAGS.flash_attention_implementation = 'xla'
        print('--flash_attention_implementation=auto -> xla (no GPU)')
      else:
        from alphafold3.model.components import platform as _platform
        _picked = _platform.attention_config()
        flags.FLAGS.flash_attention_implementation = _picked['attention']
        print(f'--flash_attention_implementation=auto -> '
              f'{_picked["attention"]}: {_picked["reason"]}')
        if flags.FLAGS.glu_kernel == 'auto' and _picked.get('glu'):
          flags.FLAGS.glu_kernel = _picked['glu']
          print(f'--glu_kernel=auto -> {_picked["glu"]}')
    if flags.FLAGS.run_inference:
      if jax_backend == JaxBackend.CPU:
        if flags.FLAGS.flash_attention_implementation != 'xla':
          if flags.FLAGS.jax_backend == JaxBackend.AUTO:
            print(
                'Setting --flash_attention_implementation=xla, required for'
                ' CPU-only inference.'
            )
            flags.FLAGS.flash_attention_implementation = 'xla'
          else:
            raise ValueError(
                'For CPU-only inference, the --flash_attention_implementation'
                ' must be set to "xla".'
            )
      elif jax_backend == JaxBackend.TPU:
        pass
      elif jax_backend == JaxBackend.GPU:
        gpu_devices = jax.local_devices(backend='gpu')
        if gpu_devices:
          compute_capability = float(
              gpu_devices[flags.FLAGS.gpu_device].compute_capability
          )
          if compute_capability < 6.0:
            raise ValueError(
                'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
                ' https://developer.nvidia.com/cuda-gpus).'
            )
          elif 7.0 <= compute_capability < 8.0:
            xla_flags = os.environ.get('XLA_FLAGS')
            required_flag = (
                '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
            )
            if not xla_flags or required_flag not in xla_flags:
              raise ValueError(
                  'For devices with GPU compute capability 7.x (see'
                  ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS'
                  f' must include "{required_flag}".'
              )
            if flags.FLAGS.flash_attention_implementation not in ('xla', 'volta'):
              raise ValueError(
                  'For devices with GPU compute capability 7.x (see'
                  ' https://developer.nvidia.com/cuda-gpus) the'
                  ' --flash_attention_implementation must be "xla", or "volta"'
                  ' with colabfold-legacy-kernels installed.'
              )
      else:
        raise ValueError(f'Unsupported JAX backend: {jax_backend}')
       
    model_name = flags.FLAGS.model
    for flag, var in ((flags.FLAGS.bioemu2_mode, 'BIOEMU2_MODE'), (flags.FLAGS.bioemu2_sampler, 'BIOEMU2_SAMPLER'),
                      (flags.FLAGS.bioemu2_potentials, 'BIOEMU2_POTENTIALS'), (flags.FLAGS.bioemu2_contacts, 'BIOEMU2_CONTACTS'),
                      (flags.FLAGS.bioemu2_propagate_from, 'BIOEMU2_X0'), (flags.FLAGS.bioemu2_lag_ns, 'BIOEMU2_LAG_NS'),
                      (flags.FLAGS.bioemu2_propagation_steps, 'BIOEMU2_STEPS'),
                      (flags.FLAGS.bioemu2_physicality, 'BIOEMU2_PHYSICALITY')):
      if flag is not None:
        if model_registry.get(model_name).name != 'bioemu2':
          raise ValueError(f'--{flag.name} applies to --model bioemu2 only')
        os.environ[var] = str(flag)
    cyclic = flags.FLAGS.cyclic or False
    if cyclic and [c.lower() for c in cyclic] == ['all']:
      cyclic = True
    wants_esm = model_name == 'chai1' or model_name in model_config.ESMFOLD2_FAMILY
    use_esm = wants_esm if flags.FLAGS.use_esm_embeddings is None else flags.FLAGS.use_esm_embeddings
    if use_esm and not wants_esm:
      raise ValueError(
          f'--use_esm_embeddings is for chai-1 and the ESMFold2 family; '
          f'{model_name} does not read a language model')
    if not use_esm and wants_esm:
      print('WARNING: %s is running WITHOUT its language model'
            ' (--nouse_esm_embeddings), which is part of the model rather than an'
            ' extra; see --use_esm_embeddings.' % model_name)
    if flags.FLAGS.model_dir is not None:
      model_dir = flags.FLAGS.model_dir
    elif model_name == 'alphafold3':
      model_dir = _DEFAULT_MODEL_DIR
    else:
      model_dir = None  # resolved by ensure_weights below
  
    if model_name == 'alphafold3':
      notice = textwrap.wrap(
          'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
          ' parameters are only available under terms of use provided at'
          ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
          ' If you do not agree to these terms and are using AlphaFold 3 derived'
          ' model parameters, cancel execution of AlphaFold 3 inference with'
          ' CTRL-C, and do not use the model parameters.',
          break_long_words=False,
          break_on_hyphens=False,
          width=80,
      )
      print('\n' + '\n'.join(notice) + '\n')
  
    if True:
      devices = jax.local_devices(backend=jax_backend)
      if jax_backend == JaxBackend.CPU:
        device = devices[0]
        print(f'Found local CPU devices: {devices}, using device 0: {device}')
      elif jax_backend == JaxBackend.TPU:
        device = devices[0]
        print(f'Found local TPU devices: {devices}, using device 0: {device}')
      elif jax_backend == JaxBackend.GPU:
        print(
            f'Found local GPU devices: {devices}, using device '
            f'{flags.FLAGS.gpu_device}: {devices[flags.FLAGS.gpu_device]}'
        )
        device = devices[flags.FLAGS.gpu_device]
      else:
        raise ValueError(f'Unsupported JAX backend: {jax_backend}')
  
      model_dir = weights.ensure_weights(
          model_name, model_dir, download=flags.FLAGS.download_weights,
          precision=flags.FLAGS.weights_precision)
  
      print('Building model from scratch...')
      model_runner = ModelRunner(
          config=make_model_config(
              flash_attention_implementation=typing.cast(
                  tokamax.DotProductAttentionImplementation,
                  flags.FLAGS.flash_attention_implementation,
              ),
              glu_kernel=flags.FLAGS.glu_kernel,
              num_diffusion_samples=flags.FLAGS.num_diffusion_samples,
              num_sampling_steps=flags.FLAGS.num_sampling_steps,
              num_recycles=flags.FLAGS.num_recycles,
              return_embeddings=flags.FLAGS.save_embeddings,
              return_distogram=flags.FLAGS.save_distogram,
              model_name=model_name,
              num_msa=flags.FLAGS.num_msa,
          ),
          device=device,
          model_dir=model_dir,
          use_dropout=flags.FLAGS.dropout,
      )
      # Check we can load the model parameters before launching anything.
      print('Checking that model parameters can be loaded...')
      #_ = model_runner.model_params
      #_precompile(model_runner, model_name, model_dir,
      #              [int(n) for n in lengths])
   
    # Determine max length of sequences to use as bucket for model compilation
    max_length = 0
    for length in lengths:
        if not isinstance(length, int):
            length = max(length)
        if length > max_length:
            max_length = length
    buckets = (max_length,)

    # Use max_length to create a fake fold_input
    json_dict = {
        "name": "compilation_noodle",
        "sequences": [{
            "protein": {
                "id": ["A"],
                "sequence": "G" * max_length
            }
        }],
        "modelSeeds": ['42']
    }
    json_str = json.dumps(json_dict)
    fold_input = [i for i in load_fold_inputs_from_path(json_str)][0]

    # Make folding prediction
    _ = model_runner.model_params
    _ = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner, 
        buckets=buckets,
        use_esm=use_esm,
    )

    return partial(run_af3, proc_id=proc_id, arg_file=arg_file, buckets=buckets, compiled_runner=model_runner)


def run_af3(json_str: str, proc_id: int, arg_file: str, buckets: Tuple[int], compiled_runner: ModelRunner) -> Sequence[Dict[str, Any]]:
    args_dict = get_af3_args(arg_file)
    
    if args_dict['jax_compilation_cache_dir'] is not None:
        jax.config.update(
            'jax_compilation_cache_dir', args_dict['jax_compilation_cache_dir']
        )

    wants_esm = args_dict["model"] == 'chai1' or args_dict["model"] in model_config.ESMFOLD2_FAMILY
    use_esm = wants_esm if args_dict["use_esm_embeddings"] is None else args_dict["use_esm_embeddings"]

    # Convert json_str to fold_input and make prediction
    fold_input = [i for i in load_fold_inputs_from_path(json_str)][0]
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=compiled_runner, 
        buckets=buckets,
        use_esm=use_esm
    )

    # Convert results into list of dict for output
    processed_results = [
        post_process_inference_result(result.inference_results[0])
        for result in all_inference_results
    ]
    results_list = [
        {
            "seed": all_inference_results[i].seed,
            "cif_str": result.cif.decode("utf-8"),
            **json.loads(result.structure_confidence_summary_json.decode("utf-8")),
            **json.loads(result.structure_full_data_json.decode("utf-8")),

        }
        for i, result in enumerate(processed_results)
    ]

    return results_list
