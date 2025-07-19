import os
import re
import jax
import jax.numpy as jnp
import haiku as hk
jax.config.update("jax_enable_x64", True)

from hqc.pbc.pes import make_pes

from src.checkpoint import load_data, save_data
from src.inference.utils import *
from src.flow import make_flow
from src.ferminet import FermiNet
from src.inference.sample_s import sample_flow_s

def eval_s_sample_lcao(ckpt_sample_filename,
                       batchsize=None,
                       flow_batchsize=2048,
                       lcao_batchsize=256,
                       auto_load=True,
                       auto_save=True,
                       silent_mode=False,
                       ):
    """
        Evaluate the sample quantity data for a given s samples.
        Energy unit: Ry
        Pressure unit: GPa
    """
    index = ckpt_sample_filename.index(".pkl")
    file_basename = os.path.basename(ckpt_sample_filename)
    if "_sample_s" in file_basename:
        if "_quantity" in file_basename:
            ckpt_sample_quantity_filename = ckpt_sample_filename
        else:
            ckpt_sample_quantity_filename = ckpt_sample_filename[:index] + "_quantity" + ".pkl"
    else:
        if batchsize is None:
            raise ValueError('Batch size must be provided for new sample file')
        new_ckpt_sample_filename = ckpt_sample_filename[:index] + f"_sample_s_{batchsize}.pkl"
        ckpt_sample_quantity_filename = ckpt_sample_filename[:index] + f"_sample_s_{batchsize}_quantity.pkl"
        if not os.path.isfile(new_ckpt_sample_filename) and not os.path.isfile(ckpt_sample_quantity_filename):
            if not silent_mode:
                print(f"{RED}Find no checkpoint sample file, drawing new samples...{RESET}")
            sample_flow_s([ckpt_sample_filename], sample_batch=batchsize, sample_flow_batch=flow_batchsize)
            ckpt_sample_filename = new_ckpt_sample_filename

    # auto load sample quantity
    if auto_load:
        if os.path.isfile(ckpt_sample_quantity_filename):
            pattern = r'.*_sample_s_bs_(\d+).*'
            match = re.search(pattern, ckpt_sample_quantity_filename)
            if match:
                load_batchsize_filename = int(match.group(1))
                if batchsize is not None:
                    if load_batchsize_filename != batchsize:
                        raise ValueError(f'Batch size {batchsize} does not match the batch size in file name {load_batchsize_filename}')
            if not silent_mode:
                print(f"{YELLOW}========= Loading sample quantity data ========{RESET}")
            data = load_data(ckpt_sample_quantity_filename)
            if not silent_mode:
                print(f'{MAGENTA}Loaded sample quantity data from:{RESET} {ckpt_sample_quantity_filename}')
                print(f'{MAGENTA}Loaded s samples shape:{RESET} {data["s"].shape}')
                print(f'{MAGENTA}Loaded F (Ry):{RESET} {data["F_ave"]} ± {data["F_std"]}')
                print(f'{MAGENTA}Loaded Etot (Ry):{RESET} {data["Etot_ave"]} ± {data["Etot_std"]}')
                print(f'{MAGENTA}Loaded Eelec (Ry):{RESET} {data["Eelec_ave"]} ± {data["Eelec_std"]}')
                print(f'{MAGENTA}Loaded Ki (Ry):{RESET} {data["Ki_ave"]} ± {data["Ki_std"]}')
                print(f'{MAGENTA}Loaded Vep (Ry):{RESET} {data["Vep_ave"]} ± {data["Vep_std"]}')
                print(f'{MAGENTA}Loaded Vee (Ry):{RESET} {data["Vee_ave"]} ± {data["Vee_std"]}')
                print(f'{MAGENTA}Loaded Vpp (Ry):{RESET} {data["Vpp_ave"]} ± {data["Vpp_std"]}')
                print(f'{MAGENTA}Loaded Sp:{RESET} {data["Sp_ave"]} ± {data["Sp_std"]}')
                print(f'{MAGENTA}Loaded Se:{RESET} {data["Se_ave"]} ± {data["Se_std"]}')
                print(f'{MAGENTA}Loaded P (GPa):{RESET} {data["P_ave"]} ± {data["P_std"]}')
                print(f'{MAGENTA}Loaded EPtot_cov (Ha*GPa):{RESET} {data["EPtot_cov"]}')
                print(f'{MAGENTA}Loaded Converged:{RESET} {data["converged_ave"]}')

            return data

    directory = os.path.dirname(ckpt_sample_filename)
    cfg_filename = os.path.join(directory, 'config.yaml')
    if os.path.isfile(cfg_filename):
        cfg = OmegaConf.load(cfg_filename)
    else:
        raise FileNotFoundError(f'Config file {cfg_filename} not found')

    key = jax.random.PRNGKey(42)
    reciprocal_beta = cfg.T/157888.088922572 # temperature in unit of Ry
    smearing_sigma = reciprocal_beta/2 # temperature in Hartree unit
    L = (4/3*jnp.pi*cfg.num)**(1/3)

    pes = make_pes(cfg.num, L, cfg.rs, cfg.pes.basis, rcut=cfg.pes.rcut, tol=cfg.pes.tol, 
                max_cycle=cfg.pes.max_cycle, grid_length=cfg.pes.grid_length, diis=cfg.pes.diis.diis, 
                diis_space=cfg.pes.diis.space, diis_start_cycle=cfg.pes.diis.start_cycle, 
                diis_damp=cfg.pes.diis.damp, use_jit=cfg.pes.use_jit, dft=(cfg.pes.type=="dft"), 
                xc=cfg.pes.xc, smearing=cfg.pes.smearing.smearing, smearing_method=cfg.pes.smearing.method, 
                smearing_sigma=smearing_sigma, search_method=cfg.pes.smearing.search.method, 
                search_cycle=cfg.pes.smearing.search.cycle, search_tol=cfg.pes.smearing.search.tol,
                gamma=cfg.pes.gamma, Gmax=cfg.pes.Gmax, kappa=cfg.pes.kappa, mode='dev')
    pes_vmap = jax.vmap(pes, 0, (0, 0, 0, 0, 0, 0, 0))
    pes_vmap_map = lambda xp: jax.lax.map(pes_vmap, xp)
    pes_vmap_map_pmap = jax.pmap(pes_vmap_map, axis_name='p')

    if not silent_mode:
        print(f"{YELLOW}========= Loading checkpoint ========{RESET}")
    data = load_data(ckpt_sample_filename)

    # load s samples
    s = data['s']
    load_s_batchsize = s.shape[0]
    if not silent_mode:
        print(f'{MAGENTA}Loaded s samples from:{RESET} {ckpt_sample_filename}')
        print(f'{MAGENTA}Loaded s samples shape:{RESET} {s.shape}')
    
    # evaluate s samples
    num_devices = jax.device_count()
    if not silent_mode:
        print(f"{YELLOW}========= Evaluating s samples ========{RESET}")
        print(f'{MAGENTA}Evaluating logp...{RESET}')
    
    # evaluate flow logp
    params_flow = data['params_flow']
    def forward_fn(x):
        for _ in range(cfg.flow.steps):
            model = FermiNet(cfg.flow.depth, cfg.flow.h1size, cfg.flow.h2size, cfg.flow.Nf, L, False, remat=cfg.flow.remat)
            x = model(x)
        return x
    network_flow = hk.transform(forward_fn)
    logprob_flow_novmap = make_flow(network_flow, cfg.num, cfg.dim, L)
    logprob_p = lambda xp: jax.vmap(logprob_flow_novmap, (None, 0), 0)(params_flow, xp)
    logprob_p_map = lambda xp: jax.lax.map(logprob_p, xp)
    logprob_map_pmap = jax.pmap(logprob_p_map, axis_name='p')
    # reshape s samples to flow batchsize
    flow_batch_per_device = flow_batchsize // num_devices
    if flow_batchsize % num_devices != 0:
        raise ValueError(f'Batch size {flow_batchsize} must be divisible by number of devices {num_devices}')
    if load_s_batchsize % flow_batchsize != 0:
        map_axis_size = load_s_batchsize // flow_batchsize + 1
        more_s = jax.random.uniform(key, (flow_batchsize*map_axis_size-load_s_batchsize, cfg.num, cfg.dim), minval=0, maxval=L)
        s_flow = jnp.concatenate([s, more_s], axis=0)
    else:
        map_axis_size = load_s_batchsize // flow_batchsize
        s_flow = s
    s_flow = s_flow.reshape(num_devices, map_axis_size, flow_batch_per_device, cfg.num, cfg.dim)
    if not silent_mode:
        print(f'{MAGENTA}flow s.shape{RESET} {s_flow.shape}')
    logp_p = logprob_map_pmap(s_flow)
    if not silent_mode:
        print(f'{MAGENTA}logp.shape{RESET} {logp_p.shape}')
    logp_p = logp_p.reshape(-1)[:load_s_batchsize]
    if not silent_mode:
        print(f'{MAGENTA}Reshaping...{RESET}')
        print(f'{MAGENTA}logp.shape{RESET} {logp_p.shape}')
    
    # evaluate pes
    if not silent_mode:
        print(f'{MAGENTA}Evaluating pes...{RESET}')
    # reshape s samples to lcao batchsize
    lcao_batch_per_device = lcao_batchsize // num_devices
    if lcao_batchsize % num_devices != 0:
        raise ValueError(f'Batch size {lcao_batchsize} must be divisible by number of devices {num_devices}')
    if load_s_batchsize % lcao_batchsize != 0:
        map_axis_size = load_s_batchsize // lcao_batchsize + 1
        more_s = jax.random.uniform(key, (lcao_batchsize*map_axis_size-load_s_batchsize, cfg.num, cfg.dim), minval=0, maxval=L)
        s_pes = jnp.concatenate([s, more_s], axis=0)
    else:
        map_axis_size = load_s_batchsize // lcao_batchsize
        s_pes = s
    s_pes = s_pes.reshape(num_devices, map_axis_size, lcao_batch_per_device, cfg.num, cfg.dim)
    if not silent_mode:
        print(f'{MAGENTA}PES s.shape{RESET} {s_pes.shape}')
    Etot, Ki, Vep, Vee, Vpp, Se, converged = pes_vmap_map_pmap(s_pes)
    if not silent_mode:
        print(f'{MAGENTA}Etot.shape{RESET} {Etot.shape}')
    Etot = Etot.reshape(-1)[:load_s_batchsize]*cfg.rs**2
    Ki = Ki.reshape(-1)[:load_s_batchsize]*cfg.rs**2
    Vep = Vep.reshape(-1)[:load_s_batchsize]*cfg.rs**2
    Vee = Vee.reshape(-1)[:load_s_batchsize]*cfg.rs**2
    Vpp = Vpp.reshape(-1)[:load_s_batchsize]*cfg.rs**2
    Se = Se.reshape(-1)[:load_s_batchsize]
    converged = converged.reshape(-1)[:load_s_batchsize]
    if not silent_mode:
        print(f'{MAGENTA}Reshaping...{RESET}')
        print(f'{MAGENTA}Etot.shape{RESET} {Etot.shape}')
    Eelec = Etot - Vpp
    Sp = -logp_p
    Floc = -(Sp+Se)*cfg.rs**2*reciprocal_beta + Etot
    P = (2*Ki + (Vpp + Vep + Vee) )/(3*(L*cfg.rs)**3)* 14710.513242194795 # GPa
    EPtot = Etot * P

    Floc = Floc/cfg.num/cfg.rs**2
    Etot = Etot/cfg.num/cfg.rs**2
    Eelec = Eelec/cfg.num/cfg.rs**2
    Ki = Ki/cfg.num/cfg.rs**2
    Vep = Vep/cfg.num/cfg.rs**2
    Vee = Vee/cfg.num/cfg.rs**2
    Vpp = Vpp/cfg.num/cfg.rs**2
    Sp = Sp/cfg.num
    Se = Se/cfg.num
    P = P/cfg.rs**2
    EPtot = EPtot/cfg.num/cfg.rs**4

    if not silent_mode:
        print(f"{YELLOW}========= Calculating average and std ========{RESET}")
    F_ave, F_std = jnp.mean(Floc), jnp.std(Floc)/jnp.sqrt(load_s_batchsize)
    Etot_ave, Etot_std  = jnp.mean(Etot), jnp.std(Etot)/jnp.sqrt(load_s_batchsize)
    Eelec_ave, Eelec_std = jnp.mean(Eelec), jnp.std(Eelec)/jnp.sqrt(load_s_batchsize)
    Ki_ave, Ki_std = jnp.mean(Ki), jnp.std(Ki)/jnp.sqrt(load_s_batchsize)
    Vep_ave, Vep_std = jnp.mean(Vep), jnp.std(Vep)/jnp.sqrt(load_s_batchsize)
    Vee_ave, Vee_std = jnp.mean(Vee), jnp.std(Vee)/jnp.sqrt(load_s_batchsize)
    Vpp_ave, Vpp_std = jnp.mean(Vpp), jnp.std(Vpp)/jnp.sqrt(load_s_batchsize)
    Sp_ave, Sp_std = jnp.mean(Sp), jnp.std(Sp)/jnp.sqrt(load_s_batchsize)
    Se_ave, Se_std = jnp.mean(Se), jnp.std(Se)/jnp.sqrt(load_s_batchsize)
    P_ave, P_std = jnp.mean(P), jnp.std(P)/jnp.sqrt(load_s_batchsize)
    EPtot_cov = (jnp.mean(EPtot)-Etot_ave*P_ave)/load_s_batchsize
    converged_ave = jnp.mean(converged)
    if not silent_mode:
        print(f"{MAGENTA}F (Ry):{RESET} {F_ave} ± {F_std}")
        print(f"{MAGENTA}Etot (Ry):{RESET} {Etot_ave} ± {Etot_std}")
        print(f"{MAGENTA}Eelec (Ry):{RESET} {Eelec_ave} ± {Eelec_std}")
        print(f"{MAGENTA}Ki (Ry):{RESET} {Ki_ave} ± {Ki_std}")
        print(f"{MAGENTA}Vep (Ry):{RESET} {Vep_ave} ± {Vep_std}")
        print(f"{MAGENTA}Vee (Ry):{RESET} {Vee_ave} ± {Vee_std}")
        print(f"{MAGENTA}Vpp (Ry):{RESET} {Vpp_ave} ± {Vpp_std}")
        print(f"{MAGENTA}Sp:{RESET} {Sp_ave} ± {Sp_std}")
        print(f"{MAGENTA}Se:{RESET} {Se_ave} ± {Se_std}")
        print(f"{MAGENTA}P (GPa):{RESET} {P_ave} ± {P_std}")
        print(f"{MAGENTA}EPtot_cov (Ha*GPa):{RESET} {EPtot_cov}")
        print(f"{MAGENTA}Converged:{RESET} {converged_ave}")

    if auto_save:
        if not silent_mode:
            print(f"{YELLOW}========= Saving results ========{RESET}")
        data = {"n": cfg.num, "rs": cfg.rs, "T": cfg.T,
                "s": s, "params_flow": params_flow,
                "Floc": Floc, "F_ave": F_ave, "F_std": F_std,
                "Etot": Etot, "Etot_ave": Etot_ave, "Etot_std": Etot_std,
                "Eelec": Eelec, "Eelec_ave": Eelec_ave, "Eelec_std": Eelec_std,
                "Ki": Ki, "Ki_ave": Ki_ave, "Ki_std": Ki_std,
                "Vep": Vep, "Vep_ave": Vep_ave, "Vep_std": Vep_std,
                "Vee": Vee, "Vee_ave": Vee_ave, "Vee_std": Vee_std,
                "Vpp": Vpp, "Vpp_ave": Vpp_ave, "Vpp_std": Vpp_std,
                "Sp": Sp, "Sp_ave": Sp_ave, "Sp_std": Sp_std,
                "Se": Se, "Se_ave": Se_ave, "Se_std": Se_std,
                "P": P, "P_ave": P_ave, "P_std": P_std,
                "EPtot": EPtot, "EPtot_cov": EPtot_cov,
                "converged": converged, "converged_ave": converged_ave,
            }
        save_data(data, ckpt_sample_quantity_filename)
        if not silent_mode:
            print(f"{MAGENTA}save sample quantity data to:{RESET}", ckpt_sample_quantity_filename)
    
    return data
