import jax
import csv
import time
import hydra
import optax
import os, sys
import haiku as hk
import jax.numpy as jnp
from loguru import logger
from functools import partial
from datetime import datetime
from scipy.special import comb
from jax.flatten_util import ravel_pytree
from omegaconf import DictConfig, OmegaConf
jax.config.update("jax_enable_x64", True)

from src.logo import print_logo
from src.checkpoint import find_ckpt_filename, load_data, save_data
from src.vmc import sample_s_and_x, make_loss
from src.mcmc import adjust_mc_width
from src.ad import make_grad_real
from src.autoregressive import Transformer
from src.ferminet import FermiNet
from src.potential import kpoints, Madelung
from src.flow import make_flow
from src.utils import shard, replicate, p_split, all_gather
from src.sampler import make_autoregressive_sampler, make_classical_score, make_classical_score_van
from src.logpsi import make_logpsi, make_logpsi_grad_laplacian, make_logpsi2, make_quantum_score
from src.sr import classical_fisher_sr, quantum_fisher_sr, occupation_fisher_sr, pytree_as_whole_block, block_ravel_pytree
from src.cg import classical_fisher_cg, quantum_fisher_cg

from hqc.pbc.lcao import make_lcao
from hqc.pbc.slater import make_slater
from cfgmanager import save

@hydra.main(version_base=None, config_path="conf/train", config_name="config14")
def main_func(cfg: DictConfig) -> None:

    if cfg.num_hosts > 1:
        jax.distributed.initialize(cfg.server_addr, cfg.num_hosts, cfg.host_idx,
                                local_device_ids=[int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')])

    print_logo()
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="{message}", level="DEBUG")
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d_%H:%M:%S")


    logger.opt(colors=True).info("<yellow>========== init path ==========</yellow>")
    if (cfg.load is not None) and os.path.isdir(cfg.load):
        ckpt_filename, epoch_finished = find_ckpt_filename(cfg.load)
    else:
        ckpt_filename = None
    if ckpt_filename is not None:
        path = cfg.load
        logger.opt(colors=True).info("<green>work in directory:</green> {}", path)
    else:
        path = cfg.folder + "n_%d_rs_%g_T_%g_" % (cfg.num, cfg.rs, cfg.T) + cfg.lcao.type \
                          + "_bs_%d_ap_%d_" %(cfg.batchsize, cfg.acc_steps) + current_time_str
        if not os.path.isdir(path):
            os.makedirs(path)
            logger.opt(colors=True).info("<green>creat directory:</green> {}", path)


    logger.opt(colors=True).info("<yellow>\n========== start alog ==========</yellow>")
    alog_filename = os.path.join(path, "alog.log")
    logger.add(alog_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {elapsed} | {level:<8} | {file}:{line} ({module}.{function}) - {message}", colorize=False, level="INFO")
    if ckpt_filename is not None:
        logger.info("Continue pretraining from checkpoint file: {}", ckpt_filename)
    else:
        logger.info("Initiate training")
    logger.info("This is the training script for the hydrogen.")
    logger.info(cfg.note)
    logger.opt(colors=True).info("<green>Save alog file:</green> {}", alog_filename)


    logger.opt(colors=True).info("<yellow>\n========= environment information =========</yellow>")
    env_info = jax.print_environment_info(return_string=True)
    devices = jax.devices()
    logger.opt(colors=True).info("<cyan>GPU devices:</cyan>")
    for i, device in enumerate(devices):
        logger.opt(colors=True).info("  ---- {} {}", i, device.device_kind)
    try:
        logger.opt(colors=True).info("Cluster connected with totally {} GPUs.", jax.device_count(backend='gpu'))
        logger.opt(colors=True).info("This is process {} with {} local GPUs.", jax.process_index(), jax.local_device_count(backend='gpu'))
    except RuntimeError:
        logger.opt(colors=True).info("Cluster connected with totally {} CPUs.", jax.device_count(backend='cpu'))
        logger.opt(colors=True).info("This is process {} with {} local CPUs.", jax.process_index(), jax.local_device_count(backend='cpu'))
    num_devices = jax.local_device_count()
    num_hosts = jax.device_count() // num_devices
    if num_hosts != cfg.num_hosts:
        raise ValueError("Number of hosts does not match the number of available GPUs.")
    if cfg.batchsize % (num_devices * num_hosts) != 0:
        raise ValueError("Batch size must be divisible by the number of GPU devices. "
                            "Got batch = %d for %d devices now." % (cfg.batchsize, num_devices * num_hosts))
    batch_per_device = cfg.batchsize // (num_devices * num_hosts)
    if cfg.batchsize % cfg.lcao.batchsize != 0:
        raise ValueError("Batch size must be divisible by LCAO batch size. "
                            "Got batch = %d for LCAO batch = %d now." % (cfg.batchsize, cfg.lcao.batchsize))
    lcao_mapsize = cfg.batchsize // cfg.lcao.batchsize
    lcao_batch_per_device = cfg.lcao.batchsize // (num_devices * num_hosts)
    key = jax.random.PRNGKey(cfg.seed)
    logger.opt(colors=True).info("<blue>num_hosts:</blue> {}", num_hosts)
    logger.opt(colors=True).info("<blue>num_devices per host:</blue> {}", num_devices)
    logger.opt(colors=True).info("<blue>batchsize:</blue> {}", cfg.batchsize)
    logger.opt(colors=True).info("<blue>batch_per_device:</blue> {}", batch_per_device)
    logger.opt(colors=True).info("<blue>lcao_batchsize:</blue> {}", cfg.lcao.batchsize)
    logger.opt(colors=True).info("<blue>lcao_batch_per_device:</blue> {}", lcao_batch_per_device)
    logger.opt(colors=True).info("<blue>random seed:</blue> {}", cfg.seed)


    logger.opt(colors=True).info("<yellow>\n========== save config ==========</yellow>")
    if cfg.num_hosts > 1:
        if jax.process_index() == 0:
            save(cfg, path, silent_mode=True)
            cfg_filename = os.path.join(path, "config.yaml")
            logger.opt(colors=True).info("\n"+OmegaConf.to_yaml(cfg))
            logger.opt(colors=True).info("<green>Save config file:</green> {}", cfg_filename)
    else:
        save(cfg, path, silent_mode=True)
        cfg_filename = os.path.join(path, "config.yaml")
        logger.opt(colors=True).info("\n"+OmegaConf.to_yaml(cfg))
        logger.opt(colors=True).info("<green>Save config file:</green> {}", cfg_filename)


    logger.opt(colors=True).info("<yellow>\n========= system information =========</yellow>")
    assert (cfg.num%2==0)
    assert (cfg.dim == 3)
    beta = 157888.088922572/cfg.T # inverse temperature in unit of 1/Ry
    reciprocal_beta = 1/beta # temperature in unit of Ry
    smearing_sigma = reciprocal_beta/2 # temperature in Hartree unit
    L = (4/3*jnp.pi*cfg.num)**(1/3)
    logger.opt(colors=True).info("<blue>n:</blue> {}", cfg.num)
    logger.opt(colors=True).info("<blue>L:</blue> {}", L)
    logger.opt(colors=True).info("<blue>rs:</blue> {}", cfg.rs)
    logger.opt(colors=True).info("<blue>L*rs:</blue> {}", L*cfg.rs)
    logger.opt(colors=True).info("<blue>T(Kelvin):</blue> {}", cfg.T)
    logger.opt(colors=True).info("<blue>T(Rydberg):</blue> {}", reciprocal_beta)


    logger.opt(colors=True).info("<yellow>\n========= Initialize single-particle orbitals =========</yellow>")
    kpt = jnp.array([0,0,0])
    lcao = make_lcao(cfg.num, L, cfg.rs, cfg.lcao.basis, rcut=cfg.lcao.rcut, tol=cfg.lcao.tol,
                     max_cycle=cfg.lcao.max_cycle, grid_length=cfg.lcao.grid_length, diis=cfg.lcao.diis.diis, 
                     diis_space=cfg.lcao.diis.space, diis_start_cycle=cfg.lcao.diis.start_cycle, 
                     diis_damp=cfg.lcao.diis.damp, use_jit=cfg.lcao.use_jit, dft=(cfg.lcao.type=="dft"), 
                     xc=cfg.lcao.xc, smearing=cfg.lcao.smearing.smearing, smearing_method=cfg.lcao.smearing.method, 
                     smearing_sigma=smearing_sigma, search_method=cfg.lcao.smearing.search.method, 
                     search_cycle=cfg.lcao.smearing.search.cycle, search_tol=cfg.lcao.smearing.search.tol,
                     gamma=cfg.lcao.gamma)
    lcao_orbitals = make_slater(cfg.num, L, cfg.rs, basis=cfg.lcao.basis, groundstate=False)
    s = jax.random.uniform(key, (cfg.num, cfg.dim), minval=0., maxval=L)
    mo_coeff, bands, _ = lcao(s)
    num_states = bands.shape[0]
    logger.opt(colors=True).info("<blue>Number of available single-particle orbitals:</blue> {}", num_states)
    logger.opt(colors=True).info("<blue>Total number of many-body states (</blue>{}<blue> in </blue>{}<blue>)^2: </blue>{}", cfg.num//2, num_states, comb(num_states, cfg.num//2)**2)
    lcao_map = lambda xp: jax.lax.map(lcao, xp)[0:2]
    lcao_map_vmap = jax.vmap(lcao_map, 0, (0, 0))
    def lcao_vmc(xp):
        xp = xp.reshape(-1, lcao_mapsize, cfg.num, cfg.dim)
        mo_coeff, bands = lcao_map_vmap(xp)
        return mo_coeff.reshape(-1, num_states, num_states), bands.reshape(-1, num_states)
    logger.opt(colors=True).info("<blue>batchsize:</blue> {}", cfg.batchsize)
    logger.opt(colors=True).info("<blue>LCAO batchsize:</blue> {}", cfg.lcao.batchsize)
    logger.opt(colors=True).info("<blue>LCAO map size:</blue> {}", lcao_mapsize)
    logger.opt(colors=True).info("<blue>LCAO type:</blue> {}", cfg.lcao.type)
    logger.opt(colors=True).info("<blue>LCAO gamma:</blue> {}", cfg.lcao.gamma)
    logger.opt(colors=True).info("<blue>LCAO basis:</blue> {}", cfg.lcao.basis)
    logger.opt(colors=True).info("<blue>LCAO rcut:</blue> {}", cfg.lcao.rcut)
    logger.opt(colors=True).info("<blue>LCAO tol:</blue> {}", cfg.lcao.tol)
    logger.opt(colors=True).info("<blue>LCAO max_cycle:</blue> {}", cfg.lcao.max_cycle)
    logger.opt(colors=True).info("<blue>LCAO grid_length:</blue> {}", cfg.lcao.grid_length)
    logger.opt(colors=True).info("<blue>LCAO diis:</blue> {}", cfg.lcao.diis.diis)
    logger.opt(colors=True).info("<blue>LCAO diis space:</blue> {}", cfg.lcao.diis.space)
    logger.opt(colors=True).info("<blue>LCAO diis start_cycle:</blue> {}", cfg.lcao.diis.start_cycle)
    logger.opt(colors=True).info("<blue>LCAO diis damp:</blue> {}", cfg.lcao.diis.damp)
    logger.opt(colors=True).info("<blue>LCAO xc:</blue> {}", cfg.lcao.xc)
    logger.opt(colors=True).info("<blue>LCAO smearing:</blue> {}", cfg.lcao.smearing.smearing)
    logger.opt(colors=True).info("<blue>LCAO smearing method:</blue> {}", cfg.lcao.smearing.method)
    logger.opt(colors=True).info("<blue>LCAO smearing sigma:</blue> {}", smearing_sigma)
    logger.opt(colors=True).info("<blue>LCAO smearing search method:</blue> {}", cfg.lcao.smearing.search.method)
    logger.opt(colors=True).info("<blue>LCAO smearing search cycle:</blue> {}", cfg.lcao.smearing.search.cycle)
    logger.opt(colors=True).info("<blue>LCAO smearing earch tol:</blue> {}", cfg.lcao.smearing.search.tol)
    logger.opt(colors=True).info("<blue>LCAO use_jit:</blue> {}", cfg.lcao.use_jit)


    logger.opt(colors=True).info("<yellow>\n========= Initialize relevant quantities for Ewald summation =========</yellow>")
    G = kpoints(cfg.dim, cfg.ewald.Gmax)
    Vconst = cfg.num * cfg.rs/L * Madelung(cfg.dim, cfg.ewald.kappa, G)
    logger.opt(colors=True).info("<blue>ewald Gmax:</blue> {}", cfg.ewald.Gmax)
    logger.opt(colors=True).info("<blue>ewald kappa:</blue> {}", cfg.ewald.kappa)
    logger.opt(colors=True).info("<blue>ewald Vconst:</blue> {}", Vconst)


    logger.opt(colors=True).info("<yellow>\n========= Initialize normalizing flow =========</yellow>")
    def forward_fn(x):
        for _ in range(cfg.flow.steps):
            model = FermiNet(cfg.flow.depth, cfg.flow.h1size, cfg.flow.h2size, cfg.flow.Nf, L, False, remat=cfg.flow.remat)
            x = model(x)
        return x
    network_flow = hk.transform(forward_fn)
    x_dummy = jax.random.uniform(key, (cfg.num, cfg.dim), minval=0., maxval=L)
    params_flow = network_flow.init(key, x_dummy)
    logprob_flow_novmap = make_flow(network_flow, cfg.num, cfg.dim, L)
    vmap_p = partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    logprob_p = vmap_p(logprob_flow_novmap)
    force_fn_p = vmap_p(make_grad_real(logprob_flow_novmap, argnums=1))
    raveled_params_flow, _ = ravel_pytree(params_flow)
    logger.opt(colors=True).info("<blue>flow steps:</blue> {}", cfg.flow.steps)
    logger.opt(colors=True).info("<blue>flow depth:</blue> {}", cfg.flow.depth)
    logger.opt(colors=True).info("<blue>flow h1size:</blue> {}", cfg.flow.h1size)
    logger.opt(colors=True).info("<blue>flow h2size:</blue> {}", cfg.flow.h2size)
    logger.opt(colors=True).info("<blue>flow Nf:</blue> {}", cfg.flow.Nf)
    logger.opt(colors=True).info("<blue>flow remat:</blue> {}", cfg.flow.remat)
    logger.opt(colors=True).info("<blue>flow parameters:</blue> {}", raveled_params_flow.size)


    logger.opt(colors=True).info("<yellow>\n========= Initialize many-body state distribution (VAN) =========</yellow>")
    def forward_fn(state):
        model = Transformer(num_states, cfg.van.nlayers, cfg.van.modelsize, cfg.van.nheads, cfg.van.nhidden, remat=cfg.van.remat)
        return model(state)
    van = hk.transform(forward_fn)
    state_idx_dummy = jnp.array([jnp.concatenate([jnp.arange(3*cfg.num//2, 2*cfg.num, dtype=jnp.float64),\
                                                jnp.arange(3*cfg.num//2, 2*cfg.num, dtype=jnp.float64)])]).T
    params_van = van.init(key, state_idx_dummy)
    raveled_params_van, _ = ravel_pytree(params_van)
    sampler, logprob_van_novmap = make_autoregressive_sampler(van, cfg.num, num_states, beta)
    logprob_e = jax.vmap(logprob_van_novmap, (None, 0, 0), 0)
    logger.opt(colors=True).info("<blue>van nlayers:</blue> {}", cfg.van.nlayers)
    logger.opt(colors=True).info("<blue>van modelsize:</blue> {}", cfg.van.modelsize)
    logger.opt(colors=True).info("<blue>van nheads:</blue> {}", cfg.van.nheads)
    logger.opt(colors=True).info("<blue>van nhidden:</blue> {}", cfg.van.nhidden)
    logger.opt(colors=True).info("<blue>van remat:</blue> {}", cfg.van.remat)
    logger.opt(colors=True).info("<blue>van number of states:</blue> {}", num_states)
    logger.opt(colors=True).info("<blue>van state_idx_dummy.shape:</blue> {}", state_idx_dummy.shape)
    logger.opt(colors=True).info("<blue>van parameters:</blue> {}", raveled_params_van.size)


    logger.opt(colors=True).info("<yellow>\n========= Initialize wavefunction =========</yellow>")
    def forward_fn(x):
        model = FermiNet(cfg.wfn.depth, cfg.wfn.h1size, cfg.wfn.h2size, cfg.wfn.Nf, L, True, remat=cfg.wfn.remat)
        return model(x)
    network_wfn = hk.transform(forward_fn)
    sx_dummy = jax.random.uniform(key, (2*cfg.num, cfg.dim), minval=0., maxval=L)
    params_wfn = network_wfn.init(key, sx_dummy)
    raveled_params_wfn, _ = ravel_pytree(params_wfn)
    logpsi_novmap = make_logpsi(network_wfn, lcao_orbitals, kpt)
    logpsi2_novmap = make_logpsi2(logpsi_novmap)
    vmap_wfn = partial(jax.vmap, in_axes=(0, None, 0, 0, 0), out_axes=0)
    logpsi = vmap_wfn(logpsi_novmap)
    logpsi2 = vmap_wfn(logpsi2_novmap)
    force_fn_e = vmap_wfn(make_grad_real(logpsi2_novmap))
    logger.opt(colors=True).info("<blue>wfn depth:</blue> {}", cfg.wfn.depth)
    logger.opt(colors=True).info("<blue>wfn h1size:</blue> {}", cfg.wfn.h1size)
    logger.opt(colors=True).info("<blue>wfn h2size:</blue> {}", cfg.wfn.h2size)
    logger.opt(colors=True).info("<blue>wfn Nf:</blue> {}", cfg.wfn.Nf)
    logger.opt(colors=True).info("<blue>wfn remat:</blue> {}", cfg.wfn.remat)
    logger.opt(colors=True).info("<blue>wfn parameters:</blue> {}", raveled_params_wfn.size)


    logger.opt(colors=True).info("<yellow>\n========= Initialize optimizer =========</yellow>")
    score_fn_flow = make_classical_score(logprob_flow_novmap)
    score_fn_van = make_classical_score_van(logprob_van_novmap)
    score_fn_wfn = make_quantum_score(logpsi_novmap)

    if cfg.optimizer.flow.type not in ["sr", "cg"]:
        raise ValueError('Currently we assume the second-order optimizer for protons to be "sr" or "cg".')
    logger.opt(colors=True).info("<blue>flow optimizer:</blue> {}", cfg.optimizer.flow.type)
    logger.opt(colors=True).info("<blue>flow learning rate:</blue> {}", cfg.optimizer.flow.lr)
    logger.opt(colors=True).info("<blue>flow decay:</blue> {}", cfg.optimizer.flow.decay)
    logger.opt(colors=True).info("<blue>flow damping:</blue> {}", cfg.optimizer.flow.damping)
    logger.opt(colors=True).info("<blue>flow maxnorm:</blue> {}", cfg.optimizer.flow.maxnorm)
    logger.opt(colors=True).info("<blue>flow clip_factor:</blue> {}", cfg.optimizer.flow.clip_factor)
    if cfg.optimizer.flow.type == "cg":
        cg_params_flow = {"mode": cfg.optimizer.flow.cg.mode,
                          "init_vec_last_step": cfg.optimizer.flow.cg.init_vec_last_step,
                          "solver_precondition": cfg.optimizer.flow.cg.solver.precondition,
                          "solver_maxiter": cfg.optimizer.flow.cg.solver.maxiter,
                          "solver_tol": cfg.optimizer.flow.cg.solver.tol,
                          "solver_style": cfg.optimizer.flow.cg.solver.style}
        logp_fn_flow_dummy = lambda x, params: logprob_p(params, x)
        score_fn_flow_dummy = lambda x, params: score_fn_flow(params, x)
        fishers_fn_flow, state_ckpt_fn_flow, optimizer_flow = classical_fisher_cg(logp_fn_flow_dummy, 
            score_fn_flow_dummy, cfg.acc_steps, cfg.optimizer.flow.cg.gamma, cfg.optimizer.flow.lr, cfg.optimizer.flow.decay,
            cfg.optimizer.flow.damping, cfg.optimizer.flow.maxnorm, **cg_params_flow)
        logger.opt(colors=True).info("<blue>flow cg mode:</blue> {}", cfg.optimizer.flow.cg.mode)
        logger.opt(colors=True).info("<blue>clow cg gamma:</blue> {}", cfg.optimizer.flow.cg.gamma)
        logger.opt(colors=True).info("<blue>flow cg init_vec_last_step:</blue> {}", cfg.optimizer.flow.cg.init_vec_last_step)
        logger.opt(colors=True).info("<blue>flow cg solver precondition:</blue> {}", cfg.optimizer.flow.cg.solver.precondition)
        logger.opt(colors=True).info("<blue>flow cg solver maxiter:</blue> {}", cfg.optimizer.flow.cg.solver.maxiter)
        logger.opt(colors=True).info("<blue>flow cg solver tol:</blue> {}", cfg.optimizer.flow.cg.solver.tol)
        logger.opt(colors=True).info("<blue>flow cg solver style:</blue> {}", cfg.optimizer.flow.cg.solver.style)
    elif cfg.optimizer.flow.type == "sr":
        block_fn_flow = pytree_as_whole_block(params_flow)
        fishers_fn_flow, state_ckpt_fn_flow, optimizer_flow = classical_fisher_sr(score_fn_flow, block_fn_flow,
                    cfg.acc_steps, cfg.optimizer.flow.sr.alpha, cfg.optimizer.flow.lr, cfg.optimizer.flow.decay, 
                    cfg.optimizer.flow.damping, cfg.optimizer.flow.maxnorm)
        logger.opt(colors=True).info("<blue>alpha:</blue> {}", cfg.optimizer.flow.sr.alpha)

    if cfg.optimizer.van.type not in ["sr", "cg"]:
        raise ValueError('Currently we assume the second-order optimizer for VAN to be "sr" or "cg".')
    logger.opt(colors=True).info("<blue>van optimizer:</blue> {}", cfg.optimizer.van.type)
    logger.opt(colors=True).info("<blue>van learning rate:</blue> {}", cfg.optimizer.van.lr)
    logger.opt(colors=True).info("<blue>van decay:</blue> {}", cfg.optimizer.van.decay)
    logger.opt(colors=True).info("<blue>van damping:</blue> {}", cfg.optimizer.van.damping)
    logger.opt(colors=True).info("<blue>van maxnorm:</blue> {}", cfg.optimizer.van.maxnorm)
    logger.opt(colors=True).info("<blue>van clip_factor:</blue> {}", cfg.optimizer.van.clip_factor)
    if cfg.optimizer.van.type == "cg":
        cg_params_van = {"mode": cfg.optimizer.van.cg.mode,
                          "init_vec_last_step": cfg.optimizer.van.cg.init_vec_last_step,
                          "solver_precondition": cfg.optimizer.van.cg.solver.precondition,
                          "solver_maxiter": cfg.optimizer.van.cg.solver.maxiter,
                          "solver_tol": cfg.optimizer.van.cg.solver.tol,
                          "solver_style": cfg.optimizer.van.cg.solver.style}
        logp_fn_van_dummy = lambda state_idx, params, bands: logprob_e(params, state_idx, bands)
        score_fn_van_dummy = lambda state_idx, params, bands: score_fn_van(params, state_idx, bands)
        fishers_fn_van, state_ckpt_fn_van, optimizer_van = classical_fisher_cg(
            logp_fn_van_dummy, score_fn_van_dummy, cfg.acc_steps,
            cfg.optimizer.van.cg.gamma, cfg.optimizer.van.lr, cfg.optimizer.van.decay, cfg.optimizer.van.damping, cfg.optimizer.van.maxnorm, 
            van=True, **cg_params_van
            )
        logger.opt(colors=True).info("<blue>van cg mode:</blue> {}", cfg.optimizer.van.cg.mode)
        logger.opt(colors=True).info("<blue>van cg gamma:</blue> {}", cfg.optimizer.van.cg.gamma)
        logger.opt(colors=True).info("<blue>van cg init_vec_last_step:</blue> {}", cfg.optimizer.van.cg.init_vec_last_step)
        logger.opt(colors=True).info("<blue>van cg solver precondition:</blue> {}", cfg.optimizer.van.cg.solver.precondition)
        logger.opt(colors=True).info("<blue>van cg solver maxiter:</blue> {}", cfg.optimizer.van.cg.solver.maxiter)
        logger.opt(colors=True).info("<blue>van cg solver tol:</blue> {}", cfg.optimizer.van.cg.solver.tol)
        logger.opt(colors=True).info("<blue>van cg solver style:</blue> {}", cfg.optimizer.van.cg.solver.style)
    elif cfg.optimizer.van.type == "sr":
        block_fn_van = pytree_as_whole_block(params_van)
        fishers_fn_van, state_ckpt_fn_van, optimizer_van = occupation_fisher_sr(score_fn_van, block_fn_van, cfg.acc_steps, 
                cfg.optimizer.van.sr.alpha, cfg.optimizer.van.lr, cfg.optimizer.van.decay, cfg.optimizer.van.damping, cfg.optimizer.van.maxnorm)

    if cfg.optimizer.wfn.type not in ["sr", "cg"]:
        raise ValueError('Currently we assume the second-order optimizer for electrons to be "sr" or "cg".')
    logger.opt(colors=True).info("<blue>wfn optimizer:</blue> {}", cfg.optimizer.wfn.type)
    logger.opt(colors=True).info("<blue>wfn learning rate:</blue> {}", cfg.optimizer.wfn.lr)
    logger.opt(colors=True).info("<blue>wfn decay:</blue> {}", cfg.optimizer.wfn.decay)
    logger.opt(colors=True).info("<blue>wfn damping:</blue> {}", cfg.optimizer.wfn.damping)
    logger.opt(colors=True).info("<blue>wfn maxnorm:</blue> {}", cfg.optimizer.wfn.maxnorm)
    logger.opt(colors=True).info("<blue>wfn clip_factor:</blue> {}", cfg.optimizer.wfn.clip_factor)
    if cfg.optimizer.wfn.type == "cg":
        cg_params_wfn = {"mode": cfg.optimizer.wfn.cg.mode,
                         "init_vec_last_step": cfg.optimizer.wfn.cg.init_vec_last_step,
                         "solver_precondition": cfg.optimizer.wfn.cg.solver.precondition,
                         "solver_maxiter": cfg.optimizer.wfn.cg.solver.maxiter,
                         "solver_tol": cfg.optimizer.wfn.cg.solver.tol,
                         "solver_style": cfg.optimizer.wfn.cg.solver.style}
        fishers_fn_wfn, state_ckpt_fn_wfn, optimizer_wfn = quantum_fisher_cg(
                logpsi, score_fn_wfn, cfg.acc_steps,
                cfg.optimizer.wfn.cg.gamma, cfg.optimizer.wfn.lr, cfg.optimizer.wfn.decay, cfg.optimizer.wfn.damping, cfg.optimizer.wfn.maxnorm,
                **cg_params_wfn)
        logger.opt(colors=True).info("<blue>wfn cg mode:</blue> {}", cfg.optimizer.wfn.cg.mode)
        logger.opt(colors=True).info("<blue>wfn cg gamma:</blue> {}", cfg.optimizer.wfn.cg.gamma)
        logger.opt(colors=True).info("<blue>wfn cg init_vec_last_step:</blue> {}", cfg.optimizer.wfn.cg.init_vec_last_step)
        logger.opt(colors=True).info("<blue>wfn cg solver precondition:</blue> {}", cfg.optimizer.wfn.cg.solver.precondition)
        logger.opt(colors=True).info("<blue>wfn cg solver maxiter:</blue> {}", cfg.optimizer.wfn.cg.solver.maxiter)
        logger.opt(colors=True).info("<blue>wfn cg solver tol:</blue> {}", cfg.optimizer.wfn.cg.solver.tol)
        logger.opt(colors=True).info("<blue>wfn cg solver style:</blue> {}", cfg.optimizer.wfn.cg.solver.style)
    elif cfg.optimizer.wfn.type == "sr":
        block_fn_wfn = pytree_as_whole_block(params_wfn)
        fishers_fn_wfn, state_ckpt_fn_wfn, optimizer_wfn = quantum_fisher_sr(score_fn_wfn, block_fn_wfn, cfg.acc_steps, 
                cfg.optimizer.wfn.sr.alpha, cfg.optimizer.wfn.lr, cfg.optimizer.wfn.decay, cfg.optimizer.wfn.damping, cfg.optimizer.wfn.maxnorm)

    # flow optimizer init
    if cfg.optimizer.flow.type == "cg":
        s_dummy = jnp.empty((num_devices, batch_per_device, cfg.num, cfg.dim))
        opt_state_flow = jax.pmap(optimizer_flow.init,
                                  in_axes=(None, 0),
                                  out_axes=0)(params_flow, s_dummy)
        opt_state_flow_pmap_axis = 0
    else:
        opt_state_flow = optimizer_flow.init(params_flow)
        opt_state_flow_pmap_axis = None

    # van optimizer init
    bands_dummy = jnp.tile(bands, (cfg.batchsize, 1))
    state_idx_dummy = sampler(params_van, key, cfg.batchsize, bands_dummy)
    if cfg.optimizer.van.type == "cg":
        state_idx_dummy = state_idx_dummy.reshape(num_devices, batch_per_device, cfg.num)
        bands_dummy = bands_dummy.reshape(num_devices, batch_per_device, num_states)
        opt_state_van = jax.pmap(optimizer_van.init,
                                in_axes=(None, 0, 0),
                                out_axes=0)(params_van, state_idx_dummy, bands_dummy)
        opt_state_van_pmap_axis = 0
    else:
        opt_state_van = optimizer_van.init(params_van)
        opt_state_van_pmap_axis = None

    # wfn optimizer init
    mo_coeff_dummy = jnp.tile(mo_coeff, (cfg.batchsize, 1, 1))
    if cfg.optimizer.wfn.type == "cg":
        s_dummy = jnp.empty((num_devices, batch_per_device, cfg.num, cfg.dim))
        x_dummy = jnp.empty((num_devices, batch_per_device, cfg.num, cfg.dim))
        state_idx_dummy = state_idx_dummy.reshape(num_devices, batch_per_device, cfg.num)
        mo_coeff_dummy = mo_coeff_dummy.reshape(num_devices, batch_per_device, num_states, num_states)
        opt_state_wfn = jax.pmap(optimizer_wfn.init, 
                                in_axes=(None, 0, 0, 0, 0),
                                out_axes=0)(params_wfn, x_dummy, s_dummy, state_idx_dummy, mo_coeff_dummy)
        opt_state_wfn_pmap_axis = 0
    else:
        opt_state_wfn = optimizer_wfn.init(params_wfn)
        opt_state_wfn_pmap_axis = None


    logger.opt(colors=True).info("<yellow>\n========= Checkpointing =========</yellow>")
    if ckpt_filename is not None:
        continue_run_therm = False
        logger.opt(colors=True).info("<green>Load checkpoint file:</green> {}", ckpt_filename)
        logger.opt(colors=True).info("<blue>epoch_finished:</blue> {}", epoch_finished)
        ckpt = load_data(ckpt_filename)
        keys, s, x, params_flow, params_van, params_wfn= \
            ckpt["keys"], ckpt["s"], ckpt["x"], ckpt["params_flow"], ckpt["params_van"], ckpt["params_wfn"]
        opt_state_flow_ckpt = jax.tree_util.tree_map(lambda x: replicate(x[0], num_devices), ckpt["opt_state_flow"]) \
                        if opt_state_flow_pmap_axis is not None else ckpt["opt_state_flow"]
        opt_state_flow.update(opt_state_flow_ckpt)
        opt_state_van_ckpt = jax.tree_util.tree_map(lambda x: replicate(x[0], num_devices), ckpt["opt_state_van"]) \
                        if opt_state_van_pmap_axis is not None else ckpt["opt_state_van"]
        opt_state_van.update(opt_state_van_ckpt)
        opt_state_wfn_ckpt = jax.tree_util.tree_map(lambda x: replicate(x[0], num_devices), ckpt["opt_state_wfn"]) \
                        if opt_state_wfn_pmap_axis is not None else ckpt["opt_state_wfn"]
        opt_state_wfn.update(opt_state_wfn_ckpt)
        try:
            mc_width_p, mc_width_e = ckpt["mc_width_p"], ckpt["mc_width_e"]
        except (NameError, KeyError):
            mc_width_p, mc_width_e = cfg.mc.width_p, cfg.mc.width_e
        if cfg.num_hosts > 1:
            keys = jax.random.split(keys[0], (num_hosts, num_devices))
            if (s.size == num_hosts*num_devices*batch_per_device*cfg.num*cfg.dim) and (x.size == num_hosts*num_devices*batch_per_device*cfg.num*cfg.dim):
                s = jnp.reshape(s, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim))
                x = jnp.reshape(x, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim))
            elif s.size > num_hosts*num_devices*batch_per_device*cfg.num*cfg.dim:
                s = jnp.reshape(s, (-1, cfg.num, cfg.dim))[:cfg.batchsize]
                s = jnp.reshape(s, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim))
                x = jnp.reshape(x, (-1, cfg.num, cfg.dim))[:cfg.batchsize]
                x = jnp.reshape(x, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim))
            else:
                s = jnp.reshape(s, (-1, cfg.num, cfg.dim))
                s = jnp.tile(s, (cfg.batchsize // s.shape[0] + 1, 1, 1))
                s = s[:cfg.batchsize]
                s = jnp.reshape(s, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim))
                x = jnp.reshape(x, (-1, cfg.num, cfg.dim))
                x = jnp.tile(x, (cfg.batchsize // x.shape[0] + 1, 1, 1))
                x = x[:cfg.batchsize]
                x = jnp.reshape(x, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim))
                continue_run_therm = True
            s = s[jax.process_index()]
            x = x[jax.process_index()]
            keys = keys[jax.process_index()]
        else:
            keys = jax.random.split(keys[0], num_devices)
            if (s.size == num_devices*batch_per_device*cfg.num*cfg.dim) and (x.size == num_devices*batch_per_device*cfg.num*cfg.dim):
                s = jnp.reshape(s, (num_devices, batch_per_device, cfg.num, cfg.dim))
                x = jnp.reshape(x, (num_devices, batch_per_device, cfg.num, cfg.dim))
            elif s.size > num_devices*batch_per_device*cfg.num*cfg.dim:
                s = jnp.reshape(s, (-1, cfg.num, cfg.dim))[:cfg.batchsize]
                s = jnp.reshape(s, (num_devices, batch_per_device, cfg.num, cfg.dim))
                x = jnp.reshape(x, (-1, cfg.num, cfg.dim))[:cfg.batchsize]
                x = jnp.reshape(x, (num_devices, batch_per_device, cfg.num, cfg.dim))
            else:
                s = jnp.reshape(s, (-1, cfg.num, cfg.dim))
                s = jnp.tile(s, (cfg.batchsize // s.shape[0] + 1, 1, 1))
                s = s[:cfg.batchsize]
                s = jnp.reshape(s, (num_devices, batch_per_device, cfg.num, cfg.dim))
                x = jnp.reshape(x, (-1, cfg.num, cfg.dim))
                x = jnp.tile(x, (cfg.batchsize // x.shape[0] + 1, 1, 1))
                x = x[:cfg.batchsize]
                x = jnp.reshape(x, (num_devices, batch_per_device, cfg.num, cfg.dim))
                continue_run_therm = True
        s, x, keys = shard(s), shard(x), shard(keys)
        params_flow, params_van, params_wfn = replicate((params_flow, params_van, params_wfn), num_devices)
        logger.opt(colors=True).info("Successfully load key, s and x.")
        logger.opt(colors=True).info("<blue>s.shape:</blue> {}", s.shape)
        logger.opt(colors=True).info("<blue>s.type:</blue> {}", type(s))
    else:
        continue_run_therm = False
        logger.opt(colors=True).info("Initializing key, s and x...")
        key, key_proton, key_electron = jax.random.split(key, 3)
        if cfg.load_pretrain.flow is not None:
            params_flow = load_data(cfg.load_pretrain.flow)["params_flow"]
            logger.opt(colors=True).info("Successfully load flow parameters from {}.", cfg.load_pretrain.flow)
            s_load = load_data(cfg.load_pretrain.flow)["s"]
            if s_load.size >= cfg.batchsize*cfg.num*cfg.dim:
                s = jnp.reshape(s_load, (-1, cfg.num, cfg.dim))[:cfg.batchsize]
                logger.opt(colors=True).info("Successfully load s from {}.", cfg.load_pretrain.flow)
            else:
                s = jax.random.uniform(key_proton, (cfg.batchsize, cfg.num, cfg.dim), minval=0., maxval=L)
                logger.opt(colors=True).info("Successfully init s from uniform distribution.")
            if cfg.num_hosts > 1:
                s = jnp.reshape(s, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim))
                s = s[jax.process_index()]
            else:
                s = jnp.reshape(s, (num_devices, batch_per_device, cfg.num, cfg.dim))
        else:
            if cfg.num_hosts > 1:
                s = jax.random.uniform(key_proton, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim), minval=0., maxval=L)
                s = s[jax.process_index()]
                logger.opt(colors=True).info("Successfully init s from uniform distribution")
            else:
                s = jax.random.uniform(key_proton, (num_devices, batch_per_device, cfg.num, cfg.dim), minval=0., maxval=L)
                logger.opt(colors=True).info("Successfully init s from uniform distribution.")
        if num_hosts > 1:
            x = jax.random.uniform(key_electron, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim), minval=0., maxval=L)
            x = x[jax.process_index()]
            keys = jax.random.split(key, (num_hosts, num_devices))
            keys = keys[jax.process_index()]
            logger.opt(colors=True).info("Successfully init x from uniform distribution.")
        else:
            x = jax.random.uniform(key_electron, (num_devices, batch_per_device, cfg.num, cfg.dim), minval=0., maxval=L)
            keys = jax.random.split(key, num_devices)
            logger.opt(colors=True).info("Successfully init x from uniform distribution.")
        epoch_finished = 0
        s, x, keys = shard(s), shard(x), shard(keys)
        params_flow, params_van, params_wfn = replicate((params_flow, params_van, params_wfn), num_devices)
        mc_width_p, mc_width_e = cfg.mc.width_p, cfg.mc.width_e

    
    if (epoch_finished == 0) or continue_run_therm:
        logger.opt(colors=True).info("<yellow>\n========= Thermalization =========</yellow>")
        for i in range(cfg.mc.therm):
            logger.opt(colors=True).info("---- thermal step {} ----", i+1)
            keys, state_idx, mo_coeff, bands, s, x, ar_s, ar_x = sample_s_and_x(keys,
                                    sampler, params_van, 
                                    logprob_p, force_fn_p, s, params_flow,
                                    logpsi2, force_fn_e, x, params_wfn,
                                    cfg.mc.steps_p, cfg.mc.steps_e, mc_width_p, mc_width_e, L, lcao_vmc, kpt)
            logger.opt(colors=True).info("acc: {} {}, proton entropy: {}", jnp.mean(ar_s), jnp.mean(ar_x), -jax.pmap(logprob_p)(params_flow, s).mean()/cfg.num)


    logger.opt(colors=True).info("<yellow>\n========= Start training =========</yellow>")
    logpsi, logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi_novmap, hutchinson=cfg.hutchinson)
    observable_and_lossfn = make_loss(logprob_p, logprob_e, logpsi, logpsi_grad_laplacian,
                                    cfg.ewald.kappa, G, L, cfg.rs, Vconst, beta, 
                                    cfg.optimizer.flow.clip_factor, 
                                    cfg.optimizer.van.clip_factor, 
                                    cfg.optimizer.wfn.clip_factor)

    @partial(jax.pmap, axis_name="p",
            in_axes=(0, 0, 0, 
                    opt_state_flow_pmap_axis, 
                    opt_state_van_pmap_axis, 
                    opt_state_wfn_pmap_axis,
                    0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, None),  
            out_axes=(0, 0, 0,
                    opt_state_flow_pmap_axis, 
                    opt_state_van_pmap_axis, 
                    opt_state_wfn_pmap_axis,
                    0, 0, 0, 0), 
            static_broadcasted_argnums=16)
    def update(params_flow, params_van, params_wfn,
            opt_state_flow, opt_state_van, opt_state_wfn,
            state_idx, mo_coeff, bands, s, x, key, data_acc,
            grad_flow_acc, grad_van_acc, grad_wfn_acc, final_step):
        data, flow_lossfn, van_lossfn, wfn_lossfn = observable_and_lossfn(params_flow, params_van, params_wfn, state_idx, mo_coeff, bands, s, x, key)
        flow_grad = jax.grad(flow_lossfn)(params_flow)
        van_grad = jax.grad(van_lossfn)(params_van)
        wfn_grad = jax.grad(wfn_lossfn)(params_wfn)
        flow_grad, van_grad, wfn_grad = jax.lax.pmean((flow_grad, van_grad, wfn_grad), axis_name="p")
        data_acc, grad_flow_acc, grad_van_acc, grad_wfn_acc = jax.tree_util.tree_map(lambda acc, i: acc + i, 
                                                        (data_acc, grad_flow_acc, grad_van_acc, grad_wfn_acc),  
                                                        (data, flow_grad, van_grad, wfn_grad))
        opt_state_flow = fishers_fn_flow(params_flow, s, opt_state_flow)
        opt_state_van = fishers_fn_van(params_van, state_idx, opt_state_van, bands)
        opt_state_wfn = fishers_fn_wfn(params_wfn, x, opt_state_wfn, s, state_idx, mo_coeff)
        if final_step:
            data_acc, grad_flow_acc, grad_van_acc, grad_wfn_acc = jax.tree_util.tree_map(
                lambda acc: acc / cfg.acc_steps, 
                (data_acc, grad_flow_acc, grad_van_acc, grad_wfn_acc)
            )
            update_flow, opt_state_flow = optimizer_flow.update(grad_flow_acc, opt_state_flow, params_flow)
            update_van, opt_state_van = optimizer_van.update(grad_van_acc, opt_state_van, params_van)
            update_wfn, opt_state_wfn = optimizer_wfn.update(grad_wfn_acc, opt_state_wfn, params_wfn)

            # jax.debug.print("update flow: {x}", x=update_flow)

            params_flow = optax.apply_updates(params_flow, update_flow)
            params_van = optax.apply_updates(params_van, update_van)
            params_wfn = optax.apply_updates(params_wfn, update_wfn)
        return params_flow, params_van, params_wfn, opt_state_flow, opt_state_van, opt_state_wfn, \
               data_acc, grad_flow_acc, grad_van_acc, grad_wfn_acc

    time_of_last_ckpt = time.time()
    log_filename = os.path.join(path, "data.txt")
    logger.opt(colors=True).info("<green>Save data in file:</green> {}", log_filename)
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename)==0:
        f.write("epoch f f_err e e_err k k_err vpp vpp_err vep vep_err vee vee_err p p_err ep_cov sp sp_err se se_err acc_s acc_x\n")
    for i in range(epoch_finished + 1, cfg.epoch + 1):
        data_acc = replicate({"F": 0., "F2": 0.,
                              "E": 0., "E2": 0.,
                              "K": 0., "K2": 0.,
                              "Vpp": 0., "Vpp2": 0.,
                              "Vep": 0., "Vep2": 0.,
                              "Vee": 0., "Vee2": 0.,
                              "P": 0., "P2": 0.,
                              "EP": 0.,
                              "Sp": 0., "Sp2": 0.,
                              "Se": 0., "Se2": 0.,
                              }, num_devices)
        grad_flow_acc = shard(jax.tree_map(jnp.zeros_like, params_flow))
        grad_van_acc = shard(jax.tree_map(jnp.zeros_like, params_van))
        grad_wfn_acc = shard(jax.tree_map(jnp.zeros_like, params_wfn))
        ar_s_acc = shard(jnp.zeros(num_devices))
        ar_x_acc = shard(jnp.zeros(num_devices))
        for acc in range(cfg.acc_steps):
            keys, state_idx, mo_coeff, bands, s, x, ar_s, ar_x = sample_s_and_x(keys,
                                                sampler, params_van, 
                                                logprob_p, force_fn_p, s, params_flow,
                                                logpsi2, force_fn_e, x, params_wfn,
                                                cfg.mc.steps_p, cfg.mc.steps_e, 
                                                mc_width_p, mc_width_e, L, lcao_vmc, kpt)
            ar_s_acc += ar_s/cfg.acc_steps
            ar_x_acc += ar_x/cfg.acc_steps
            final_step = (acc == cfg.acc_steps - 1)
            params_flow, params_van, params_wfn, opt_state_flow, opt_state_van, opt_state_wfn, \
            data_acc, grad_flow_acc, grad_van_acc, grad_wfn_acc = update(
                params_flow, params_van, params_wfn, opt_state_flow, opt_state_van, opt_state_wfn, 
                state_idx, mo_coeff, bands, s, x, keys, data_acc, 
                grad_flow_acc, grad_van_acc, grad_wfn_acc, final_step)
            # logger.opt(colors=True).info("<red>flow grad:</red> {}", grad_flow_acc)
            # logger.opt(colors=True).info("<red>van grad:</red> {}", grad_van_acc)
            # logger.opt(colors=True).info("<red>wfn grad:</red> {}", grad_wfn_acc)

        data = jax.tree_map(lambda x: x[0], data_acc)
        ar_s = ar_s_acc[0] 
        ar_x = ar_x_acc[0] 
        F, F2, E, E2, K, K2, Vpp, Vpp2, Vep, Vep2, Vee, Vee2, P, P2, EP, Sp, Sp2, Se, Se2 = \
                data["F"], data["F2"], \
                data["E"], data["E2"], \
                data["K"], data["K2"], \
                data["Vpp"], data["Vpp2"],\
                data["Vep"], data["Vep2"],\
                data["Vee"], data["Vee2"],\
                data["P"], data["P2"], \
                data["EP"], \
                data["Sp"], data["Sp2"], \
                data["Se"], data["Se2"]
        K_std = jnp.sqrt((K2- K**2) / (cfg.batchsize*cfg.acc_steps))
        Vpp_std = jnp.sqrt((Vpp2- Vpp**2) / (cfg.batchsize*cfg.acc_steps))
        Vep_std = jnp.sqrt((Vep2- Vep**2) / (cfg.batchsize*cfg.acc_steps))
        Vee_std = jnp.sqrt((Vee2- Vee**2) / (cfg.batchsize*cfg.acc_steps))
        P_std = jnp.sqrt((P2- P**2) / (cfg.batchsize*cfg.acc_steps))
        E_std = jnp.sqrt((E2- E**2) / (cfg.batchsize*cfg.acc_steps))
        EP_cov = (EP- E*P) / (cfg.batchsize*cfg.acc_steps)
        F_std = jnp.sqrt((F2- F**2) / (cfg.batchsize*cfg.acc_steps))
        Sp_std = jnp.sqrt((Sp2- Sp**2) / (cfg.batchsize*cfg.acc_steps))
        Se_std = jnp.sqrt((Se2- Se**2) / (cfg.batchsize*cfg.acc_steps))
        logger.opt(colors=True).info("<blue>iter:</blue> %04d " % i + \
            "<blue>F:</blue> {} <blue>F_std:</blue> {} <blue>E:</blue> {} <blue>E_std:</blue> {} " + \
            "<blue>accept_rate:</blue> {} {}", 
            F/cfg.num/cfg.rs**2, F_std/cfg.num/cfg.rs**2, 
            E/cfg.num/cfg.rs**2, E_std/cfg.num/cfg.rs**2, ar_s, ar_x)
        f.write( ("%6d" + "  %.6f"*19 + "  %.4f"*2 + "\n") % (i,
                                                    F/cfg.num/cfg.rs**2, F_std/cfg.num/cfg.rs**2,
                                                    E/cfg.num/cfg.rs**2, E_std/cfg.num/cfg.rs**2,
                                                    K/cfg.num/cfg.rs**2, K_std/cfg.num/cfg.rs**2,
                                                    Vpp/cfg.num/cfg.rs**2, Vpp_std/cfg.num/cfg.rs**2,
                                                    Vep/cfg.num/cfg.rs**2, Vep_std/cfg.num/cfg.rs**2,
                                                    Vee/cfg.num/cfg.rs**2, Vee_std/cfg.num/cfg.rs**2, # Ry
                                                    P/cfg.rs**2, P_std/cfg.rs**2, # GPa 
                                                    EP_cov/cfg.num/cfg.rs**4, # GPa
                                                    Sp/cfg.num, Sp_std/cfg.num, 
                                                    Se/cfg.num, Se_std/cfg.num, 
                                                    ar_s, ar_x) )
        if time.time() - time_of_last_ckpt > 600:
            opt_state_flow_ckpt = state_ckpt_fn_flow(opt_state_flow)
            opt_state_van_ckpt = state_ckpt_fn_van(opt_state_van)
            opt_state_wfn_ckpt = state_ckpt_fn_wfn(opt_state_wfn)
            ckpt = {"keys": keys,
                    "s": s,
                    "x": x,
                    "state_idx": state_idx, 
                    "mo_coeff": mo_coeff, 
                    "bands": bands,
                    "params_flow": jax.tree_map(lambda x: x[0], params_flow),
                    "params_van": jax.tree_map(lambda x: x[0], params_van),
                    "params_wfn": jax.tree_map(lambda x: x[0], params_wfn),
                    "opt_state_flow": jax.tree_util.tree_map(lambda x: x[0], all_gather(opt_state_flow_ckpt, "p")) 
                                if opt_state_flow_pmap_axis is not None else opt_state_flow_ckpt, 
                    "opt_state_van": jax.tree_util.tree_map(lambda x: x[0], all_gather(opt_state_van_ckpt, "p")) 
                                if opt_state_van_pmap_axis is not None else opt_state_van_ckpt, 
                    "opt_state_wfn": jax.tree_util.tree_map(lambda x: x[0], all_gather(opt_state_wfn_ckpt, "p")) 
                                if opt_state_wfn_pmap_axis is not None else opt_state_wfn_ckpt, 
                    "mc_width_p": mc_width_p, 
                    "mc_width_e": mc_width_e
                }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %i)
            save_data(ckpt, ckpt_filename)
            logger.opt(colors=True).info("<green>Save checkpoint file:</green> {}", ckpt_filename)
            time_of_last_ckpt = time.time()
        if jnp.isnan(F):
            raise RuntimeError("Free energy is nan")
        if ar_s < 1e-7 or ar_x < 1e-7:
            raise RuntimeError("Acceptance rate nearly zero")
        if i % 100 == 0:
            mc_width_p = adjust_mc_width(mc_width_p, ar_s, "mcmc")
            mc_width_e = adjust_mc_width(mc_width_e, ar_x, "mcmc")
            logger.opt(colors=True).info("adjust mc width p to {}", mc_width_p) 
            logger.opt(colors=True).info("adjust mc width e to {}", mc_width_e)            
    f.close()
    logger.opt(colors=True).info("<yellow>\n========= Training finished =========</yellow>")

if __name__ == "__main__": 
    main_func()