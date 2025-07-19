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
from jax.flatten_util import ravel_pytree
from omegaconf import DictConfig, OmegaConf
jax.config.update("jax_enable_x64", True)

from src.logo import print_logo
from src.flow import make_flow
from src.ad import make_grad_real
from src.ferminet import FermiNet
from src.sampler import make_classical_score
from src.sr import classical_fisher_sr, pytree_as_whole_block
from src.cg import classical_fisher_cg
from src.utils import shard, replicate, p_split, all_gather
from src.checkpoint import find_ckpt_filename, load_data, save_data
from src.mcmc import adjust_mc_width
from src.vmc import sample_s, make_loss_pretrain_flow

from hqc.pbc.pes import make_pes
from cfgmanager import save

@hydra.main(version_base=None, config_path="conf/pretrain/flow", config_name="config14")
def main_func(cfg: DictConfig) -> None:    


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
        path = cfg.folder + "n_%d_rs_%g_T_%g_" % (cfg.num, cfg.rs, cfg.T) + cfg.pes.type \
                          + "_st_%d_dp_%d_h1_%d_h2_%d" % (cfg.flow.steps, cfg.flow.depth, cfg.flow.h1size, cfg.flow.h2size) \
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
        logger.info("Initiate pretraining")
    logger.info("This is the pretraining script for the normalizing flow model.")
    logger.info(cfg.note)
    logger.opt(colors=True).info("<green>Save alog file:</green> {}", alog_filename)


    logger.opt(colors=True).info("<yellow>\n========== save config ==========</yellow>")
    save(cfg, path, silent_mode=True)
    cfg_filename = os.path.join(path, "config.yaml")
    logger.opt(colors=True).info("\n"+OmegaConf.to_yaml(cfg))
    logger.opt(colors=True).info("<green>Save config file:</green> {}", cfg_filename)


    logger.opt(colors=True).info("<yellow>\n========= environment information =========</yellow>")
    if cfg.num_hosts > 1:
        jax.distributed.initialize(cfg.server_addr, cfg.num_hosts, cfg.host_idx,
                                local_device_ids=[int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')])
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
    pes_batch_per_device = cfg.pes.batchsize // (num_devices * num_hosts)
    key = jax.random.PRNGKey(cfg.seed)
    logger.opt(colors=True).info("<blue>num_hosts:</blue> {}", num_hosts)
    logger.opt(colors=True).info("<blue>num_device:</blue> {}", num_devices)
    logger.opt(colors=True).info("<blue>batchsize:</blue> {}", cfg.batchsize)
    logger.opt(colors=True).info("<blue>batch_per_device:</blue> {}", batch_per_device)
    logger.opt(colors=True).info("<blue>pes_batch_per_device:</blue> {}", pes_batch_per_device)
    logger.opt(colors=True).info("<blue>random seed:</blue> {}", cfg.seed)
    

    logger.opt(colors=True).info("<yellow>\n========= system information =========</yellow>")
    assert (cfg.num%2==0)
    reciprocal_beta = cfg.T/157888.088922572 # temperature in unit of Ry
    smearing_sigma = reciprocal_beta/2 # temperature in Hartree unit
    L = (4/3*jnp.pi*cfg.num)**(1/3)
    logger.opt(colors=True).info("<blue>n:</blue> {}",cfg.num)
    logger.opt(colors=True).info("<blue>L:</blue> {}", L)
    logger.opt(colors=True).info("<blue>rs:</blue> {}", cfg.rs)
    logger.opt(colors=True).info("<blue>L*rs:</blue> {}", L*cfg.rs)
    logger.opt(colors=True).info("<blue>T(Kelvin):</blue> {}", cfg.T)
    logger.opt(colors=True).info("<blue>T(Rydberg):</blue> {}", reciprocal_beta)
    logger.opt(colors=True).info("<blue>smearing sigma (Hartree):</blue> {}", smearing_sigma)
    logger.opt(colors=True).info("<blue>batchsize:</blue> {}", cfg.batchsize)
    logger.opt(colors=True).info("<blue>mc_width_p:</blue> {}", cfg.mc_width_p)
    logger.opt(colors=True).info("<blue>mc_steps_p:</blue> {}", cfg.mc_steps_p)
    logger.opt(colors=True).info("<blue>acc_steps:</blue> {}", cfg.acc_steps)


    logger.opt(colors=True).info("<yellow>\n========= Initialize potential energy surface (PES) =========</yellow>")
    if cfg.pes.type == "hf" or cfg.pes.type == "dft":
        pes_novmap = make_pes(cfg.num, L, cfg.rs, cfg.pes.basis, rcut=cfg.pes.rcut, tol=cfg.pes.tol, 
                              max_cycle=cfg.pes.max_cycle, grid_length=cfg.pes.grid_length, diis=cfg.pes.diis.diis, 
                              diis_space=cfg.pes.diis.space, diis_start_cycle=cfg.pes.diis.start_cycle, 
                              diis_damp=cfg.pes.diis.damp, use_jit=cfg.pes.use_jit, dft=(cfg.pes.type=="dft"), 
                              xc=cfg.pes.xc, smearing=cfg.pes.smearing.smearing, smearing_method=cfg.pes.smearing.method, 
                              smearing_sigma=smearing_sigma, search_method=cfg.pes.smearing.search.method, 
                              search_cycle=cfg.pes.smearing.search.cycle, search_tol=cfg.pes.smearing.search.tol,
                              gamma=cfg.pes.gamma, Gmax=cfg.pes.Gmax, kappa=cfg.pes.kappa, mode='dev')
        pes_novmap_array = lambda xp: jnp.array(pes_novmap(xp))
        pes_vmap = jax.vmap(pes_novmap_array)
        batch_num = cfg.batchsize//cfg.pes.batchsize
        if cfg.batchsize % cfg.pes.batchsize != 0:
            raise ValueError("Batch size must be divisible by PES batch size. "
                             "Got batch = %d for PES batch = %d now." % (cfg.batchsize, cfg.pes.batchsize))
        logger.opt(colors=True).info("<blue>batchsize:</blue> {}", cfg.batchsize)
        logger.opt(colors=True).info("<blue>PES batchsize:</blue> {}", cfg.pes.batchsize)
        logger.opt(colors=True).info("<blue>batch number:</blue> {}", batch_num)

        # @partial(jax.pmap, axis_name="p", in_axes=0, out_axes=(0,0,0,0,0,0,0))
        def pes(xp):
            def body_fn(carry, s):
                return carry, pes_vmap(s)
            xp = jnp.reshape(xp, (batch_num, pes_batch_per_device, cfg.num, cfg.dim))
            return jax.lax.scan(body_fn, 0, xp)[1].reshape(batch_per_device, 7)
            
        # @partial(jax.pmap, axis_name="p", in_axes=0, out_axes=0)
        # def pes(xp):
        #     for i in range(batch_num):
        #         if i == 0:
        #             pes_value = pes_vmap(xp[i*cfg.pes.batchsize:(i+1)*cfg.pes.batchsize])
        #         elif i < batch_num - 1:
        #             pes_value = jnp.concatenate((pes_value, pes_vmap(xp[i*cfg.pes.batchsize:(i+1)*cfg.pes.batchsize])))
        #         else:
        #             pes_value = jnp.concatenate((pes_value, pes_vmap(xp[i*cfg.pes.batchsize:])))
        #     return pes_value

        # jax 0.4.25 does not support "batch_size" inside jax.lax.map, use this when jax is updated
        # pes_map = jax.lax.map(pes_novmap, batch_size=cfg.pes.batchsize)
        # pes = jax.pmap(pes_map, in_axes=0, out_axes=0, axis_name="p")

    else:
        raise ValueError("Unknown PES type: %s" % cfg.pes.type)
    logger.opt(colors=True).info("<blue>PES type:</blue> {}", cfg.pes.type)
    logger.opt(colors=True).info("<blue>PES gamma:</blue> {}", cfg.pes.gamma)
    logger.opt(colors=True).info("<blue>PES basis:</blue> {}", cfg.pes.basis)
    logger.opt(colors=True).info("<blue>PES rcut:</blue> {}", cfg.pes.rcut)
    logger.opt(colors=True).info("<blue>PES tol:</blue> {}", cfg.pes.tol)
    logger.opt(colors=True).info("<blue>PES max_cycle:</blue> {}", cfg.pes.max_cycle)
    logger.opt(colors=True).info("<blue>PES grid_length:</blue> {}", cfg.pes.grid_length)
    logger.opt(colors=True).info("<blue>PES diis:</blue> {}", cfg.pes.diis.diis)
    logger.opt(colors=True).info("<blue>PES diis space:</blue> {}", cfg.pes.diis.space)
    logger.opt(colors=True).info("<blue>PES diis start_cycle:</blue> {}", cfg.pes.diis.start_cycle)
    logger.opt(colors=True).info("<blue>PES diis damp:</blue> {}", cfg.pes.diis.damp)
    logger.opt(colors=True).info("<blue>PES xc:</blue> {}", cfg.pes.xc)
    logger.opt(colors=True).info("<blue>PES smearing:</blue> {}", cfg.pes.smearing.smearing)
    logger.opt(colors=True).info("<blue>PES smearing method:</blue> {}", cfg.pes.smearing.method)
    logger.opt(colors=True).info("<blue>PES smearing sigma:</blue> {}", smearing_sigma)
    logger.opt(colors=True).info("<blue>PES smearing search method:</blue> {}", cfg.pes.smearing.search.method)
    logger.opt(colors=True).info("<blue>PES smearing search cycle:</blue> {}", cfg.pes.smearing.search.cycle)
    logger.opt(colors=True).info("<blue>PES smearing earch tol:</blue> {}", cfg.pes.smearing.search.tol)
    logger.opt(colors=True).info("<blue>PES Gmax:</blue> {}", cfg.pes.Gmax)
    logger.opt(colors=True).info("<blue>PES kappa:</blue> {}", cfg.pes.kappa)
    logger.opt(colors=True).info("<blue>PES use_jit:</blue> {}", cfg.pes.use_jit)


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


    logger.opt(colors=True).info("<yellow>\n========= Initialize optimizer =========</yellow>")
    score_fn_flow = make_classical_score(logprob_flow_novmap)
    if cfg.optimizer_flow not in ["sr", "cg"]:
        raise ValueError('Currently we assume the second-order optimizer for protons to be "sr" or "cg".')
    logger.opt(colors=True).info("<blue>flow optimizer:</blue> {}", cfg.optimizer_flow)
    logger.opt(colors=True).info("<blue>flow learning rate:</blue> {}", cfg.lr_flow)
    logger.opt(colors=True).info("<blue>flow decay:</blue> {}", cfg.decay_flow)
    logger.opt(colors=True).info("<blue>flow damping:</blue> {}", cfg.damping_flow)
    logger.opt(colors=True).info("<blue>flow maxnorm:</blue> {}", cfg.maxnorm_flow)
    logger.opt(colors=True).info("<blue>clip_factor:</blue> {}", cfg.clip_factor)
    if cfg.optimizer_flow == "cg":
        cg_params = {"mode": cfg.cg.mode,
                     "init_vec_last_step": cfg.cg.init_vec_last_step,
                     "solver_precondition": cfg.cg.solver.precondition,
                     "solver_maxiter": cfg.cg.solver.maxiter,
                     "solver_tol": cfg.cg.solver.tol,
                     "solver_style": cfg.cg.solver.style}
        logp_fn_flow_dummy = lambda x, params: logprob_p(params, x)
        score_fn_flow_dummy = lambda x, params: score_fn_flow(params, x)
        fishers_fn_flow, state_ckpt_fn_flow, optimizer_flow = classical_fisher_cg(logp_fn_flow_dummy, 
            score_fn_flow_dummy, cfg.acc_steps, cfg.gamma, cfg.lr_flow, cfg.decay_flow, 
            cfg.damping_flow, cfg.maxnorm_flow, **cg_params)
        logger.opt(colors=True).info("<blue>flow cg mode:</blue> {}", cfg.cg.mode)
        logger.opt(colors=True).info("<blue>clow cg gamma:</blue> {}", cfg.gamma)
        logger.opt(colors=True).info("<blue>flow cg init_vec_last_step:</blue> {}", cfg.cg.init_vec_last_step)
        logger.opt(colors=True).info("<blue>flow cg solver precondition:</blue> {}", cfg.cg.solver.precondition)
        logger.opt(colors=True).info("<blue>flow cg solver maxiter:</blue> {}", cfg.cg.solver.maxiter)
        logger.opt(colors=True).info("<blue>flow cg solver tol:</blue> {}", cfg.cg.solver.tol)
        logger.opt(colors=True).info("<blue>flow cg solver style:</blue> {}", cfg.cg.solver.style)
    elif cfg.optimizer_flow == "sr":
        block_fn_flow = pytree_as_whole_block(params_flow)
        fishers_fn_flow, state_ckpt_fn_flow, optimizer_flow = classical_fisher_sr(score_fn_flow, block_fn_flow,
                    cfg.acc_steps, cfg.alpha, cfg.lr_flow, cfg.decay_flow, cfg.damping_flow, cfg.maxnorm_flow)
        logger.opt(colors=True).info("<blue>alpha:</blue> {}", cfg.alpha)
    else:
        raise ValueError("Both the optimizer for protons and electrons should be either sr/fac/cg or adam. "
                        'Got optimizer_flow = "%s" devices now.' % (cfg.optimizer_flow))
    # flow optimizer init
    if cfg.optimizer_flow == "cg":
        s_dummy = jnp.empty((num_devices, batch_per_device, cfg.num, cfg.dim))
        opt_state_flow = jax.pmap(optimizer_flow.init,
                                  in_axes=(None, 0),
                                  out_axes=0)(params_flow, s_dummy)
        opt_state_flow_pmap_axis = 0
    else:
        opt_state_flow = optimizer_flow.init(params_flow)
        opt_state_flow_pmap_axis = None


    logger.opt(colors=True).info("<yellow>\n========= Checkpointing =========</yellow>")
    if ckpt_filename is not None:
        logger.opt(colors=True).info("<green>Load checkpoint file:</green> {}", ckpt_filename)
        logger.opt(colors=True).info("<blue>epoch_finished:</blue> {}", epoch_finished)
        ckpt = load_data(ckpt_filename)
        keys, s, params_flow = ckpt["keys"], ckpt["s"], ckpt["params_flow"]
        opt_state_flow_ckpt = jax.tree_util.tree_map(lambda x: replicate(x[0], num_devices), ckpt["opt_state_flow"]) \
                        if opt_state_flow_pmap_axis is not None else ckpt["opt_state_flow"]
        opt_state_flow.update(opt_state_flow_ckpt)
        logger.opt(colors=True).info("Successfully load key, s, params_flowc opt_state_flow.")
        if num_hosts > 1:
            keys = jax.random.split(keys[0], (num_hosts, num_devices))
            if (s.size == num_hosts*num_devices*batch_per_device*cfg.num*cfg.dim):
                s = jnp.reshape(s, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim))
            else:    
                keys, subkeys = p_split(keys)
                s = jax.pmap(jax.random.uniform, static_broadcasted_argnums=(1,2,3,4))(subkeys, (batch_per_device, cfg.num, cfg.dim), 0., L)
                epoch_finished = 0 
            s = s[jax.process_index()]
            keys = keys[jax.process_index()]
        else:
            keys = jax.random.split(keys[0], num_devices)
            if (s.size == num_devices*batch_per_device*cfg.num*cfg.dim):
                s = jnp.reshape(s, (num_devices, batch_per_device, cfg.num, cfg.dim))
            else:    
                keys, subkeys = p_split(keys)
                s = jax.pmap(jax.random.uniform, static_broadcasted_argnums=(1,2,3,4))(subkeys, (batch_per_device, cfg.num, cfg.dim), 0., L)
                epoch_finished = 0
        s, keys = shard(s), shard(keys)
        logger.opt(colors=True).info("Successfully load key and s.")
        logger.opt(colors=True).info("<blue>s.shape:</blue> {}", s.shape)
        logger.opt(colors=True).info("<blue>s.type:</blue> {}", type(s))
        params_flow = replicate((params_flow), num_devices)
        try:
            mc_width_p = ckpt["mc_width_p"]
        except (NameError, KeyError):
            mc_width_p = cfg.mc_width_p
    else:
        logger.opt(colors=True).info("Initializing key and s...")
        key, key_proton = jax.random.split(key)
        if num_hosts > 1:
            s = jax.random.uniform(key_proton, (num_hosts, num_devices, batch_per_device, cfg.num, cfg.dim), minval=0., maxval=L)
            keys = jax.random.split(key, (num_hosts, num_devices))
            s = s[jax.process_index()]
            keys = keys[jax.process_index()]
        else:
            s = jax.random.uniform(key_proton, (num_devices, batch_per_device, cfg.num, cfg.dim), minval=0., maxval=L)
            keys = jax.random.split(key, num_devices)
        s, keys = shard(s), shard(keys)
        logger.opt(colors=True).info("Successfully initialized key and s.")
        logger.opt(colors=True).info("<blue>s.shape:</blue> {}", s.shape)
        logger.opt(colors=True).info("<blue>s.type:</blue> {}", type(s))
        params_flow = replicate((params_flow), num_devices)
        mc_width_p = cfg.mc_width_p
        epoch_finished = 0
    
        
    if epoch_finished == 0:
        logger.opt(colors=True).info("<yellow>\n========= Thermalization =========</yellow>")
        for i in range(cfg.mc_therm):
            logger.opt(colors=True).info("---- thermal step {} ----", i+1)
            keys, s, ar_s = sample_s(keys, logprob_p, force_fn_p, s, params_flow, cfg.mc_steps_p, mc_width_p, L)
            logger.opt(colors=True).info('<blue>proton acc:</blue> {}', jnp.mean(ar_s))
            logger.opt(colors=True).info('<blue>proton entropy:</blue> {}', -jax.pmap(logprob_p)(params_flow, s).mean()/cfg.num)
        
    
    logger.opt(colors=True).info("<yellow>\n========= Start pretraining =========</yellow>")
    observable_and_lossfn = make_loss_pretrain_flow(logprob_p, pes, L, cfg.rs, reciprocal_beta, cfg.clip_factor)

    @partial(jax.pmap, axis_name="p",
             in_axes=(0, opt_state_flow_pmap_axis, 0, 0, 0, 0, None),  
             out_axes=(0, opt_state_flow_pmap_axis, 0, 0), 
             static_broadcasted_argnums=6)
    def update(params_flow, opt_state_flow, s, key, data_acc, grad_flow_acc, final_step):
        data, flow_lossfn = observable_and_lossfn(params_flow, s)
        flow_grad = jax.grad(flow_lossfn)(params_flow)
        flow_grad = jax.lax.pmean((flow_grad), axis_name="p")
        data_acc, grad_flow_acc = jax.tree_util.tree_map(lambda acc, i: acc + i, 
                                                        (data_acc, grad_flow_acc),  
                                                        (data, flow_grad))
        opt_state_flow = fishers_fn_flow(params_flow, s, opt_state_flow)
        if final_step:
            data_acc, grad_flow_acc = jax.tree_util.tree_map(lambda acc: acc / cfg.acc_steps, 
                                                             (data_acc, grad_flow_acc))
            
            # jax.debug.print("grad_flow_acc fermi_net linear_4:\n {x}", x=grad_flow_acc['fermi_net/~/linear_4']['w'])

            # jax.debug.print("opt_state_flow:\n {x}", x=opt_state_flow)

            update_flow, opt_state_flow = optimizer_flow.update(grad_flow_acc, opt_state_flow, params_flow)

            # jax.debug.print("update flow: {x}", x=update_flow)
            # jax.debug.print("update params_flow fermi_net linear_4:\n {x}", x=update_flow['fermi_net/~/linear_4']['w'])

            params_flow = optax.apply_updates(params_flow, update_flow)

            # jax.debug.print("params_flow fermi_net linear_4:\n {x}", x=params_flow['fermi_net/~/linear_4']['w'])

        return params_flow, opt_state_flow, data_acc, grad_flow_acc

    time_of_last_ckpt = time.time()
    data_filename = os.path.join(path, "data.txt")
    f = open(data_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")

    csv_filename = os.path.join(path, "data.csv")
    csv_file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(csv_file)

    if os.path.getsize(data_filename)==0:
        f.write("epoch f f_err e e_err k k_err vep vep_err vee vee_err vpp vpp_err p p_err ep_cov sp sp_err se se_err acc_s convergence\n")
        writer.writerows([['epoch', 'f', 'f_err', 'e', 'e_err', 'k', 'k_err', 
                           'vep', 'vep_err', 'vee', 'vee_err', 'vpp', 'vpp_err', 
                           'p', 'p_err', 'ep_cov', 'sp', 'sp_err', 'se', 'se_err', 
                           'acc_s', 'convergence']])
    for i in range(epoch_finished + 1, cfg.epoch + 1):
        data_acc = replicate({"F": 0., "F2": 0.,
                              "E": 0., "E2": 0.,
                              "K": 0., "K2": 0.,
                              "Vep": 0., "Vep2": 0.,
                              "Vee": 0., "Vee2": 0.,
                              "Vpp": 0., "Vpp2": 0.,
                              "P": 0., "P2": 0.,
                              "ep": 0.,
                              "Sp": 0., "Sp2": 0.,
                              "Se": 0., "Se2": 0.,
                              "convergence": 0.}, num_devices)
        grad_flow_acc = shard(jax.tree_map(jnp.zeros_like, params_flow)) # check
        ar_s_acc = shard(jnp.zeros(num_devices))
        for acc in range(cfg.acc_steps):
            keys, s, ar_s = sample_s(keys, logprob_p, force_fn_p, s, params_flow, cfg.mc_steps_p, mc_width_p, L)
            ar_s_acc += ar_s/cfg.acc_steps
            final_step = (acc == cfg.acc_steps - 1)
            params_flow, opt_state_flow, data_acc, grad_flow_acc = update(params_flow, opt_state_flow, s, keys, 
                                                                          data_acc, grad_flow_acc, final_step)
        data = jax.tree_map(lambda x: x[0], data_acc)
        ar_s = ar_s_acc[0] 
        F, F2, E, E2, K, K2, Vep, Vep2, Vee, Vee2, Vpp, Vpp2, \
        P, P2, ep, Sp, Sp2, Se, Se2, convergence = \
                                data["F"], data["F2"], \
                                data["E"], data["E2"], \
                                data["K"], data["K2"], \
                                data["Vep"], data["Vep2"], \
                                data["Vee"], data["Vee2"], \
                                data["Vpp"], data["Vpp2"], \
                                data["P"], data["P2"], \
                                data["ep"], \
                                data["Sp"], data["Sp2"], \
                                data["Se"], data["Se2"], \
                                data["convergence"]
        F_std = jnp.sqrt((F2- F**2) / (cfg.batchsize*cfg.acc_steps))
        E_std = jnp.sqrt((E2- E**2) / (cfg.batchsize*cfg.acc_steps))
        K_std = jnp.sqrt((K2- K**2) / (cfg.batchsize*cfg.acc_steps))
        Vep_std = jnp.sqrt((Vep2- Vep**2) / (cfg.batchsize*cfg.acc_steps))
        Vee_std = jnp.sqrt((Vee2- Vee**2) / (cfg.batchsize*cfg.acc_steps))
        Vpp_std = jnp.sqrt((Vpp2- Vpp**2) / (cfg.batchsize*cfg.acc_steps))
        P_std = jnp.sqrt((P2- P**2) / (cfg.batchsize*cfg.acc_steps))
        ep_cov = (ep- E*P) / (cfg.batchsize*cfg.acc_steps)
        Sp_std = jnp.sqrt((Sp2- Sp**2) / (cfg.batchsize*cfg.acc_steps))
        Se_std = jnp.sqrt((Se2- Se**2) / (cfg.batchsize*cfg.acc_steps))
        # Note the quantities with energy dimension has a prefactor 1/rs^2
        logger.opt(colors=True).info("<blue>iter:</blue> %04d " % i + \
            "<blue>F:</blue> {} <blue>F_std:</blue> {} <blue>E:</blue> {} <blue>E_std:</blue> {} " + \
            "<blue>Sp:</blue> {} <blue>Sp_std:</blue> {} <blue>accept_rate:</blue> {} <blue>converged_rate:</blue> {}", 
            F/cfg.num/cfg.rs**2, F_std/cfg.num/cfg.rs**2, 
            E/cfg.num/cfg.rs**2, E_std/cfg.num/cfg.rs**2, 
            Sp/cfg.num, Sp_std/cfg.num, ar_s, convergence)
        f.write( ("%6d" + "  %.6f"*19 + "  %.4f"*2 + "\n") % (i, F/cfg.num/cfg.rs**2, F_std/cfg.num/cfg.rs**2,
                                                                 E/cfg.num/cfg.rs**2, E_std/ cfg.num/cfg.rs**2,
                                                                 K/cfg.num/cfg.rs**2, K_std/cfg.num/cfg.rs**2,
                                                                 Vep/cfg.num/cfg.rs**2, Vep_std/cfg.num/cfg.rs**2,
                                                                 Vee/cfg.num/cfg.rs**2, Vee_std/cfg.num/cfg.rs**2,
                                                                 Vpp/cfg.num/cfg.rs**2, Vpp_std/cfg.num/cfg.rs**2,
                                                                 P/cfg.rs**2, P_std/cfg.rs**2,
                                                                 ep_cov/cfg.num/cfg.rs**4,
                                                                 Sp/cfg.num, Sp_std/cfg.num, 
                                                                 Se/cfg.num, Se_std/cfg.num,
                                                                 ar_s, convergence))
        writer.writerows([[i, format(F/cfg.num/cfg.rs**2, '.6f'), format(F_std/cfg.num/cfg.rs**2, '.6f'), 
                              format(E/cfg.num/cfg.rs**2, '.6f'), format(E_std/cfg.num/cfg.rs**2, '.6f'), 
                              format(K/cfg.num/cfg.rs**2, '.6f'), format(K_std/cfg.num/cfg.rs**2, '.6f'), 
                              format(Vep/cfg.num/cfg.rs**2, '.6f'), format(Vep_std/cfg.num/cfg.rs**2, '.6f'), 
                              format(Vee/cfg.num/cfg.rs**2, '.6f'), format(Vee_std/cfg.num/cfg.rs**2, '.6f'), 
                              format(Vpp/cfg.num/cfg.rs**2, '.6f'), format(Vpp_std/cfg.num/cfg.rs**2, '.6f'), 
                              format(P/cfg.rs**2, '.6f'), format(P_std/cfg.rs**2, '.6f'), 
                              format(ep_cov/cfg.num/cfg.rs**4, '.6f'), 
                              format(Sp/cfg.num, '.6f'), format(Sp_std/cfg.num, '.6f'), 
                              format(Se/cfg.num, '.6f'), format(Se_std/cfg.num, '.6f'), 
                              format(ar_s, '.4f'), format(convergence, '.4f')]])
        csv_file.flush()
        if time.time() - time_of_last_ckpt > 600:
            opt_state_flow_ckpt = state_ckpt_fn_flow(opt_state_flow)
            ckpt = {"keys": keys,
                    "s": s,
                    "params_flow": jax.tree_map(lambda x: x[0], params_flow),
                    "opt_state_flow": jax.tree_util.tree_map(lambda x: x[0], all_gather(opt_state_flow_ckpt, "p")) 
                                if opt_state_flow_pmap_axis is not None else opt_state_flow_ckpt, 
                    "mc_width_p": mc_width_p, 
                }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %i)
            save_data(ckpt, ckpt_filename)
            logger.opt(colors=True).info("<green>Save checkpoint file:</green> {}", ckpt_filename)
            time_of_last_ckpt = time.time()
        if jnp.isnan(F):
            raise RuntimeError("Free energy is nan")
        if ar_s < 1e-7:
            raise RuntimeError("Acceptance rate nearly zero")
        if i % 100 == 0:
            mc_width_p = adjust_mc_width(mc_width_p, ar_s, "mcmc")
            logger.opt(colors=True).info("adjust mc width p to {}", mc_width_p) 
    f.close()
    logger.opt(colors=True).info("<yellow>\n========= Pretraining finished =========</yellow>")

if __name__ == "__main__": 
    main_func()
