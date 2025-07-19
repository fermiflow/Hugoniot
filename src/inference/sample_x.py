import time
import math
import haiku as hk
from functools import partial
from typing import List, Union

from src.flow import make_flow
from src.ad import make_grad_real
from src.ferminet import FermiNet
from src.inference.utils import *
from src.utils import shard, replicate
from src.checkpoint import load_data, save_data
from src.inference.parser import parse_filename
from src.inference.colors import default_colors
from src.inference.find_ckpt_sample import find_ckpt_sample
from src.inference.sample_s import get_gr
from src.autoregressive import Transformer
from src.sampler import make_autoregressive_sampler
from src.logpsi import make_logpsi, make_logpsi2, make_logpsi_grad_laplacian
from src.vmc import sample_s_and_x
from src.potential import potential_energy, kpoints, Madelung

from hqc.pbc.lcao import make_lcao
from hqc.pbc.slater import make_slater


def make_obs(logprob_p, logprob_e, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta, steps):
    
    def observable_fn(params_flow, params_van, params_wfn, state_idx, mo_coeff, bands, s, x, key):
        '''
            s: (B, n, dim)
            x: (B, n, dim)
        '''
        batchsize = s.shape[0]
        assert batchsize % steps == 0
        eval_grad_lap_batchsize = batchsize // steps
        for step in range(steps):
            grad_acc, laplacian_acc = logpsi_grad_laplacian(x[step*eval_grad_lap_batchsize:(step+1)*eval_grad_lap_batchsize], params_wfn, 
                                                    s[step*eval_grad_lap_batchsize:(step+1)*eval_grad_lap_batchsize], 
                                                    state_idx[step*eval_grad_lap_batchsize:(step+1)*eval_grad_lap_batchsize], 
                                                    mo_coeff[step*eval_grad_lap_batchsize:(step+1)*eval_grad_lap_batchsize], key)
            grad = jnp.concatenate((grad, grad_acc), axis=0) if step > 0 else grad_acc
            laplacian = jnp.concatenate((laplacian, laplacian_acc), axis=0) if step > 0 else laplacian_acc

        logp_e = logprob_e(params_van, state_idx, bands) # (B, )
        logp_p = logprob_p(params_flow, s) # (B,)
        
        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
        v_pp, v_ep, v_ee = potential_energy(jnp.concatenate([s, x], axis=1), kappa, G, L, rs) 
        v_pp += Vconst
        v_ee += Vconst
        Eloc = kinetic + (v_ep + v_ee) # (B, ) 
        Floc = (logp_p + logp_e)*rs**2/ beta + Eloc.real + v_pp # (B,)
        P = (2*kinetic.real + (v_pp + v_ep + v_ee) )/(3*(L*rs)**3)* 14710.513242194795

        K = kinetic.real
        Eloc = Eloc.real
        Sp = -logp_p
        Se = -logp_e

        return K, v_pp, v_ep, v_ee, P, Eloc, Floc, Sp, Se

    return observable_fn

def sample_sx(files: List[str], 
              files_draw_new_sample: Union[List[bool], bool] = True,
              labels: Union[List[str], None] = None,
              auto_load: bool = True,
              auto_save: bool = True,
              sample_total_batch: int = 1024,
              sample_batch: int = 256,
              eval_grad_lap_step: int = 4,
              hf_batch: int = 128,
              hf_grid_length: float = 0.5,
              sample_therm: int = 10,
              sample_mc_proton_steps: Union[int, None] = None,
              sample_mc_proton_width: Union[float, None] = None,
              sample_mc_electron_steps: Union[int, None] = None,
              sample_mc_electron_width: Union[float, None] = None,
              savefigname: Union[str, None] = None,
              bins: int = 100, 
              alpha: float = 1.0,
              figsize: tuple = (10,7), 
              dpi: int = 300,
              legend_loc: List[str] = ['lower right', 'upper right', 'upper right', 'lower right'],
              grid: bool = True,
              dark_mode: bool = False,
              max_batch_mode: bool = False,
    ):
    """
        Draw samples of s and x from the checkpoint files.
    Args:
        files: list of checkpoint files to load.
        files_draw_new_sample: if draw new samples from list of checkpoint files, list of bool.
        labels: list of labels for each file, used in ploting when savefigname is not None.
        auto_load: if True, will automatically load the biggest checkpoint file in the directory.
        auto_save: if True, will automatically save the samples to a file.
        sample_total_batch: total number of samples to draw.
        sample_batch: number of samples to draw in each sample epoch.
        eval_grad_lap_step: number of steps to evaluate the gradient and laplacian in each sample epoch.
        hf_batch: Hartree-Fock batchsize in each sample epoch.
        hf_grid_length: grid length for Hartree-Fock calculation.
        sample_therm: number of thermalization steps.
        sample_mc_proton_steps: number of MC steps for proton sampling in each sample epoch.
        sample_mc_proton_width: MC sample width for proton sampling.
        sample_mc_electron_steps: number of MC steps for electron sampling in each sample epoch.
        sample_mc_electron_width: MC sample width for electron sampling.
        savefigname: if not None, will save the figure to this file.
        bins: number of bins for RDF calculation.
        alpha: non-linear parameter for RDF calculation, when alpha=1, bins are still linear-distributed.
        figsize: figure size for the plot.
        dpi: dpi for the plot.
        legend_loc: list of legend locations for each subplot.
        grid: if True, will show grid in the plot.
        dark_mode: if True, will use dark mode for the plot.
        max_batch_mode: if True, will use the biggest checkpoint file in the directory and 
                        set sample_total_batch to the batch size of the biggest checkpoint file.
    """
    if isinstance(files_draw_new_sample, bool):
        files_draw_new_sample = [files_draw_new_sample] * len(files)
    else:
        assert len(files) == len(files_draw_new_sample)

    if dark_mode:
        facecolor = (0.2, 0.2, 0.2)
        fontcolor = 'white'
    else:
        facecolor = 'white'
        fontcolor = 'black'

    colors = default_colors

    if labels is not None:
        assert len(labels) == len(files)

    if savefigname is not None:
        if not savefigname.endswith('.png') and not savefigname.endswith('.pdf'):
            savefigname += '.png'
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi, facecolor=facecolor)
        axes[0,0].set_facecolor(facecolor) 
        axes[0,0].spines['bottom'].set_color(fontcolor)
        axes[0,0].spines['top'].set_color(fontcolor)
        axes[0,0].spines['left'].set_color(fontcolor)
        axes[0,0].spines['right'].set_color(fontcolor)
        axes[0,0].tick_params(axis='both', colors=fontcolor)
        axes[0,0].set_xlabel("$r(Bohr)$", color=fontcolor)
        axes[0,0].set_ylabel("g(r)", color=fontcolor)
        axes[0,0].set_ylim(-0.13, 1.23)
        if grid:
            axes[0,0].grid(color='gray', linestyle='--', alpha=0.5 )

        axes[0,1].set_facecolor(facecolor) 
        axes[0,1].spines['bottom'].set_color(fontcolor)
        axes[0,1].spines['top'].set_color(fontcolor)
        axes[0,1].spines['left'].set_color(fontcolor)
        axes[0,1].spines['right'].set_color(fontcolor)
        axes[0,1].tick_params(axis='both', colors=fontcolor)
        axes[0,1].set_xlabel("$r(Bohr)$", color=fontcolor)
        axes[0,1].set_ylabel("g(r)", color=fontcolor)
        axes[0,1].set_ylim(0, 10)
        if grid:
            axes[0,1].grid(color='gray', linestyle='--', alpha=0.5)
        
        axes[1,0].set_facecolor(facecolor) 
        axes[1,0].spines['bottom'].set_color(fontcolor)
        axes[1,0].spines['top'].set_color(fontcolor)
        axes[1,0].spines['left'].set_color(fontcolor)
        axes[1,0].spines['right'].set_color(fontcolor)
        axes[1,0].tick_params(axis='both', colors=fontcolor)
        axes[1,0].set_xlabel("$r(Bohr)$", color=fontcolor)
        axes[1,0].set_ylabel("g(r)", color=fontcolor)
        axes[1,0].set_ylim(0.8, 1.9)
        if grid:
            axes[1,0].grid(color='gray', linestyle='--', alpha=0.5)
        
        axes[1,1].set_facecolor(facecolor) 
        axes[1,1].spines['bottom'].set_color(fontcolor)
        axes[1,1].spines['top'].set_color(fontcolor)
        axes[1,1].spines['left'].set_color(fontcolor)
        axes[1,1].spines['right'].set_color(fontcolor)
        axes[1,1].tick_params(axis='both', colors=fontcolor)
        axes[1,1].set_xlabel("$r(Bohr)$", color=fontcolor)
        axes[1,1].set_ylabel("g(r)", color=fontcolor)
        axes[1,1].set_ylim(-0.13, 1.2)
        if grid:
            axes[1,1].grid(color='gray', linestyle='--', alpha=0.5)
    
    for f in files:
    
        directory = os.path.dirname(f)
        cfg_filename = os.path.join(directory, 'config.yaml')
        if os.path.isfile(cfg_filename):
            cfg = OmegaConf.load(cfg_filename)
            rs = cfg.rs
            n = cfg.num
            T = cfg.T
            dim = cfg.dim
            Nf = cfg.flow.Nf
            basis = cfg.lcao.basis
            flow_steps = cfg.flow.steps
            flow_depth = cfg.flow.depth
            flow_h1size = cfg.flow.h1size
            flow_h2size = cfg.flow.h2size
            nlayers = cfg.van.nlayers
            modelsize = cfg.van.modelsize
            nheads = cfg.van.nheads
            nhidden = cfg.van.nhidden
            wfn_depth = cfg.wfn.depth
            wfn_h1size = cfg.wfn.h1size
            wfn_h2size = cfg.wfn.h2size
            Gmax = cfg.ewald.Gmax
            kappa = cfg.ewald.kappa

        else:
            params = parse_filename(f)
            rs = params["rs"]
            n = params["n"]
            T = params["T"]
            dim = params["dim"]
            Nf = params["Nf"]
            flow_steps = params["flow_steps"]
            flow_depth = params["flow_depth"]
            flow_h1size = params["flow_h1size"]
            flow_h2size = params["flow_h2size"]
            nlayers = params["nlayers"]
            modelsize = params["modelsize"]
            nheads = params["nheads"]
            nhidden = params["nhidden"]
            wfn_depth = params["wfn_depth"]
            wfn_h1size = params["wfn_h1size"]
            wfn_h2size = params["wfn_h2size"]
            Gmax = params["Gmax"]
            kappa = params["kappa"]
            basis = 'gth-dzv'

        if labels is None:
            label = r'$n=%g,rs=%g,T=%g$'%(n,  rs, T)
        else:
            label = labels[files.index(f)]
        color = colors[files.index(f)]
    
        beta = 157888.088922572/T # inverse temperature in unit of 1/Ry
        smearing_sigma = 1/beta/2 # temperature in Hartree unit
        kpt = jnp.array([0,0,0])
        L = (4/3*jnp.pi*n)**(1/3)
        key = jax.random.PRNGKey(42)

        # lcao
        lcao = make_lcao(n, L, rs, basis=basis, grid_length=hf_grid_length, 
                        smearing=True, smearing_method='fermi', smearing_sigma=smearing_sigma)
        lcao_orbitals = make_slater(n, L, rs, basis=basis, groundstate=False)

        s = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
        mo_coeff, bands, _ = lcao(s)
        num_states = bands.shape[0]
        lcao_map = lambda xp: jax.lax.map(lcao, xp)[0:2]
        lcao_map_vmap = jax.vmap(lcao_map, 0, (0, 0))
        if hf_batch > sample_batch:
            hf_batch = sample_batch
        lcao_mapsize = sample_batch // hf_batch
        def lcao_vmc(xp):
            xp = xp.reshape(-1, lcao_mapsize, n, dim)
            mo_coeff, bands = lcao_map_vmap(xp)
            return mo_coeff.reshape(-1, num_states, num_states), bands.reshape(-1, num_states)

        # flow model
        def forward_fn(x):
            for _ in range(flow_steps):
                model = FermiNet(flow_depth, flow_h1size, flow_h2size, Nf, L, False)
                x = model(x)
            return x
        network_flow = hk.transform(forward_fn)

        logprob_flow_novmap = make_flow(network_flow, n, dim, L)
        vmap_p = partial(jax.vmap, in_axes=(None, 0), out_axes=0)
        logprob_p = vmap_p(logprob_flow_novmap)
        force_fn_p = vmap_p(make_grad_real(logprob_flow_novmap, argnums=1))

        # van
        def forward_fn(state):
            model = Transformer(num_states, nlayers, modelsize, nheads, nhidden, remat=False)
            return model(state)
        van = hk.transform(forward_fn)

        sampler, logprob_van_novmap = make_autoregressive_sampler(van, n, num_states, beta)
        logprob_e = jax.vmap(logprob_van_novmap, (None, 0, 0), 0)

        # wfn
        def forward_fn(x):
            model = FermiNet(wfn_depth, wfn_h1size, wfn_h2size, Nf, L, True, remat=False)
            return model(x)
        network_wfn = hk.transform(forward_fn)

        logpsi_novmap = make_logpsi(network_wfn, lcao_orbitals, kpt)
        logpsi2_novmap = make_logpsi2(logpsi_novmap)
        vmap_wfn = partial(jax.vmap, in_axes=(0, None, 0, 0, 0), out_axes=0)
        logpsi2 = vmap_wfn(logpsi2_novmap)
        force_fn_e = vmap_wfn(make_grad_real(logpsi2_novmap))

        # observable function
        G = kpoints(dim, Gmax)
        Vconst = n * rs/L * Madelung(dim, kappa, G)
        logpsi, logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi_novmap, hutchinson=False)
        observable_fn = make_obs(logprob_p, logprob_e, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta, eval_grad_lap_step)
        observable_fn = jax.pmap(observable_fn, axis_name="p", in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))

        print(f"{YELLOW}========= Loading checkpoint ========{RESET}")
        # auto load the biggest checkpoint file
        if auto_load:
            auto_find_f, auto_find_f_batch = find_ckpt_sample(f, mode="sx", silent_mode=True)
            if auto_find_f is not None:
                ckpt_file = auto_find_f
                print(f"{GREEN}auto load the biggest ckeckpoint sample file:\n{RESET}File:", ckpt_file)
                if max_batch_mode and (auto_find_f_batch > sample_total_batch) and (not files_draw_new_sample[files.index(f)]):
                    sample_total_batch_thisfile = auto_find_f_batch
                else:
                    sample_total_batch_thisfile = sample_total_batch
            else:
                ckpt_file = f
                sample_total_batch_thisfile = sample_total_batch
                print(f"{GREEN}no ckeckpoint sample file found, load checkpoint file:\n{RESET}File:", ckpt_file)
        else:
            ckpt_file = f
            print(f"{GREEN}load checkpoint file:\n{RESET}File:", ckpt_file)

        data = load_data(ckpt_file)
        keys = data["keys"]
        s = data["s"]
        x = data["x"]
        state_idx = data["state_idx"]
        mo_coeff = data["mo_coeff"]
        bands = data["bands"]
        params_flow = data["params_flow"]
        params_van = data["params_van"]
        params_wfn = data["params_wfn"]

        if sample_mc_proton_width is None:
            mc_proton_width = data["mc_width_p"]
        else:
            mc_proton_width = sample_mc_proton_width
        if sample_mc_electron_width is None:
            mc_electron_width = data["mc_width_e"]
        else:
            mc_electron_width = sample_mc_electron_width
        if sample_mc_proton_steps is None:
            mc_proton_steps = cfg.mc.steps_p
        else:
            mc_proton_steps = sample_mc_proton_steps
        if sample_mc_electron_steps is None:
            mc_electron_steps = cfg.mc.steps_e
        else:
            mc_electron_steps = sample_mc_electron_steps

        num_devices = jax.device_count()
        batch_per_device = sample_batch // num_devices

        params_flow = replicate((params_flow), num_devices)
        params_van = replicate((params_van), num_devices)
        params_wfn = replicate((params_wfn), num_devices)

        s = jnp.reshape(s, (-1, n, dim))
        x = jnp.reshape(x, (-1, n, dim))
        state_idx = jnp.reshape(state_idx, (-1, n))
        mo_coeff = jnp.reshape(mo_coeff, (-1, num_states, num_states))
        bands = jnp.reshape(bands, (-1, num_states))
        
        s_data = s * rs
        x_data = x * rs
        state_idx_data = state_idx
        mo_coeff_data = mo_coeff
        bands_data = bands

        keys = jax.random.split(keys[0], num_devices)
        load_batch = s.shape[0]
        print(f"{MAGENTA}load batch:{RESET}", load_batch)
        print(f"{MAGENTA}sample total batch:{RESET}", sample_total_batch_thisfile)
        print(f"{MAGENTA}if draw new sample:{RESET}", files_draw_new_sample[files.index(f)])

        if "F" in data:
            K_data = data["K"]
            Vpp_data = data["Vpp"]
            Vep_data = data["Vep"]
            Vee_data = data["Vee"]
            P_data = data["P"]
            E_data = data["E"]
            F_data = data["F"]
            Sp_data = data["Sp"]
            Se_data = data["Se"]
        else:
            if load_batch <= sample_batch:
                eval_steps = 1
            elif load_batch % sample_batch == 0:
                eval_steps = load_batch // sample_batch
            else:
                eval_steps = load_batch // sample_batch + 1
            print(f"{MAGENTA}Evaluating checkpoint total steps:{RESET}", eval_steps)
            
            for i in range(eval_steps):
                if i < eval_steps - 1:
                    s_ckpt = s[i*sample_batch:(i+1)*sample_batch].reshape(num_devices, batch_per_device, n, dim)
                    x_ckpt = x[i*sample_batch:(i+1)*sample_batch].reshape(num_devices, batch_per_device, n, dim)
                    state_idx_ckpt = state_idx[i*sample_batch:(i+1)*sample_batch].reshape(num_devices, batch_per_device, n)
                    mo_coeff_ckpt = mo_coeff[i*sample_batch:(i+1)*sample_batch].reshape(num_devices, batch_per_device, num_states, num_states)
                    bands_ckpt = bands[i*sample_batch:(i+1)*sample_batch].reshape(num_devices, batch_per_device, num_states)
                else:
                    batch_temp = (load_batch - i*sample_batch) // num_devices
                    s_ckpt = s[i*sample_batch:].reshape(num_devices, batch_temp, n, dim)
                    x_ckpt = x[i*sample_batch:].reshape(num_devices, batch_temp, n, dim)
                    state_idx_ckpt = state_idx[i*sample_batch:].reshape(num_devices, batch_temp, n)
                    mo_coeff_ckpt = mo_coeff[i*sample_batch:].reshape(num_devices, batch_temp, num_states, num_states)
                    bands_ckpt = bands[i*sample_batch:].reshape(num_devices, batch_temp, num_states)

                K, Vpp, Vep, Vee, P, E, F, Sp, Se = observable_fn(params_flow, params_van, params_wfn, state_idx_ckpt, 
                                                                  mo_coeff_ckpt, bands_ckpt, s_ckpt, x_ckpt, keys)
                if i == 0:
                    K_data = K.reshape(-1)/n/rs**2
                    Vpp_data = Vpp.reshape(-1)/n/rs**2
                    Vep_data = Vep.reshape(-1)/n/rs**2
                    Vee_data = Vee.reshape(-1)/n/rs**2
                    P_data = P.reshape(-1)/rs**2
                    E_data = E.reshape(-1)/n/rs**2
                    F_data = F.reshape(-1)/n/rs**2
                    Sp_data = Sp.reshape(-1)/n
                    Se_data = Se.reshape(-1)/n
                else:
                    K_data = jnp.concatenate((K_data, K.reshape(-1)/n/rs**2), axis=0)
                    Vpp_data = jnp.concatenate((Vpp_data, Vpp.reshape(-1)/n/rs**2), axis=0)
                    Vep_data = jnp.concatenate((Vep_data, Vep.reshape(-1)/n/rs**2), axis=0)
                    Vee_data = jnp.concatenate((Vee_data, Vee.reshape(-1)/n/rs**2), axis=0)
                    P_data = jnp.concatenate((P_data, P.reshape(-1)/rs**2), axis=0)
                    E_data = jnp.concatenate((E_data, E.reshape(-1)/n/rs**2), axis=0)
                    F_data = jnp.concatenate((F_data, F.reshape(-1)/n/rs**2), axis=0)
                    Sp_data = jnp.concatenate((Sp_data, Sp.reshape(-1)/n), axis=0)
                    Se_data = jnp.concatenate((Se_data, Se.reshape(-1)/n), axis=0)
        
        K_mean, K_std = K_data.mean(), K_data.std()/jnp.sqrt(load_batch)
        Vpp_mean, Vpp_std = Vpp_data.mean(), Vpp_data.std()/jnp.sqrt(load_batch)
        Vep_mean, Vep_std = Vep_data.mean(), Vep_data.std()/jnp.sqrt(load_batch)
        Vee_mean, Vee_std = Vee_data.mean(), Vee_data.std()/jnp.sqrt(load_batch)
        P_mean, P_std = P_data.mean(), P_data.std()/jnp.sqrt(load_batch)
        E_mean, E_std = E_data.mean(), E_data.std()/jnp.sqrt(load_batch)
        F_mean, F_std = F_data.mean(), F_data.std()/jnp.sqrt(load_batch)
        Sp_mean, Sp_std = Sp_data.mean(), Sp_data.std()/jnp.sqrt(load_batch)
        Se_mean, Se_std = Se_data.mean(), Se_data.std()/jnp.sqrt(load_batch)

        print(f"{MAGENTA}checkpoint F:{RESET} %.4f ± %.4f" % (F_mean, F_std), f"{MAGENTA} E:{RESET} %.4f ± %.4f" % (E_mean, E_std), 
                        f"{MAGENTA} P:{RESET} %.4f ± %.4f" % (P_mean, P_std), f"{MAGENTA} Se:{RESET} %.4f ± %.4f" % (Se_mean, Se_std))

        if load_batch >= sample_total_batch_thisfile and (not files_draw_new_sample[files.index(f)]):

            s_data = s_data[:sample_total_batch_thisfile]
            x_data = x_data[:sample_total_batch_thisfile]
            state_idx_data = state_idx_data[:sample_total_batch_thisfile]
            mo_coeff_data = mo_coeff_data[:sample_total_batch_thisfile]
            bands_data = bands_data[:sample_total_batch_thisfile]
            K_data = K_data[:sample_total_batch_thisfile]
            Vpp_data = Vpp_data[:sample_total_batch_thisfile]
            Vep_data = Vep_data[:sample_total_batch_thisfile]
            Vee_data = Vee_data[:sample_total_batch_thisfile]
            P_data = P_data[:sample_total_batch_thisfile]
            E_data = E_data[:sample_total_batch_thisfile]
            F_data = F_data[:sample_total_batch_thisfile]
            Sp_data = Sp_data[:sample_total_batch_thisfile]
            Se_data = Se_data[:sample_total_batch_thisfile]

            # plotting
            if savefigname is not None:
                rmesh1, gr1 = get_gr(s_data, s_data, L*rs, bins, alpha=alpha)
                rmesh2, gr2 = get_gr(s_data, x_data, L*rs, bins, alpha=alpha)
                rmesh3, gr3 = get_gr(x_data[:, :n//2], x_data[:, n//2:], L*rs, bins, alpha=alpha)
                rmesh4, gr4 = get_gr(x_data[:, :n//2], x_data[:, :n//2], L*rs, bins, alpha=alpha)
                rmesh4, gr5 = get_gr(x_data[:, n//2:], x_data[:, n//2:], L*rs, bins, alpha=alpha)
                line1, = axes[0,0].plot(rmesh1, gr1, label="pp, "+label, color=color)
                line2, = axes[0,1].plot(rmesh2, gr2, label="pe, "+label, color=color)
                line3, = axes[1,0].plot(rmesh3, gr3, label=r"ee opposite, "+label, color=color)
                line4, = axes[1,1].plot(rmesh4, (gr4+gr5)/2, label=r"ee parallel, "+label, color=color)
                axes[0,0].legend(loc=legend_loc[0], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                axes[0,1].legend(loc=legend_loc[1], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                axes[1,0].legend(loc=legend_loc[2], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                axes[1,1].legend(loc=legend_loc[3], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                plt.savefig(savefigname, dpi=dpi)
                print(f"{GREEN}save figure to:{RESET}", savefigname)

            if auto_save:
                print(f"{YELLOW}========= Saving samples ========{RESET}")
                ckpt = {"keys": keys,
                        "s": s_data / rs,
                        "x": x_data / rs,
                        "state_idx": state_idx_data,
                        "mo_coeff": mo_coeff_data,
                        "bands": bands_data,
                        "params_flow": jax.tree_map(lambda x: x[0], params_flow),
                        "params_van": jax.tree_map(lambda x: x[0], params_van),
                        "params_wfn": jax.tree_map(lambda x: x[0], params_wfn),
                        "K": K_data,
                        "Vpp": Vpp_data,
                        "Vep": Vep_data,
                        "Vee": Vee_data,
                        "P": P_data,
                        "E": E_data,
                        "F": F_data,
                        "Sp": Sp_data,
                        "Se": Se_data,
                    }
                if "_sample" in f:
                    index = f.index("_sample")
                else:
                    index = f.index(".pkl")
                sample_ckpt_filename = f[:index] + "_sample_sx_bs_%d" % s_data.shape[0] + ".pkl"
                if not os.path.isfile(sample_ckpt_filename):
                    save_data(ckpt, sample_ckpt_filename)
                    print(f"{GREEN}save sample data to:{RESET}", sample_ckpt_filename)
                else:
                    print(f"{GREEN}sample data exists:{RESET}", sample_ckpt_filename)
        else:
            if files_draw_new_sample[files.index(f)]:
                sample_steps = math.ceil(sample_total_batch_thisfile / sample_batch)
            else:
                sample_steps = math.ceil((sample_total_batch_thisfile - load_batch) / sample_batch)
            print(f"{MAGENTA}sample batch:{RESET}", sample_batch)
            print(f"{MAGENTA}sample steps:{RESET}", sample_steps)
            step_times = np.zeros(sample_steps)

            if sample_batch <= load_batch:
                s = jnp.reshape(s[:sample_batch], (num_devices, batch_per_device, n, dim))
                x = jnp.reshape(x[:sample_batch], (num_devices, batch_per_device, n, dim))
                state_idx = jnp.reshape(state_idx[:sample_batch], (num_devices, batch_per_device, n))
                mo_coeff = jnp.reshape(mo_coeff[:sample_batch], (num_devices, batch_per_device, num_states, num_states))
                bands = jnp.reshape(bands[:sample_batch], (num_devices, batch_per_device, num_states))
            else:
                s = jnp.tile(s, (sample_batch // load_batch + 1, 1, 1))[:sample_batch].reshape(num_devices, batch_per_device, n, dim)
                x = jnp.tile(x, (sample_batch // load_batch + 1, 1, 1))[:sample_batch].reshape(num_devices, batch_per_device, n, dim)
                state_idx = jnp.tile(state_idx, (sample_batch // load_batch + 1, 1))[:sample_batch]
                mo_coeff = jnp.tile(mo_coeff, (sample_batch // load_batch + 1, 1, 1, 1))[:sample_batch]
                bands = jnp.tile(bands, (sample_batch // load_batch + 1, 1))[:sample_batch]
            
            if savefigname is not None:
                print(f"{GREEN}save figure to:{RESET}", savefigname)
                
                if not files_draw_new_sample[files.index(f)]:
                    rmesh1, gr1 = get_gr(s_data, s_data, L*rs, bins, alpha=alpha)
                    rmesh2, gr2 = get_gr(s_data, x_data, L*rs, bins, alpha=alpha)
                    rmesh3, gr3 = get_gr(x_data[:, :n//2], x_data[:, n//2:], L*rs, bins, alpha=alpha)
                    rmesh4, gr4 = get_gr(x_data[:, :n//2], x_data[:, :n//2], L*rs, bins, alpha=alpha)
                    rmesh4, gr5 = get_gr(x_data[:, n//2:], x_data[:, n//2:], L*rs, bins, alpha=alpha)
                    line1, = axes[0,0].plot(rmesh1, gr1, label="pp, "+label, color=color)
                    line2, = axes[0,1].plot(rmesh2, gr2, label="pe, "+label, color=color)
                    line3, = axes[1,0].plot(rmesh3, gr3, label=r"ee opposite, "+label, color=color)
                    line4, = axes[1,1].plot(rmesh4, (gr4+gr5)/2, label=r"ee parallel, "+label, color=color)
                    axes[0,0].legend(loc=legend_loc[0], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    axes[0,1].legend(loc=legend_loc[1], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    axes[1,0].legend(loc=legend_loc[2], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    axes[1,1].legend(loc=legend_loc[3], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    plt.savefig(savefigname, dpi=dpi)

            s = shard(s)
            x = shard(x)

            if sample_batch > load_batch or files_draw_new_sample[files.index(f)]:
                print(f"{YELLOW}========= Thermlization ========{RESET}")
                for step in range(sample_therm):
                    keys, state_idx, mo_coeff, bands, s, x, ar_s, ar_x = sample_s_and_x(keys,
                                    sampler, params_van, 
                                    logprob_p, force_fn_p, s, params_flow,
                                    logpsi2, force_fn_e, x, params_wfn,
                                    mc_proton_steps, mc_electron_steps, 
                                    mc_proton_width, mc_electron_width, L, lcao_vmc, kpt)
                    print(f"{MAGENTA}therm step:{RESET}", step+1, f"{MAGENTA}ar_s:{RESET} %.4f" % ar_s[0], f"{MAGENTA}ar_x:{RESET} %.4f" % ar_x[0])

            print(f"{YELLOW}========= Sampling ========{RESET}")
            for step in range(sample_steps):
                time_start = time.time()
                keys, state_idx, mo_coeff, bands, s, x, ar_s, ar_x = sample_s_and_x(keys,
                                sampler, params_van, 
                                logprob_p, force_fn_p, s, params_flow,
                                logpsi2, force_fn_e, x, params_wfn,
                                mc_proton_steps, mc_electron_steps, 
                                mc_proton_width, mc_electron_width, L, lcao_vmc, kpt)
                K, Vpp, Vep, Vee, P, E, F, Sp, Se = observable_fn(params_flow, params_van, params_wfn, state_idx, mo_coeff, bands, s, x, keys)

                if step == 0 and files_draw_new_sample[files.index(f)]:
                    s_data = s.reshape(-1, n, dim)*rs
                    x_data = x.reshape(-1, n, dim)*rs
                    state_idx_data = state_idx.reshape(-1, n)
                    mo_coeff_data = mo_coeff.reshape(-1, num_states, num_states)
                    bands_data = bands.reshape(-1, num_states)

                    K_data = K.reshape(-1)/n/rs**2
                    Vpp_data = Vpp.reshape(-1)/n/rs**2
                    Vep_data = Vep.reshape(-1)/n/rs**2
                    Vee_data = Vee.reshape(-1)/n/rs**2
                    P_data = P.reshape(-1)/rs**2
                    E_data = E.reshape(-1)/n/rs**2
                    F_data = F.reshape(-1)/n/rs**2
                    Sp_data = Sp.reshape(-1)/n
                    Se_data = Se.reshape(-1)/n

                else:
                    s_data = jnp.concatenate((s_data, s.reshape(-1, n, dim)*rs), axis=0)
                    x_data = jnp.concatenate((x_data, x.reshape(-1, n, dim)*rs), axis=0)
                    state_idx_data = jnp.concatenate((state_idx_data, state_idx.reshape(-1, n)), axis=0)
                    mo_coeff_data = jnp.concatenate((mo_coeff_data, mo_coeff.reshape(-1, num_states, num_states)), axis=0)
                    bands_data = jnp.concatenate((bands_data, bands.reshape(-1, num_states)), axis=0)

                    K_data = jnp.concatenate((K_data, K.reshape(-1)/n/rs**2), axis=0)
                    Vpp_data = jnp.concatenate((Vpp_data, Vpp.reshape(-1)/n/rs**2), axis=0)
                    Vep_data = jnp.concatenate((Vep_data, Vep.reshape(-1)/n/rs**2), axis=0)
                    Vee_data = jnp.concatenate((Vee_data, Vee.reshape(-1)/n/rs**2), axis=0)
                    P_data = jnp.concatenate((P_data, P.reshape(-1)/rs**2), axis=0)
                    E_data = jnp.concatenate((E_data, E.reshape(-1)/n/rs**2), axis=0)
                    F_data = jnp.concatenate((F_data, F.reshape(-1)/n/rs**2), axis=0)
                    Sp_data = jnp.concatenate((Sp_data, Sp.reshape(-1)/n), axis=0)
                    Se_data = jnp.concatenate((Se_data, Se.reshape(-1)/n), axis=0)

                K_mean, K_std = K.mean()/n/rs**2, K.std()/jnp.sqrt(sample_batch)/n/rs**2
                Vpp_mean, Vpp_std = Vpp.mean()/n/rs**2, Vpp.std()/jnp.sqrt(sample_batch)/n/rs**2
                Vep_mean, Vep_std = Vep.mean()/n/rs**2, Vep.std()/jnp.sqrt(sample_batch)/n/rs**2
                Vee_mean, Vee_std = Vee.mean()/n/rs**2, Vee.std()/jnp.sqrt(sample_batch)/n/rs**2
                P_mean, P_std = P.mean()/rs**2, P.std()/jnp.sqrt(sample_batch)/rs**2
                E_mean, E_std = E.mean()/n/rs**2, E.std()/jnp.sqrt(sample_batch)/n/rs**2
                F_mean, F_std = F.mean()/n/rs**2, F.std()/jnp.sqrt(sample_batch)/n/rs**2
                Sp_mean, Sp_std = Sp.mean()/n, Sp.std()/jnp.sqrt(sample_batch)/n
                Se_mean, Se_std = Se.mean()/n, Se.std()/jnp.sqrt(sample_batch)/n

                print("-------------------------")
                Kp = 1.5*T/157887.66 # proton kinetic energy in Ry
                mp = 3697.9 # 1836
                ds = -jax.scipy.special.gammaln(n+1)/n + 1.5*(1+np.log(mp*rs**2/(4*np.pi)*T/157887.66))
                print(f"{MAGENTA}sample step:{RESET} %d/%d" % (step+1, sample_steps), 
                      f"{MAGENTA} F:{RESET} %.4f ± %.4f" % (F_mean + Kp - ds*T/157887.66, F_std), 
                      f"{MAGENTA} E:{RESET} %.4f ± %.4f" % (E_mean + Vpp_mean + Kp, jnp.sqrt(E_std**2 + Vpp_std**2)), 
                      f"{MAGENTA} P:{RESET} %.4f ± %.4f" % (P_mean + (2*Kp)/(3*(L*rs)**3)*14710.513242194795*n, P_std), 
                      f"{MAGENTA} Se:{RESET} %.4f ± %.4f" % (Se_mean, Se_std), 
                      f"{MAGENTA} ar_s:{RESET} %.4f" % ar_s[0], 
                      f"{MAGENTA} ar_x:{RESET} %.4f" % ar_x[0])
                
                F_data_mean, F_data_std = F_data.mean(), F_data.std()/jnp.sqrt(F_data.shape[0])
                E_data_mean, E_data_std = E_data.mean(), E_data.std()/jnp.sqrt(E_data.shape[0])
                P_data_mean, P_data_std = P_data.mean(), P_data.std()/jnp.sqrt(P_data.shape[0])
                Se_data_mean, Se_data_std = Se_data.mean(), Se_data.std()/jnp.sqrt(Se_data.shape[0])
                Vpp_data_mean, Vpp_data_std = Vpp_data.mean(), Vpp_data.std()/jnp.sqrt(Vpp_data.shape[0])

                print(f"{BLUE}accumulate samples:{RESET}", F_data.shape[0], 
                      f"{BLUE} F:{RESET} %.4f ± %.4f" % (F_data_mean + Kp - ds*T/157887.66, F_data_std), 
                      f"{BLUE} E:{RESET} %.4f ± %.4f" % (E_data_mean + Vpp_data_mean + Kp, jnp.sqrt(E_data_std**2 + Vpp_data_std**2)), 
                      f"{BLUE} P:{RESET} %.4f ± %.4f" % (P_data_mean + (2*Kp)/(3*(L*rs)**3)*14710.513242194795*n, P_data_std), 
                      f"{BLUE} Se:{RESET} %.4f ± %.4f" % (Se_data_mean, Se_data_std))

                time_end = time.time()
                step_times[step] = time_end - time_start
                if step == 0:
                    average_time = step_times[0]
                else:
                    average_time = step_times[1:step+1].mean()
                est_remaining_time = average_time*(sample_steps-step-1)
                print(f"{CYAN}run time:{RESET} %.2f " % step_times.sum(), 
                      f"{CYAN}step time:{RESET} %.2f " % step_times[step], 
                      f"{CYAN}average time:{RESET} %.2f " % average_time, 
                      f"{CYAN}remain step:{RESET} %d " % (sample_steps-step-1),
                      f"{CYAN}remain time:{RESET} %.2f " % est_remaining_time, 
                      f"{CYAN}finish time:{RESET}", time.strftime("%H:%M:%S", time.localtime(time.time() + est_remaining_time)))

                if savefigname is not None:
                    try:
                        line1
                    except NameError:
                        pass
                    else:
                        line1.remove()
                        line2.remove()
                        line3.remove()
                        line4.remove()
                    rmesh1, gr1 = get_gr(s_data, s_data, L*rs, bins, alpha=alpha)
                    rmesh2, gr2 = get_gr(s_data, x_data, L*rs, bins, alpha=alpha)
                    rmesh3, gr3 = get_gr(x_data[:, :n//2], x_data[:, n//2:], L*rs, bins, alpha=alpha)
                    rmesh4, gr4 = get_gr(x_data[:, :n//2], x_data[:, :n//2], L*rs, bins, alpha=alpha)
                    rmesh4, gr5 = get_gr(x_data[:, n//2:], x_data[:, n//2:], L*rs, bins, alpha=alpha)
                    line1, = axes[0,0].plot(rmesh1, gr1, label="pp, "+label, color=color)
                    line2, = axes[0,1].plot(rmesh2, gr2, label="pe, "+label, color=color)
                    line3, = axes[1,0].plot(rmesh3, gr3, label=r"ee opposite, "+label, color=color)
                    line4, = axes[1,1].plot(rmesh4, (gr4+gr5)/2, label=r"ee parallel, "+label, color=color)
                    axes[0,0].legend(loc=legend_loc[0], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    axes[0,1].legend(loc=legend_loc[1], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    axes[1,0].legend(loc=legend_loc[2], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    axes[1,1].legend(loc=legend_loc[3], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    plt.savefig(savefigname, dpi=dpi)

            s_data = s_data[:sample_total_batch_thisfile]
            x_data = x_data[:sample_total_batch_thisfile]
            state_idx_data = state_idx_data[:sample_total_batch_thisfile]
            mo_coeff_data = mo_coeff_data[:sample_total_batch_thisfile]
            bands_data = bands_data[:sample_total_batch_thisfile]
            K_data = K_data[:sample_total_batch_thisfile]
            Vpp_data = Vpp_data[:sample_total_batch_thisfile]
            Vep_data = Vep_data[:sample_total_batch_thisfile]
            Vee_data = Vee_data[:sample_total_batch_thisfile]
            P_data = P_data[:sample_total_batch_thisfile]
            E_data = E_data[:sample_total_batch_thisfile]
            F_data = F_data[:sample_total_batch_thisfile]
            Sp_data = Sp_data[:sample_total_batch_thisfile]
            Se_data = Se_data[:sample_total_batch_thisfile]

            if savefigname is not None:
                try:
                    line1
                except NameError:
                    pass
                else:
                    line1.remove()
                    line2.remove()
                    line3.remove()
                    line4.remove()
                rmesh1, gr1 = get_gr(s_data, s_data, L*rs, bins, alpha=alpha)
                rmesh2, gr2 = get_gr(s_data, x_data, L*rs, bins, alpha=alpha)
                rmesh3, gr3 = get_gr(x_data[:, :n//2], x_data[:, n//2:], L*rs, bins, alpha=alpha)
                rmesh4, gr4 = get_gr(x_data[:, :n//2], x_data[:, :n//2], L*rs, bins, alpha=alpha)
                rmesh4, gr5 = get_gr(x_data[:, n//2:], x_data[:, n//2:], L*rs, bins, alpha=alpha)
                line1, = axes[0,0].plot(rmesh1, gr1, label="pp, "+label, color=color)
                line2, = axes[0,1].plot(rmesh2, gr2, label="pe, "+label, color=color)
                line3, = axes[1,0].plot(rmesh3, gr3, label=r"ee opposite, "+label, color=color)
                line4, = axes[1,1].plot(rmesh4, (gr4+gr5)/2, label=r"ee parallel, "+label, color=color)
                axes[0,0].legend(loc=legend_loc[0], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                axes[0,1].legend(loc=legend_loc[1], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                axes[1,0].legend(loc=legend_loc[2], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                axes[1,1].legend(loc=legend_loc[3], facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                plt.savefig(savefigname, dpi=dpi)
                print(f"{GREEN}save figure to:{RESET}", savefigname)

            if auto_save:
                print(f"{YELLOW}========= Saving samples ========{RESET}")
                ckpt = {"keys": keys,
                        "s": s_data / rs,
                        "x": x_data / rs,
                        "state_idx": state_idx_data,
                        "mo_coeff": mo_coeff_data,
                        "bands": bands_data,
                        "params_flow": jax.tree_map(lambda x: x[0], params_flow),
                        "params_van": jax.tree_map(lambda x: x[0], params_van),
                        "params_wfn": jax.tree_map(lambda x: x[0], params_wfn),
                        "K": K_data,
                        "Vpp": Vpp_data,
                        "Vep": Vep_data,
                        "Vee": Vee_data,
                        "P": P_data,
                        "E": E_data,
                        "F": F_data,
                        "Sp": Sp_data,
                        "Se": Se_data,
                    }
                if "_sample" in f:
                    index = f.index("_sample")
                else:
                    index = f.index(".pkl")
                sample_ckpt_filename = f[:index] + "_sample_sx_bs_%d" % s_data.shape[0] + ".pkl"
                save_data(ckpt, sample_ckpt_filename)
                print(f"{GREEN}save sample data to:{RESET}", sample_ckpt_filename)

        try:
            line1
        except NameError:
            pass
        else:
            del line1, line2, line3, line4
