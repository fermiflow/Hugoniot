import re
import time
import haiku as hk
from functools import partial

from src.vmc import sample_s
from src.flow import make_flow
from src.ad import make_grad_real
from src.ferminet import FermiNet
from src.inference.utils import *
from src.utils import shard, replicate
from src.checkpoint import load_data, save_data
from src.inference.parser import parse_filename
from src.inference.colors import default_colors, gen_gradient_colors
from src.inference.find_ckpt_sample import find_ckpt_sample

def get_gr(x, y, L, bins=100, alpha=1.0, rm_dist=1e-6):
    """
        Compute radial distribution function (RDF) between two particle sets x and y.
    Input:
        x: array, shape (batchsize, nx, dim), positions of particles in the first set
        y: array, shape (batchsize, ny, dim), positions of particles in the second set
        L: float, box size
        bins: int, number of bins for histogram
        alpha: float, exponent for non-linear binning
        rm_dist: float, minimum distance to consider (to avoid same particle zero distance)
    Output:
        rmesh: array, shape (bins,), midpoints of the bins
        gr: array, shape (bins,), radial distribution function values
    """

    # Validate input shapes
    batchsize, nx, dim = x.shape
    batchsize, ny, dim = y.shape

    # Calculate all pairwise differences with periodic boundary conditions
    rij = x.reshape(-1, nx, 1, dim) - y.reshape(-1, 1, ny, dim)  # Shape: (batchsize, nx, ny, dim)
    rij = rij - L * np.rint(rij / L)  # Apply periodic boundary correction

    # Compute pairwise distances
    dij = np.linalg.norm(rij, axis=-1)  # Shape: (batchsize, nx, ny)
    dij_flat = dij.reshape(-1)  # Flatten to 1D array
    dij_flat = dij_flat[dij_flat > rm_dist] # Remove zero distances

    # non-linear bin edges
    i = np.linspace(0, bins, bins + 1)
    bin_edges = (i / bins) ** alpha * (L / 2)

    # Generate distance histogram
    hist, bin_edges = np.histogram(dij_flat, bins=bin_edges)

    # Calculate expected particle count in ideal gas
    volume_bins = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)  # Spherical shell volumes
    density_pairs = (nx * ny * batchsize) / (L**3)  # Pair density
    h_id = volume_bins * density_pairs  # Expected count in ideal gas

    # Compute radial distribution function
    rmesh = (bin_edges[1:] + bin_edges[:-1]) / 2
    gr = hist / h_id  # Normalize actual counts by ideal gas expectation
    
    return rmesh, gr

def sample_flow_s(files, 
                  savefigname=None,
                  labels=None, 
                  auto_load=True,
                  auto_save=True,
                  sample_batch=1000,
                  sample_flow_batch=256,
                  sample_therm=10,
                  sample_mc_proton_steps = 200,
                  sample_mc_proton_width = 0.05,
                  bins=100, 
                  figsize=(10,4), 
                  dpi=300,
                  gradient_color=True,
                  gradient_color_start='white',
                  gradient_color_end='red',
                  legend_loc='lower right',
                  grid = True,
                  dark_mode=False):
    """
        Draw samples from flow model in files.
    Input:
        files: list of str, checkpoint files
        savefigname: str, save name of the figure
        labels: list of str, labels of the files
        auto_load: bool, whether to auto load the biggest sample file
        auto_save: bool, whether to auto save the sample file

        sample_batch: int, number of samples to plot
        # when load_batch < sample_batch, we need the following parameters for new samples
        sample_flow_batch: int, batchsize of new flow samples
        sample_therm: int, number of thermalization steps
        sample_mc_proton_steps: int, number of MCMC steps
        sample_mc_proton_width: float, width of MCMC step

        bins: int, number of plot bins
        figsize: tuple, figure size
        dpi: int, dpi of the figure

        gradient_color: bool, whether to use gradient color
        gradient_color_start: str, start color
        gradient_color_end: str, end color
        legend_loc: str, location of the legend
        grid: bool, whether to show grid
        dark_mode: bool, whether to use dark mode
    """

    if dark_mode:
        facecolor = (0.2, 0.2, 0.2)
        fontcolor = 'white'
    else:
        facecolor = 'white'
        fontcolor = 'black'

    if gradient_color:
        colors = gen_gradient_colors(gradient_color_start, gradient_color_end, len(files))
    else:
        colors = default_colors

    if labels is not None:
        assert len(labels) == len(files)


    if savefigname is not None:
        if not savefigname.endswith('.png') or not savefigname.endswith('.pdf'):
            savefigname += '.png'
        plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
        plt.gca().set_facecolor(facecolor)
        plt.gca().spines['bottom'].set_color(fontcolor)
        plt.gca().spines['top'].set_color(fontcolor)
        plt.gca().spines['right'].set_color(fontcolor)
        plt.gca().spines['left'].set_color(fontcolor)
        plt.tick_params(axis='both', colors=fontcolor)
        plt.xlabel("$r(Bohr)$", color=fontcolor)
        plt.ylabel("g(r)", color=fontcolor)
        plt.ylim(-0.13, 1.23)
        if grid:
            plt.grid(color='gray')
    
    for f in files:     

        directory = os.path.dirname(f)
        cfg_filename = os.path.join(directory, 'config.yaml')
        if os.path.isfile(cfg_filename):
            cfg = OmegaConf.load(cfg_filename)
            rs = cfg.rs
            n = cfg.num
            dim = cfg.dim
            Nf = cfg.flow.Nf
            flow_steps = cfg.flow.steps
            flow_depth = cfg.flow.depth
            flow_h1size = cfg.flow.h1size
            flow_h2size = cfg.flow.h2size

        else:
            params = parse_filename(f)
            rs = params["rs"]
            n = params["n"]
            dim = params["dim"]
            Nf = params["Nf"]
            flow_steps = params["flow_steps"]
            flow_depth = params["flow_depth"]
            flow_h1size = params["flow_h1size"]
            flow_h2size = params["flow_h2size"]

        if labels is None:
            label = r'$rs=%g,n=%g$'%(rs, n)
        else:
            label = labels[files.index(f)]
        color = colors[files.index(f)]

        L = (4/3*jnp.pi*n)**(1/3)

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

        print(f"{YELLOW}========= Loading checkpoint ========{RESET}")
        # auto load the biggest checkpoint file
        if auto_load:
            auto_find_f, auto_find_f_batch = find_ckpt_sample(f, mode="s", silent_mode=True)
            if auto_find_f is not None:
                ckpt_file = auto_find_f
                print(f"{GREEN}auto load the biggest ckeckpoint sample file:\n{RESET}File:", ckpt_file)
            else:
                ckpt_file = f
                print(f"{GREEN}no ckeckpoint sample file found, load checkpoint file:\n{RESET}File:", ckpt_file)
        else:
            ckpt_file = f
            print(f"{GREEN}load checkpoint file:\n{RESET}File:", ckpt_file)
        data = load_data(ckpt_file)

        keys, s, params_flow = data["keys"], data["s"], data["params_flow"]
        s = jnp.reshape(s, (-1, n, dim))
        s_data = s * rs
    
        num_devices = jax.device_count()
        batch_per_device = sample_flow_batch // num_devices
        
        keys = jax.random.split(keys[0], num_devices)
        
        load_batch = s.shape[0]
        print(f"{MAGENTA}load s batch:{RESET}", load_batch)
        if savefigname is not None:
            print(f"{MAGENTA}plot s batch:{RESET}", sample_batch)
        if load_batch >= sample_batch:
            s_data = s_data[:sample_batch]
            
            # plotting
            if savefigname is not None:
                rmesh, gr = get_gr(s_data, s_data, L*rs, bins)
                line, = plt.plot(rmesh, gr, label=label, color=color) # label+' bs %d' % s_data.shape[0]
                plt.legend(loc=legend_loc, facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                plt.savefig(savefigname, dpi=dpi)
                print(f"{GREEN}save figure to:{RESET}", savefigname)
            
            if auto_save:
                print(f"{YELLOW}========= Saving samples ========{RESET}")
                ckpt = {"keys": keys,
                        "s": s_data / rs,
                        "params_flow": params_flow,
                    }
                if "_sample" in f:
                    index = f.index("_sample")
                else:
                    index = f.index(".pkl")
                sample_ckpt_filename = f[:index] + "_sample_s_bs_%d" % s_data.shape[0] + ".pkl"
                if not os.path.isfile(sample_ckpt_filename):
                    save_data(ckpt, sample_ckpt_filename)
                    print(f"{GREEN}save sample data to:{RESET}", sample_ckpt_filename)
                else:
                    print(f"{GREEN}sample data exists:{RESET}", sample_ckpt_filename)
                
        else:
            sample_steps = (sample_batch - load_batch) // sample_flow_batch + 1
            print(f"{MAGENTA}sample batch:{RESET}", sample_flow_batch)
            print(f"{MAGENTA}sample steps:{RESET}", sample_steps)
            step_times = np.zeros(sample_steps)

            if sample_flow_batch <= load_batch:
                s = jnp.reshape(s[:sample_flow_batch], (num_devices, batch_per_device, n, dim))
            else:
                s = jnp.tile(s, (sample_flow_batch // load_batch + 1, 1, 1))[:sample_flow_batch].reshape(num_devices, batch_per_device, n, dim)
            
            # plotting
            if savefigname is not None:
                rmesh, gr = get_gr(s_data, s_data, L*rs, bins)
                line, = plt.plot(rmesh, gr, label=label, color=color) # label+' bs %d' % s_data.shape[0]
                plt.legend(loc=legend_loc, facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                plt.savefig(savefigname, dpi=dpi)

            s, keys = shard(s), shard(keys)
            params_flow = replicate((params_flow), num_devices)

            if sample_flow_batch > load_batch:
                print(f"{YELLOW}========= Thermlization ========{RESET}")
                for step in range(sample_therm):
                    keys, s, ar_s = sample_s(keys, logprob_p, force_fn_p, s, params_flow, sample_mc_proton_steps, sample_mc_proton_width, L)
                    print(f"{MAGENTA}therm step:{RESET}", step, f"{MAGENTA}ar_s:{RESET} %.4f" % ar_s[0])
                s_data = jnp.concatenate((s_data, s.reshape(-1, n, dim)*rs), axis=0)
                print(f"{MAGENTA}s_data.shape:{RESET}", s_data.shape)

                # plotting
                if savefigname is not None:
                    line.remove()
                    rmesh, gr = get_gr(s_data, s_data, L*rs, bins)
                    line, = plt.plot(rmesh, gr, label=label, color=color) # label+' bs %d' % s_data.shape[0]
                    plt.legend(loc=legend_loc, facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    plt.savefig(savefigname, dpi=dpi)

            print(f"{YELLOW}========= Sampling ========{RESET}")
            for step in range(sample_steps):              
                time_start = time.time()
                keys, s, ar_s = sample_s(keys, logprob_p, force_fn_p, s, params_flow, sample_mc_proton_steps, sample_mc_proton_width, L)
                s_data = jnp.concatenate((s_data, s.reshape(-1, n, dim)*rs), axis=0)
                
                print("-------------------------")
                print(f"{MAGENTA}sample step:{RESET}", step, f"{MAGENTA}ar_s:{RESET} %.4f" % ar_s[0], f"{MAGENTA}batch:{RESET}", s_data.shape[0])
                
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


                # plotting
                if savefigname is not None:
                    line.remove()
                    rmesh, gr = get_gr(s_data, s_data, L*rs, bins)
                    line, = plt.plot(rmesh, gr, label=label, color=color) # label+' bs %d' % s_data.shape[0]
                    plt.legend(loc=legend_loc, facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                    plt.savefig(savefigname, dpi=dpi)

            s_data = s_data[:sample_batch]

            # plotting
            if savefigname is not None:
                line.remove()
                rmesh, gr = get_gr(s_data, s_data, L*rs, bins)
                line, = plt.plot(rmesh, gr, label=label, color=color) # label+' bs %d' % s_data.shape[0]
                plt.legend(loc=legend_loc, facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
                plt.savefig(savefigname, dpi=dpi)
                print(f"{GREEN}save figure to:{RESET}", savefigname)

            if auto_save:
                print(f"{YELLOW}========= Saving samples ========{RESET}")
                ckpt = {"keys": keys,
                        "s": s_data / rs,
                        "params_flow": jax.tree_map(lambda x: x[0], params_flow),
                    }
                if "_sample" in f:
                    index = f.index("_sample")
                else:
                    index = f.index(".pkl")
                sample_ckpt_filename = f[:index] + "_sample_s_bs_%d" % s_data.shape[0] + ".pkl"
                save_data(ckpt, sample_ckpt_filename)
                print(f"{GREEN}save sample data to:{RESET}", sample_ckpt_filename)
            
if __name__ == "__main__":
    files = []
    files += ['/data/lizh/hydrogen/finiteT/final_hf_gth-dzv_cgqr_n20/n_20_rs_2_T_10000_hf_bs_320_ap_8_2025-01-11_19:17:57/epoch_000002.pkl']
    files += ['/data/lizh/hydrogen/finiteT/final_hf_gth-dzv_cgqr_n20/n_20_rs_2_T_10000_hf_bs_320_ap_8_2025-01-11_19:17:57/epoch_000893.pkl']

    labels = [
                # 'pre epoch 8381', 
                'train epoch 2', 
                'train epoch 893', 
                ]

    sample_flow_s(files, 
                savefigname="figures/rdf_pp_n20_T10000_rs2", 
                labels=labels, 
                sample_batch=50000, 
                sample_flow_batch=2048, 
                sample_therm=10, 
                gradient_color=True,
                gradient_color_start='blue',
                gradient_color_end='red',
                bins=100, 
                figsize=(9,4), 
                dpi=300, 
                dark_mode=False)