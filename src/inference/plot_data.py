from src.inference.utils import *
from src.inference.parser import parse_filename
from src.inference.markers import *
from src.inference.colors import default_colors
from src.inference.quantity import quantity
from src.inference.label import auto_label, add_newline_if_long

def extract_average_epoch_number_per_checkpoint(directory_path):
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist")
    
    # List to store the numbers
    epoch_numbers = []
    
    # Pattern to match epoch_#####.pkl format
    pattern = r'epoch_(\d{6})\.pkl'
    
    # Get all files in directory
    for filename in os.listdir(directory_path):
        # Check if file matches our pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the number and convert to int
            number = int(match.group(1))
            epoch_numbers.append(number)

    # Convert to numpy array, remove duplicates, and sort
    unique_sorted_numbers = np.sort(np.unique(epoch_numbers))

    if len(unique_sorted_numbers) < 2:
        raise ValueError("Need at least 2 unique points for linear regression")
    
    # Create x values (indices) and y values (epoch numbers)
    x = np.arange(len(unique_sorted_numbers))
    y = unique_sorted_numbers
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Convert to numpy array and return
    return slope

def plot_data(files: list, 
              quantities: list, 
              labels: list = None,
              label_dict: dict = None,
              unit='Ry', 
              mode='per atom', 
              savename=None, 
              running_average=1, 
              figsize=(10,6), 
              dpi=300, 
              lower_percentile=1, 
              upper_percentile = 99, 
              shift=0.1,
              grid=True, 
              log=True,
              dark_mode=False,
              max_columns=4,
              ref=True,
              legend_ncol=4,
              layout_rect=[0, 0, 1, 1],
              title=True,
              isotope='H',
              xlim=None,
              ylim=None,
              x_axis_time=False,
            ):
    """
        Plot data from multiple files.
        quantities: list, list of quantities to plot
        'f', 'etot', 'p', 'k', 'vtot', 'vep', 'vee', 'vpp', 's', 'se', 'sp', 'acc_s', 'acc_x'
    """

    # dark mode
    if dark_mode:
        # facecolor = 'black'
        # facecolor = '#404040'
        facecolor = (0.2, 0.2, 0.2)
        fontcolor = 'white'
    else:
        facecolor = 'white'
        fontcolor = 'black'

    # rows and columns
    if len(quantities) > max_columns:
        cols = max_columns
        rows = len(quantities) // max_columns + 1
    else:
        cols = len(quantities)
        rows = 1

    fig, axes = plt.subplots(nrows=rows, 
                             ncols=cols, 
                             figsize=figsize, 
                             dpi=dpi, 
                             facecolor=facecolor)
    for i in range(len(quantities)):
        obs = quantities[i]
        if rows == 1:
            fig_i = i
        else:
            fig_i = (i//cols, i%cols)
        if len(quantities) == 1:
            ax = axes
        else:
            ax = axes[fig_i]

        # init plot limits
        lower = 10000
        upper = -10000

        for f, marker, color in zip(files, markers, default_colors):
            
            # get parameters
            directory = os.path.dirname(f)
            cfg_filename = os.path.join(directory, 'config.yaml')
            if os.path.isfile(cfg_filename):
                cfg = OmegaConf.load(cfg_filename)
                rs = cfg.rs
                n = cfg.num
                T = cfg.T
            else:
                params = parse_filename(f)
                n = params["n"]
                rs = params["rs"]
                T = params["T"]
                batchsize = params["batchsize"]
                acc_steps = params["acc_steps"]

            # generate label
            if quantities.index(obs) != 0:
                label=None
            else:
                if labels is not None:
                    label = labels[files.index(f)]
                else:
                    if os.path.isfile(cfg_filename):
                        if label_dict is None:
                            label_dict = {'num': None, 
                                          'rs': None, 
                                          'T': None,
                                          'batchsize': None,
                                          'acc_steps': None}
                        label = auto_label(cfg, label_dict)
                    else:
                        label = f"n_{n}_rs_{rs}_T_{T}_bs_{batchsize}_acc_{acc_steps}"
                label = add_newline_if_long(label)

            # load data
            df = pd.read_csv(f, delimiter=r"\s+")
            y, yerr = quantity(obs, df, n, rs, T, isotope=isotope)

            # unit conversion
            if unit == 'Ha':
                unit_conv = 0.5 # 1 Ry = 2 Ha
            elif unit == 'Ry':
                unit_conv = 1
            elif unit == "eV":
                unit_conv = 13.6056980659
            if obs in ['f', 'etot', 'k', 'vtot', 'vep', 'vee', 'vpp']:
                y *= unit_conv
                yerr *= unit_conv

            # all atom or per atom
            if obs in ['f', 'etot', 'k', 'vtot', 'vep', 'vee', 'vpp', 's', 'se', 'sp']:
                if mode == 'all atom':
                    y *= n
                    yerr *= n
                elif mode == 'per atom':
                    pass
                else:
                    print('mode must be \'all atom\' or \'per atom\'')
            
            # remove nan
            y = y[~np.isnan(y)]
            yerr = yerr[~np.isnan(yerr)]

            # running average
            y_average = np.convolve(y, np.ones(running_average), 'valid') / running_average
            yerr_average = np.convolve(yerr, np.ones(running_average), 'valid') / running_average
            epochmax=len(y_average)

            # x axis epoch or time
            x_epoch = df['epoch'].values[0:epochmax]
            if x_axis_time:
                epoch_slope = extract_average_epoch_number_per_checkpoint(directory)
                average_epoch_time_s = 600 / epoch_slope
                x = x_epoch * average_epoch_time_s
            else:
                x = x_epoch

            # plot
            ax.errorbar(x, y_average[0:epochmax], yerr=yerr_average[0:epochmax],
                        marker=marker, color=color,markerfacecolor='none', 
                        markeredgewidth=1, linewidth=0.8, ms=2, capsize=4, label=label, alpha=0.3)
            ax.scatter(x, y_average[0:epochmax], marker=marker, 
                        s=2, color=color, alpha=1.0)
            
            # update plot limits
            if np.percentile(y_average, lower_percentile)-shift < lower:
                lower = np.percentile(y_average, lower_percentile)-shift
            if np.percentile(y_average, upper_percentile)+shift > upper:
                upper = np.percentile(y_average, upper_percentile)+shift

        # plot reference automaticaly
        if ref:
            ref_dir = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/'
            for filename in os.listdir(ref_dir):
                ref_file = os.path.join(ref_dir, filename)
                ref_df = pd.read_csv(ref_file, delimiter=r"\s+")
                try:
                    ref_value = ref_df.query('(T==%g) & (rs==%g)'%(T, rs))[obs].values[0]
                    if obs=='etot':
                        ref_value = ref_value/13.6056980659
                        ax.axhline(y=ref_value, linewidth=1.5, color=fontcolor, label='ref '+ref_file.split("/")[-1].split(".")[0], zorder=2.5)
                        break
                    ax.axhline(y=ref_value, linewidth=1.5, color=fontcolor, zorder=2.5)
                    break
                except:
                    pass
        
        # update plot limits
        if 'ref_value' in locals():
            if ref_value-shift < lower:
                lower = ref_value-shift
            if ref_value+shift > upper:
                upper = ref_value+shift
            del ref_value

        if title:
            if isotope == "D" or isotope == "Deuterium":
                isotope_title = "Deuterium"
            elif isotope == "H" or isotope == "Hydrogen":
                isotope_title = "Hydrogen"
            
            if obs == 'f':
                ax.set_title(isotope_title + ' Free Energy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'etot':
                ax.set_title(isotope_title + ' Total Energy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'vtot':
                ax.set_title(isotope_title + ' Total Potential Energy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'vep':
                ax.set_title(isotope_title + ' Electron-Nucleus Potential Energy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'vee':
                ax.set_title(isotope_title + ' Electron-Electron Potential Energy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'vpp':
                ax.set_title(isotope_title + ' Nucleus-Nucleus Potential Energy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'k':
                ax.set_title(isotope_title + ' Electron Kinetic Energy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'p':
                ax.set_title(isotope_title + ' Pressure', color=fontcolor)
            elif obs == 's':
                ax.set_title(isotope_title + ' Total Entropy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'se':
                ax.set_title(isotope_title + ' Electron Entropy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'sp':
                ax.set_title(isotope_title + ' Nucleus Entropy '+'(%s)' % mode, color=fontcolor)
            elif obs == 'acc_s':
                ax.set_title('Acceptance Rate of s', color=fontcolor)
            elif obs == 'acc_x':
                ax.set_title('Acceptance Rate of x', color=fontcolor)
        
        # y axis label
        if isotope == "D" or isotope == "Deuterium":
            isotope_label = "D"
        elif isotope == "H" or isotope == "Hydrogen":
            isotope_label = "H"

        if mode == 'all atom':
            mode_unit = f'/{n}{isotope_label}'
        elif mode == 'per atom':
            mode_unit = f'/{isotope_label}'

        if obs=='f':
            ax.set_ylabel("F"+" ("+unit+mode_unit+")", color=fontcolor)
        elif obs=='etot':
            ax.set_ylabel("E"+" ("+unit+mode_unit+")", color=fontcolor)
        elif obs=='vtot':
            ax.set_ylabel("V"+" ("+unit+mode_unit+")", color=fontcolor)
        elif obs=='vep':
            ax.set_ylabel("Vep"+" ("+unit+mode_unit+")", color=fontcolor)
        elif obs=='vee':
            ax.set_ylabel("Vee"+" ("+unit+mode_unit+")", color=fontcolor)
        elif obs=='vpp':
            ax.set_ylabel("Vpp"+" ("+unit+mode_unit+")", color=fontcolor)
        elif obs=='k':
            ax.set_ylabel("Ke"+" ("+unit+mode_unit+")", color=fontcolor)
        elif obs == 'p':
            ax.set_ylabel("pressure (GPa)", color=fontcolor)
        elif obs == "s":
            ax.set_ylabel("S (kB"+mode_unit+")", color=fontcolor)
        elif obs == 'se':
            ax.set_ylabel("Se (kB"+mode_unit+")", color=fontcolor)
        elif obs == 'sp':
            ax.set_ylabel("Sp (kB"+mode_unit+")", color=fontcolor)
        elif obs == 'acc_s' or obs == 'acc_x':
            ax.set_ylabel(obs, color=fontcolor)
        
        if x_axis_time:
            ax.set_xlabel('time (s)', color=fontcolor)
        else:
            ax.set_xlabel('epochs', color=fontcolor)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(1)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([lower, upper])
        if grid:
            ax.grid(True)
        if log:
            ax.set_xscale('log')
        ax.set_facecolor(facecolor) 
        ax.spines['bottom'].set_color(fontcolor)
        ax.spines['top'].set_color(fontcolor)
        ax.spines['left'].set_color(fontcolor)
        ax.spines['right'].set_color(fontcolor)
        ax.tick_params(axis='both', colors=fontcolor)
    
    for i in range(len(quantities), rows*cols):
        fig_i = (i//cols, i%cols)
        ax = axes[fig_i]
        ax.axis('off')

    fig.legend(loc='upper center', ncol=legend_ncol, bbox_to_anchor=(0.5, 0.95), 
               facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
    plt.tight_layout(rect=layout_rect) 
    plt.show()

    if savename is not None:
        plt.savefig(savename+'.png', dpi=dpi)
        print(f"{GREEN}Figure saved as {savename}.png{RESET}")

if __name__ == '__main__':

    # 10000K rs=1.86
    quantities = ['f', 'etot', 'p', 's']
    # files = ['/data/lizh/hydrogen/finiteT/hf_van_cond_mala_jax/hf_van_cond_gth-dzv/n_14_dim_3_rs_1.86_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_3_fh1_64_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_400_mw_0.02_0.04_lr_0.01_0.01_0.01_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_256_ap_1/data.txt']
    # files += ['/data/lizh/hydrogen/finiteT/hf_van_cond_mcmc/hf_van_cond_gth-dzv/n_14_dim_3_rs_1.86_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_3_fh1_64_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_50_400_mw_0.02_0.04_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_512_ap_2/data.txt']
    # files += ['/data/lizh/hydrogen/finiteT/hf_van_cond_mcmc/hf_van_cond_gth-dzv/n_14_dim_3_rs_1.86_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_3_fh1_64_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_800_mw_0.02_0.07_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_1024_ap_8/data.txt']
    # files += ['/data/lizh/hydrogen/finiteT/dft_van_cond_mcmc/dft_lda,vwn_van_cond_gth-dzv/n_14_dim_3_rs_1.86_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_3_fh1_64_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_800_mw_0.02_0.07_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_256_ap_1/data.txt']
    # files += ['/data/lizh/hydrogen/finiteT/dft_van_cond_mcmc/dft_lda,vwn_van_cond_gth-dzv/n_14_dim_3_rs_1.86_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_3_fh1_32_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_800_mw_0.02_0.07_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_200_ap_1/data.txt']
    # labels = ['HF 256', 'HF 512*2', 'HF 1024*8', 'DFT 256', 'pre HF 200', 'pre HF T10000 rs2 800*4']
    # plot_quantity(files, quantities, running_average=1, figsize=(15,4), dpi=300, lower_percentile=1, upper_percentile = 99.9, labels=labels, savename="figures/quantity_10000_1.86")

    
    files = ["/data/lizh/hydrogen/finiteT/hf_van_cond_mcmc/hf_van_cond_gth-dzv/n_14_dim_3_rs_2_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_3_fh1_64_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_50_800_mw_0.02_0.04_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_256_ap_1/data.txt"]
    files += ["/data/lizh/hydrogen/finiteT/hf_van_cond_mcmc/hf_van_cond_gth-dzv/n_14_dim_3_rs_2_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_3_fh1_64_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_50_400_mw_0.02_0.04_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_512_ap_2/data.txt"]
    files += ['/data/lizh/hydrogen/finiteT/dft_van_cond_mcmc/final_hf_gth-dzv/n_14_dim_3_rs_2_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_6_fh1_16_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_800_mw_0.02_0.07_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_800_ap_4/data.txt']
    files += ['/data/lizh/hydrogen/finiteT/dft_van_cond_mcmc/final_hf_gth-dzv/n_14_dim_3_rs_1.86_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_6_fh1_16_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_800_mw_0.02_0.07_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_400_ap_8/data.txt']
    # files += ['/data/lizh/hydrogen/finiteT/dft_van_cond_mcmc/final_hf_gth-dzv/n_14_dim_3_rs_2_T_31250_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_6_fh1_16_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_800_mw_0.02_0.07_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_400_ap_8/data.txt']
    # files += ['/data/lizh/hydrogen/finiteT/dft_van_cond_mcmc/final_hf_gth-dzv/n_14_dim_3_rs_1.86_T_31250_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_6_fh1_16_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_800_mw_0.02_0.07_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_400_ap_8/data.txt']

    labels = ['HF 256', 'HF 512*2', 'pre HF 800*4', 'pre 10000 1.86', 'pre 31250 2', 'pre 31250 1.86' ]
    plot_data(files, quantities, running_average=1, figsize=(15,4), dpi=300, lower_percentile=1, upper_percentile = 99.9, labels=labels, savename="figures/quantity_10000_2")
