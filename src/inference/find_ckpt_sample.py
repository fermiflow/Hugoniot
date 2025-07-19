import os
import re

from src.inference.utils import *

def find_max_bs_file(directory, pattern):
    """
        find the file with the maximum s batchsize in the directory
    """

    max_number = -1
    max_file_name = None

    for filename in os.listdir(directory):
        match = re.search(pattern, filename)
        if match:
            number = int(match.group(1))

            if number > max_number:
                max_number = number
                max_file_name = filename

    return max_file_name, max_number

def find_ckpt_sample(ckpt_filename, 
                     mode: str = "s",
                     silent_mode: bool = False,
                     ):
    """
    Checkpoint sample file names and contents:
        sample_s_bs_10000: {keys, s, params_flow}
        sample_s_bs_10000_quantity: {keys, s, params_flow, n, rs, T, f, f_err, etot, etot_err, k, k_err, vpp, vpp_err, vep, vep_err, vee, vee_err, p, p_err, ep_cov, sp, sp_err, se, se_err}
        sample_sx_bs_10000: {keys, s, state_idx, x, params_flow, params_van, params_wfn, n, rs, T}
        sample_sx_bs_10000_quantity: {keys, s, state_idx, x, params_flow, params_van, params_wfn, n, rs, T, f, f_err, etot, etot_err, k, k_err, vpp, vpp_err, vep, vep_err, vee, vee_err, p, p_err, ep_cov, sp, sp_err, se, se_err}
    """
    directory = os.path.dirname(ckpt_filename)
    file_basename = os.path.basename(ckpt_filename)
    if "_sample" in file_basename:
        index = file_basename.index("_sample")
        file_basename = file_basename[:index]
    else:
        index = file_basename.index(".pkl")
        file_basename = file_basename[:index]

    if mode == "s" or mode == "s_quantity":
        if mode == "s":
            pattern = file_basename[:index] + r'_sample_s_bs_(\d+).*'
        else:
            pattern = file_basename[:index] + r'_sample_s_bs_(\d+)_.*_quantity.*'
        auto_find_f, auto_find_f_batch = find_max_bs_file(directory, pattern)
        if auto_find_f is not None:
            ckpt_sample_filename = os.path.join(directory, auto_find_f)
            if not silent_mode:
                print(f"{MAGENTA}Auto find sample file:\n{RESET} File:", ckpt_sample_filename)
        else:
            ckpt_sample_filename = None
            auto_find_f_batch = -1
            if not silent_mode:
                print(f"{MAGENTA}No sample file found{RESET}")

    elif mode == "sx" or mode == "sx_quantity":
        if mode == "sx":
            pattern = file_basename[:index] + r'_sample_sx_bs_(\d+).*'
        elif mode == "sx_quantity":
            pattern = file_basename[:index] + r'_sample_sx_bs_(\d+).*_quantity.*'
        auto_find_f, auto_find_f_batch = find_max_bs_file(directory, pattern)
        if auto_find_f is not None:
            ckpt_sample_filename = os.path.join(directory, auto_find_f)
            if not silent_mode:
                print(f"{MAGENTA}Auto find sample file:\n{RESET} File:", ckpt_sample_filename)
        else:
            ckpt_sample_filename = None
            auto_find_f_batch = -1
            if not silent_mode:
                print(f"{MAGENTA}No sample file found{RESET}")
    else:
        raise ValueError("mode not recognized")

    return ckpt_sample_filename, auto_find_f_batch

if __name__ == "__main__":
    ckpt_filename = "/data/lizh/hydrogen/finiteT/final_hf_gth-dzv/n_14_dim_3_rs_1.86_T_10000_Em_10_l_2_m_16_he_4_hi_32_fs_1_fd_6_fh1_16_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_100_800_mw_0.02_0.07_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_400_ap_8/epoch_000002_sample_s_bs_100000.pkl"
    mode = "sx"
    find_ckpt_sample(ckpt_filename, mode)