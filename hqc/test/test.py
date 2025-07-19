import jax
import time
import hydra
import numpy as np
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
jax.config.update("jax_enable_x64", True)

from hqc.pbc.lcao import make_lcao
from test_pbc_lcao import pyscf_hf, pyscf_dft

@hydra.main(version_base=None, config_path="conf", config_name="speed")
def main(cfg : DictConfig) -> None:
    
    print("\n========== test config ==========")
    print(OmegaConf.to_yaml(cfg))

    print("\n========== envinfo ==========")
    jax.print_environment_info()
    devices = jax.devices()
    num_devices = jax.device_count()
    print("GPU devices:")
    for i, device in enumerate(devices):
        print("---- ", i, " ", device.device_kind)
    print("Cluster connected with totally %d GPUs" % jax.device_count())
    print("This is process %d with %d local GPUs." % (jax.process_index(), jax.local_device_count()))

    n = cfg.num
    rs = cfg.rs
    L = (4/3*jnp.pi*n)**(1/3)
    batchsize = cfg.batchsize
    basis = cfg.basis
    diis = cfg.diis
    jit = cfg.jit
    dft = (cfg.method == "dft")

    print("\n========== config ==========")
    print("test mode:", cfg.mode)
    if cfg.mode == "memory":
        print("batchsize:", batchsize)
    if cfg.mode == "speed":
        print("batchsize:", batchsize)
        print("total batch:", cfg.totalbatch)
    print("n:", n)
    print("L:", L)
    print("rs:", rs)
    print("L*rs:", L*rs)

    print("\n========== scfinfo ==========")
    print("scf method:", cfg.method)
    print("basis:", basis)
    print("gridlength:", cfg.gridlength)
    print("rcut:", cfg.rcut)
    if cfg.method == "dft":
        print("xc:", cfg.xc)
    print("diis:", diis)
    print("tol:", cfg.tol)
    print("max cycle:", cfg.maxcycle)
    print("smearing temperature:", cfg.smearing_temperature)
    print("smearing:", cfg.smearing)
    if cfg.smearing:
        Ry = 2 # Ha/Ry
        beta = 157888.088922572/cfg.smearing_temperature # inverse temperature in unit of 1/Ry
        smearing_sigma = 1/beta/Ry # temperature in Hartree unit
        print("smearing_method:", cfg.smearing_method)
        print("smearing_sigma:", smearing_sigma)
    print("jit:", cfg.jit)

    print("\n************* begin test *************")

    print("making solver...")
    time1 = time.time()
    lcao_novmap = make_lcao(n, L, rs, basis=basis, rcut=cfg.rcut, tol=cfg.tol, max_cycle=cfg.maxcycle,
                        grid_length=cfg.gridlength, diis=diis, diis_damp=cfg.damp, use_jit=jit, dft=dft, xc=cfg.xc,
                        smearing=cfg.smearing, smearing_method=cfg.smearing_method, smearing_sigma=smearing_sigma,
                        search_method=cfg.search_method, search_cycle=cfg.search_cycle, search_tol=cfg.search_tol)
    time2 = time.time()
    print("make solver time:", time2-time1)

    print("initializing xp...")
    if cfg.mode == "random_scf" or cfg.mode == "memory" or cfg.mode == "search_random":
        key = jax.random.PRNGKey(43)
        xp = jax.random.uniform(key, (batchsize, n, 3), minval=0., maxval=L)
    elif cfg.mode == "speed":
        key = jax.random.PRNGKey(43)
        xp = jax.random.uniform(key, (cfg.totalbatch//batchsize, batchsize, n, 3), minval=0., maxval=L)
    elif cfg.mode == "checkpoint_scf" or cfg.mode == "search_checkpoint":
        checkpoint_filename = "data/checkpoint/epoch_004140.pkl"
        data = load_data(checkpoint_filename)
        xp = data["s"]
        assert xp.shape[2] == n
        assert xp.shape[3] == 3
        xp = xp.reshape(-1, n, 3)

    print("testing \'"+ cfg.mode + "\'...")
    if cfg.mode == "random_scf" or cfg.mode == "checkpoint_scf":

        # run twice to compare the time
        time1 = time.time()
        mo_coeff, bands, E = lcao_novmap(xp[0])
        time2 = time.time()

        time3 = time.time()
        mo_coeff, bands, E = lcao_novmap(xp[cfg.batchid])
        time4 = time.time()

        print("\n========== solver ==========")
        print("solver bands:\n", bands)
        print("solver E:", E)
        print("compile time:", time2-time1-time4+time3)
        print("run time:", time4-time3)
        print("finished!")

        if cfg.pyscf:
            print("\n========== pyscf ==========")
            if dft:
                time1 = time.time()
                mo_coeff_dft, energy_dft, E_dft = pyscf_dft(n, L, rs, smearing_sigma, xp[cfg.batchid], basis, xc=cfg.xc, 
                                                            smearing=cfg.smearing, smearing_method=cfg.smearing_method)
                time2 = time.time()
                print("pyscf time:", time2-time1)
                print("pyscf bands:\n", energy_dft)
                print("pyscf E:", E_dft)
            else:
                time1 = time.time()
                mo_coeff_hf, energy_hf, E_hf = pyscf_hf(n, L, rs, smearing_sigma, xp[cfg.batchid], basis, 
                                                        smearing=cfg.smearing, smearing_method=cfg.smearing_method)
                time2 = time.time()
                print("pyscf time:", time2-time1)
                print("pyscf bands:\n", energy_hf)
                print("pyscf E:", E_hf)
            print("finished!")

    elif cfg.mode == "search_checkpoint" or cfg.mode == "search_random":

        # run twice to compare the time
        time1 = time.time()
        mo_coeff, bands, E = lcao_novmap(xp[0])
        time2 = time.time()

        time3 = time.time()
        mo_coeff, bands, E = lcao_novmap(xp[cfg.batchid])
        time4 = time.time()

        print("compile time:", time2-time1-time4+time3)

        print("begin searching...")
        for i in range(1, len(xp)):
            time1 = time.time()
            mo_coeff, bands, E = lcao_novmap(xp[i])
            time2 = time.time()
            print("batchid:", i, "time:", time2-time1)

    elif cfg.mode == "memory":

        print("vmap lcao...")
        lcao = jax.vmap(lcao_novmap, 0, (0, 0, 0))

        # run twice to compare the time
        time1 = time.time()
        mo_coeff, bands, E = lcao(xp)
        time2 = time.time()

        # run twice to compare the time
        time3 = time.time()
        mo_coeff, bands, E = lcao(xp)
        time4 = time.time()

        print("\n========== solver ==========")
        # print("solver bands[0]:\n", bands[0])
        # print("solver E:", E)
        print("compile time:", time2-time1-time4+time3)
        print("run time:", time4-time3)
        print("finished!")
    
    elif cfg.mode == "speed":

        print("vmap lcao...")
        lcao = jax.vmap(lcao_novmap, 0, (0, 0, 0))

        # run twice to compare the time
        time1 = time.time()
        mo_coeff, bands, E = lcao(xp[0])
        time2 = time.time()

        time1 = time.time()
        print("total loop:" , xp.shape[0])
        for i in range(xp.shape[0]):
            time3 = time.time()
            mo_coeff, bands, E = lcao(xp[i])
            time4 = time.time()
            print("loop:", i, "time:", time4-time3)
        time2 = time.time()

        print("\n========== solver ==========")
        # print("solver bands[0]:\n", bands[0])
        # print("solver E:", E)
        print("total time:", time2-time1)
        print("finished!")
        
if __name__ == "__main__":
    main()
