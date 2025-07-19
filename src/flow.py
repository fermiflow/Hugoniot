import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def make_flow(network, n, dim, L):
    
    #base_logp = make_base(n, dim, L, indices)
    @partial(jax.jit, static_argnums=2)
    def logprob(params, s, scan=False):
        flow_flatten = lambda x: network.apply(params, None, x.reshape(n, dim)).reshape(-1)
        
        s_flatten = s.reshape(-1)
        if scan:
            _, jvp = jax.linearize(flow_flatten, s_flatten)
            def _body_fun(carry, x):
                return carry, jvp(x)
            _, jac = jax.lax.scan(_body_fun, None, jnp.eye(n*dim)) # this is actually jacobian transposed 
        else:
            jac = jax.jacfwd(flow_flatten)(s_flatten)
        
        _, logdetjac = jnp.linalg.slogdet(jac)
        
        return logdetjac - (n*dim*np.log(L) + jax.scipy.special.gammaln(n+1)) #uniform base

        #z = network.apply(params, None, s)
        #return logdetjac + base_logp(z)
       
    return logprob

