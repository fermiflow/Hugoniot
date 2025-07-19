import jax
import jax.numpy as jnp

def make_grad_complex(f, argnums=0):
    """
        Given a COMPLEX-VALUED scalar function `f`, return the gradient w.r.t.
    the positional argument(s) specified by `argnums`.
    """
    def f_tuple(*args):
        res = f(*args)
        return jnp.stack([res.real, res.imag])

    def f_grad(*args):
        grad = jax.jacrev(f_tuple, argnums)(*args)
        return jax.tree_map(lambda jac: jac[0] + 1j * jac[1], grad)

    return f_grad

def make_laplacian_complex(f, argnums=0, method="forloop", **kwargs):
    """
        Given a COMPLEX-VALUED scalar function `f`, return the laplacian w.r.t.
    a positional argument specified by the integer `argnums`.
    """
    if not isinstance(argnums, int):
        raise ValueError("argnums should be an integer.")

    def f_tuple(*args):
        res = f(*args)
        return jnp.stack([res.real, res.imag])

    def f_laplacian(*args):
        x = args[argnums]
        shape, size = x.shape, x.size
        grad_f = jax.jacrev(lambda x: f_tuple( *[x.reshape(shape) if i == argnums else arg
                                                 for (i, arg) in enumerate(args)] ))
        x_flatten = x.reshape(-1)
        eye = jnp.eye(size, dtype=x.dtype)

        print(f"Computing laplacian ({method}, {kwargs}) ...")

        if method == "forloop":
            def body_fun(i, val):
                _, tangent = jax.jvp(grad_f, (x_flatten,), (eye[i],))
                return val + tangent[0, i] + 1j * tangent[1, i]
            laplacian = jax.lax.fori_loop(0, size, body_fun, 0.+0.j)

        elif method == "vmap":
            def body_fun(x, basevec):
                _, tangent = jax.jvp(grad_f, (x,), (basevec,))
                return jnp.dot(tangent, basevec)
            laplacian = jax.vmap(body_fun, (None, 1), 1)(x_flatten, eye).sum(axis=-1)
            laplacian = laplacian[0] + 1j * laplacian[1]

        elif method == "hessian":
            flatten_f = lambda x: f_tuple( *[x.reshape(shape) if i == argnums else arg
                                                for (i, arg) in enumerate(args)] )
            hessian = jax.hessian(flatten_f, holomorphic=False)(x_flatten)
            laplacian = jnp.trace(hessian[0] + 1j * hessian[1])

        else:
            raise ValueError("Unknown method to compute the laplacian: %s" % method)

        print("Done.")

        return laplacian
    
    return f_laplacian

def make_laplacian_complex_hutchinson(f, argnums=0, vmap_fn=lambda _: _):
    if not isinstance(argnums, int):
        raise ValueError("argnums should be an integer.")

    def f_tuple(*args):
        res = f(*args)
        return jnp.stack([res.real, res.imag])

    def f_laplacian(key, *args):
        x = args[argnums]
        v = jax.random.normal(key, x.shape)

        @vmap_fn
        def f_random_laplacian(v, *args):
            x = args[argnums]
            jvp_fn = lambda x: jax.jvp(
                lambda x: f_tuple( *[x if i == argnums else arg for (i, arg) in enumerate(args)] ),
                (x,), (v,)
            )[1]
            _, laplacian = jax.jvp(jvp_fn, (x,), (v,))
            laplacian = laplacian[0] + 1j * laplacian[1]
            print("Computed laplacian using hutchinson's trick.")
            return laplacian
        
        return f_random_laplacian(v, *args)

    return f_laplacian

def make_vjp_complex(f, argnums=0):
    """
        Given a COMPLEX-VALUED function `f`, return the vector-Jacobian product
    w.r.t. the positional argument(s) specified by `argnums`.
    """
    return lambda vec: make_grad_complex(lambda *args: (vec * f(*args)).sum(), argnums)

make_grad_real = jax.grad

def make_laplacian_real(f, argnums=0, method="forloop"):
    """
        Given a REAL-VALUED scalar function `f`, return the laplacian w.r.t.
    a positional argument specified by the integer `argnums`.
    """
    if not isinstance(argnums, int):
        raise ValueError("argnums should be an integer.")

    def f_laplacian(*args):
        x = args[argnums]
        shape, size = x.shape, x.size
        grad_f = jax.grad(lambda x: f( *[x.reshape(shape) if i == argnums else arg
                                                for (i, arg) in enumerate(args)] ))
        x_flatten = x.reshape(-1)
        eye = jnp.eye(size)

        if method == "forloop":
            print("forloop version...")
            def body_fun(i, val):
                _, tangent = jax.jvp(grad_f, (x_flatten,), (eye[i],))
                return val + tangent[i]
            laplacian = jax.lax.fori_loop(0, size, body_fun, 0.)
        elif method == "vmap":
            print("vmap version...")
            def body_fun(x, basevec):
                _, tangent = jax.jvp(grad_f, (x,), (basevec,))
                return (tangent * basevec).sum()
            laplacian = jax.vmap(body_fun, (None, 1), 1)(x_flatten, eye).sum()
        else:
            raise ValueError("Unknown method to compute the laplacian: %s" % method)
        print("Computed laplacian.")

        return laplacian
    
    return f_laplacian

def make_vjp_real(f, argnums=0):
    """
        Given a REAL-VALUED function `f`, return the vector-Jacobian product
    w.r.t. the positional argument(s) specified by `argnums`.
    """
    return lambda vec: make_grad_real(lambda *args: (vec * f(*args)).sum(), argnums)
