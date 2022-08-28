from cmath import isnan
import warnings

from dowel import logger
import numpy as np
import torch
from torch.optim import Optimizer

from garage.np import unflatten_tensors


def _build_hessian_vector_product(func, params, reg_coeff=1e-5):
    """Computes Hessian-vector product using Pearlmutter's algorithm.
    `Pearlmutter, Barak A. "Fast exact multiplication by the Hessian." Neural
    computation 6.1 (1994): 147-160.`
    Args:
        func (callable): A function that returns a torch.Tensor. Hessian of
            the return value will be computed.
        params (list[torch.Tensor]): A list of function parameters.
        reg_coeff (float): A small value so that A -> A + reg*I.
    Returns:
        function: It can be called to get the final result.
    """
    param_shapes = [p.shape or torch.Size([1]) for p in params]
    f = func()
    f_grads = torch.autograd.grad(f, params, create_graph=True)

    def _eval(vector):
        """The evaluation function.
        Args:
            vector (torch.Tensor): The vector to be multiplied with
                Hessian.
        Returns:
            torch.Tensor: The product of Hessian of function f and v.
        """
        unflatten_vector = unflatten_tensors(vector, param_shapes)

        assert len(f_grads) == len(unflatten_vector)
        grad_vector_product = torch.sum(
            torch.stack(
                [torch.sum(g * x) for g, x in zip(f_grads, unflatten_vector)]))

        hvp = list(
            torch.autograd.grad(grad_vector_product, params,
                                retain_graph=True))
        for i, (hx, p) in enumerate(zip(hvp, params)):
            if hx is None:
                hvp[i] = torch.zeros_like(p)

        flat_output = torch.cat([h.reshape(-1) for h in hvp])
        return flat_output + reg_coeff * vector

    return _eval


def _conjugate_gradient(f_Ax, b, cg_iters, residual_tol=1e-10):
    """Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.

    Args:
        f_Ax (callable): A function to compute Hessian vector product.
        b (torch.Tensor): Right hand side of the equation to solve.
        cg_iters (int): Number of iterations to run conjugate gradient
            algorithm.
        residual_tol (float): Tolerence for convergence.

    Returns:
        torch.Tensor: Solution x* for equation Ax = b.

    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x



class DRSOMOptimizer(Optimizer):
    def __init__(self,
                 params,
                 max_constraint_value,
                 cg_iters=10,
                 hvp_reg_coeff=1e-5):
        super().__init__(params, {})
        self._cg_iters = cg_iters
        self._hvp_reg_coeff = hvp_reg_coeff
        self._max_constraint_value = max_constraint_value
        self._max_backtracks = 15
        self._backtrack_ratio = 0.8
        self._accept_violation = False

    def compute_alpha(self, m, f_loss, f_constraint, itr):
        params = []
        grads = []
        # params_values = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad.reshape(-1))
                    # params_values.append(p.data.reshape(-1))
        g = torch.cat(grads)

        # flat_params_values = torch.cat(params_values)
        # print('flat params value is:')
        # print(flat_params_values)
        # print('-------------------------------------')

        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             if p.grad.grad_fn is not None:
        #                 p.grad.detach_()
        #             else:
        #                 p.grad.requires_grad_(False)
        #             p.grad.zero_()
        
        f_Ax = _build_hessian_vector_product(f_constraint, params,
                                             self._hvp_reg_coeff)

        if itr <= -1:
            # with torch.no_grad():
            #     tmp = torch.zeros_like(m)
            #     if torch.equal(tmp, m):
            #         per = 1e-8
            #         m = m + per * torch.ones_like(m)

            if itr > 0:
                Fg = f_Ax(g)
                Fm = f_Ax(m)

                print('g is: ')
                print(g)
                print('-------------------------------')

                print('g norm is:')
                print(torch.norm(g))
                print('-------------------------------')

                print('m is: ')
                print(m)
                print('-------------------------------')

                gg = torch.dot(g, g)
                gm = torch.dot(g, m)
                gFg = torch.dot(g, Fg)
                gFm = torch.dot(g, Fm)
                FFg = f_Ax(Fg)
                FFm = f_Ax(Fm)

                # for four directions
                # c = torch.tensor([gg, gm, gFg, gFm], requires_grad=False)

                # for two directions
                c = torch.tensor([gg, gm], requires_grad=False)

                print('c is:')
                print(c)
                print('------------------------------')

                g_ = torch.unsqueeze(g, dim=0)
                m_ = torch.unsqueeze(m, dim=0)
                Fg_ = torch.unsqueeze(Fg, dim=0)
                Fm_ = torch.unsqueeze(Fm, dim=0)
                FFg_ = torch.unsqueeze(FFg, dim=0)
                FFm_ = torch.unsqueeze(FFm, dim=0)

                # for four directions
                # left = torch.cat((g_, m_, Fg_, Fm_), axis=0)
                # right = torch.cat((Fg_.t(), Fm_.t(), FFg_.t(), FFm_.t()), axis=1)

                # for two directions
                left = torch.cat((g_, m_), axis=0)
                right = torch.cat((Fg_.t(), Fm_.t()), axis=1)

                G = torch.matmul(left, right).detach()
                print('G is:')
                print(G)
                print('-----------------------------')

                eigen, _ = torch.eig(G)
                var = eigen.min()
                if var < 0:
                    print('Find an indefinite G')
                    G = G - (var - 1e-8) * torch.eye(2)
                print('eigen of G is:')
                print(eigen)
                print('-----------------------------')

                inverse = torch.pinverse(G)

                print('inverse of G is:')
                print(inverse)
                print('----------------------------')

                x = inverse @ c
                print("x is:")
                print(x)
                print('----------------------------')

                print('xTGx is: ')
                print(torch.dot(x, G @ x))
                print('----------------------------')

                alpha = np.sqrt(2. * self._max_constraint_value * (1. / (torch.dot(x, G @ x) + 1e-8))) * x

                print(1/2 * torch.dot(alpha, G@alpha))

                if torch.isnan(alpha).sum():
                    print('find nan step size!')
                    alpha = torch.ones(4)

                print('alpha is:')
                print(alpha)
                print('--------------------------')
                
                # return alpha, g, Fg, Fm, params

                steps = alpha[0] * g + alpha[1] * m
                param_shapes = [p.shape or torch.Size([1]) for p in params]
                steps = unflatten_tensors(steps, param_shapes)
                assert len(steps) == len(params)
                prev_params = [p.clone() for p in params]
                
                loss_before = f_loss()
                print('loss before mean is')
                print(loss_before)
                print('---------------------------')

                for step, prev_param, param in zip(steps, prev_params, params):
                    new_param = prev_param.data - step
                    param.data = new_param.data

                loss = f_loss()
                print('after loss is:')
                print(loss)
                print('--------------------------')

                constraint_val = f_constraint()
                print('constraint value is:')
                print(constraint_val)
                print('--------------------------')
            
            else:
                Fg = f_Ax(g)
                alpha = np.sqrt(2.0 * self._max_constraint_value *
                            (1. /
                             (torch.dot(g, Fg) + 1e-8)))
                steps = alpha * g
                param_shapes = [p.shape or torch.Size([1]) for p in params]
                steps = unflatten_tensors(steps, param_shapes)
                assert len(steps) == len(params)
                prev_params = [p.clone() for p in params]
                
                loss_before = f_loss()
                print('loss before mean is')
                print(loss_before)
                print('---------------------------')

                for step, prev_param, param in zip(steps, prev_params, params):
                    new_param = prev_param.data - step
                    param.data = new_param.data

                loss = f_loss()
                print('after loss is:')
                print(loss)
                print('--------------------------')

                constraint_val = f_constraint()
                print('constraint value is:')
                print(constraint_val)
                print('--------------------------')

        else:
            
            step_dir = _conjugate_gradient(f_Ax, g, self._cg_iters)
            step_dir[step_dir.ne(step_dir)] = 0.
            step_size = np.sqrt(2.0 * self._max_constraint_value *
                            (1. /
                             (torch.dot(step_dir, f_Ax(step_dir)) + 1e-8)))
            if np.isnan(step_size):
                print("find a nan stepsize!")
                step_size = 1.
            descent_step = step_size * step_dir

            # backtracking linesearch

            prev_params = [p.clone() for p in params]
            ratio_list = self._backtrack_ratio**np.arange(self._max_backtracks)
            loss_before = f_loss()

            param_shapes = [p.shape or torch.Size([1]) for p in params]
            descent_step = unflatten_tensors(descent_step, param_shapes)
            assert len(descent_step) == len(params)

            for ratio in ratio_list:
                for step, prev_param, param in zip(descent_step, prev_params,
                                                params):
                    step = ratio * step
                    new_param = prev_param.data - step
                    param.data = new_param.data

                loss = f_loss()
                constraint_val = f_constraint()
                if (loss < loss_before
                        and constraint_val <= self._max_constraint_value):
                    # print('loss before is:')
                    # print(loss_before)
                    # print('loss now is:')
                    # print(loss)
                    # print('----------------------------')
                    break

            if ((torch.isnan(loss) or torch.isnan(constraint_val)
                or loss >= loss_before
                or constraint_val >= self._max_constraint_value)
                    and not self._accept_violation):
                logger.log('Line search condition violated. Rejecting the step!')
                if torch.isnan(loss):
                    logger.log('Violated because loss is NaN')
                if torch.isnan(constraint_val):
                    logger.log('Violated because constraint is NaN')
                if loss >= loss_before:
                    logger.log('Violated because loss not improving')
                if constraint_val >= self._max_constraint_value:
                    logger.log('Violated because constraint is violated')
                for prev, cur in zip(prev_params, params):
                    cur.data = prev.data




        

    # for backup
    # mFm = torch.dot(m, Fm)
    # mFg = torch.dot(m, Fg)
    # G = torch.tensor([[gFg, mFg + per], [mFg + per, mFm + per]], requires_grad=False)
    # print("G is:")
    # print(G)
    # coff = 1. / (G[0][0] * G[1][1] - G[0][1] * G[1][0])
    # inverse = coff * torch.tensor([[G[1][1], (-1) * G[0][1]], [(-1) * G[1][0], G[0][0]]], requires_grad=False)