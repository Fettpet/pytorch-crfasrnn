# See http://graphics.stanford.edu/papers/permutohedral/ for details
# https://github.com/idofr/pymutohedral_lattice for understanding

import torch
import Permutohedral
import Permutohedral_gpu


class PermutohedralFiltering(torch.autograd.Function):

    """
    This function is used for the permutohedral filtering. This methode consists of four steps:
    1. Generating Position vektors (I dont know what to do )
    2. Splatting: Define the enclosing simplex and compute barycentric weights. (complex and I dont know what to do)
    3. Blurring: Gaussian Blur with [1 2 1] along each lattice direction.
    4. Inverse of Splatting
    """
    @staticmethod
    def forward(
            ctx,
            cur_state,
            input_image,
            bilateral,
            theta_alpha,
            theta_beta,
            theta_gamma
    ):
        assert len(cur_state.shape) == 4

        # here the values are stored. Since the autograd forward and backward have to be static we use ctx to store the
        # values.
        # found here https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845/3
        ctx.bilateral = bilateral
        ctx.theta_alpha = theta_alpha
        ctx.theta_beta = theta_beta
        ctx.theta_gamma = theta_gamma
        ctx.input_image = input_image

        if torch.cuda.is_available():
            test = Permutohedral_gpu.forward(
                cur_state,
                input_image,
                ctx.bilateral,
                ctx.theta_alpha,
                ctx.theta_beta,
                ctx.theta_gamma
            )
            return test
        else:
            return Permutohedral.forward(
                cur_state,
                ctx.input_image,
                ctx.bilateral,
                ctx.theta_alpha,
                ctx.theta_beta,
                ctx.theta_gamma
            )

    @staticmethod
    def backward(
            ctx,
            cur_state
    ):
        if torch.cuda.is_available():
            test = Permutohedral_gpu.backward(
                cur_state,
                ctx.input_image,
                ctx.bilateral,
                ctx.theta_alpha,
                ctx.theta_beta,
                ctx.theta_gamma
            )

            return test, torch.zeros_like(ctx.input_image), None, None, None, None
        else:
            return Permutohedral.backward(
                cur_state,
                ctx.input_image,
                ctx.bilateral,
                ctx.theta_alpha,
                ctx.theta_beta,
                ctx.theta_gamma
            ), torch.zeros_like(ctx.input_image), None, None, None, None


class PermutohedralLayer(torch.nn.Module):
    """
    Hasnt any parameter
    """

    def __init__(
            self,
            bilateral,
            theta_alpha,
            theta_beta,
            theta_gamma

    ):
        super(PermutohedralLayer, self).__init__()
        self.bilateral = bilateral
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma

    def forward(
            self,
            cur_state,
            input_image
    ):
        return PermutohedralFiltering.apply(
            cur_state,
            input_image,
            self.bilateral,
            self.theta_alpha,
            self.theta_beta,
            self.theta_gamma
        )

