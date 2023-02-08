import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional
import torch.nn as nn
import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import torch.nn as nn
from torch.optim import functional as F

class Tco(nn.Module):
    def __init__(self,net,test_loader,drop_p = 0.8,s_l = 0.7,gamma = 0.1,steps = 22,lr = 1e-4,weight_decay = 1e-4,momentum=0.9,solver_name="doubleSGD"):
        super(Tco,self).__init__()
        print("Already use Tco method")
        self.drop_p = drop_p
        self.s_l = s_l
        self.criterion = nn.CrossEntropyLoss()
        if solver_name=="doubleAdam":
            self.solver_other = doubleAdam(net.parameters(), lr=lr,weight_decay=weight_decay)
        elif solver_name=="doubleSGD":
            self.solver_other = doubleSGD(net.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        else:
            print("Undefined optimizer: ",solver_name,", doubleSGD is used by default.")
            self.solver_other = doubleSGD(net.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.schedule_other = torch.optim.lr_scheduler.StepLR(self.solver_other, step_size=steps, gamma=gamma)
        self.test_loader = test_loader
        self.softmax_ = nn.Softmax(dim=1)
        
    def select(self,imgs,net):
        net.train(False)
        ret = []
        er = 0
        with torch.no_grad():
             for i in imgs:
               i = i.reshape([1,i.shape[0],i.shape[1],i.shape[2]])
               output = net(i)
               output = self.softmax_(output)
               _, pred = torch.max(output, 1)
               if _[0] > self.s_l:
                   er += 1
                   if er == 1:
                       ret = i
                   elif er >= 2:
                       ret = torch.cat((ret,i),0)
        if er >= 2:
            er = True
        else:
            er = False
        return ret,er

    def forward(self,Wcynet):
        for imgs, _none in self.test_loader:
           self.solver_other.zero_grad()
           imgs = imgs.cuda()
           imgs,er = self.select(imgs,Wcynet)
           if not er:
               continue
           Wcynet.train(True)
           output = Wcynet(imgs)
           _, pred = torch.max(output, 1)
           loss = self.criterion(output, pred)
           loss.backward()
           self.solver_other.step(self.drop_p)
        self.schedule_other.step()
        return Wcynet
    

class doubleAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(doubleAdam, self).__init__(params, defaults)
        print("Use doubleAdam Optimizer")

    def __setstate__(self, state):
        super(doubleAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self,p_, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
            
            l_drop = nn.Dropout(p=p_)
            for i in range(len(grads)):
                e = l_drop(grads[i])
                e = e*(1-p_)
                grads[i] = e
            beta1, beta2 = group['betas']
            F.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )
        return loss
        
class doubleSGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach)
        print("Use doubleSGD Optimizer")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(doubleSGD, self).__init__(params, defaults)
        
        
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, p_,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            
            l_drop = nn.Dropout(p=p_)
            for i in range(len(d_p_list)):
                e = l_drop(d_p_list[i])
                e = e*(1-p_)
                d_p_list[i] = e
            
            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd
    
    
    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)


def _multi_tensor_sgd(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any([grad.is_sparse for grad in grads])

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    alpha = lr if maximize else -lr
    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=alpha)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=alpha)
    
'''
data_she = shelve.open("save/baocun")
r = []
w = []

Wcynet.train(False)
softmax_ = nn.Softmax(dim=1)
num_total = 0
num_acc = 0
r_top = 0
r_bot = 1
w_top = 0
w_bot = 1
for imgs, labels in test_loader:
    imgs = imgs.cuda()
    labels = labels.cuda()
    output = Wcynet(imgs)
    output = softmax_(output)
    _, pred = torch.max(output, 1)
    e1 = torch.sum(pred == labels.detach_())
    e2 = labels.size(0)
    num_acc += e1
    num_total += e2
    if pred[0] == labels[0]:
        #print("正确：",_[0])
        r_top = max(r_top,_[0])
        r_bot = min(r_bot,_[0])
        r.append(float(_[0]))
    else:
        #print("错误：",_[0])
        w_top = max(w_top,_[0])
        w_bot = min(w_bot,_[0])
        w.append(float(_[0]))
    #print(output)
    #print(_)
    #print(pred)
    #print(labels)
test_acc_epoch = num_acc.detach().cpu().numpy() * 100 / num_total
print("测试集精度是：",test_acc_epoch,"%")
print(r_top,r_bot,w_top,w_bot)
data_she["r"] = r
data_she["w"] = w
data_she.close()
'''
