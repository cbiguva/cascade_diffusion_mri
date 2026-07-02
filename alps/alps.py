import torch
import torch.nn as nn


class Denoiser(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, sigma, class_labels):
        y = self.net(inputs, sigma, class_labels)
        return y

    def jvp(self, outputs, inputs, conditioning, vector, precondition):
        if precondition:
            vector = vector / conditioning
            grad_JT_eps = torch.autograd.grad(
                outputs=outputs, inputs=inputs, grad_outputs=vector,
                create_graph=True, only_inputs=True
            )[0]
            grad_JT_eps = grad_JT_eps * conditioning  # undo preconditioning
        else:
            grad_JT_eps = torch.autograd.grad(
                outputs=outputs, inputs=inputs, grad_outputs=vector,
                create_graph=True, only_inputs=True
            )[0]
        return grad_JT_eps

def giveScore(x, net, sigma, precondition, class_labels):
    x = x.clone().detach().requires_grad_(True)
    denoised = net(x, sigma, class_labels)
    eps = x - denoised
    E = 0.5 * torch.sum((eps).abs()**2, dim=(1, 2, 3))
    base = getattr(net, "module", net) 
    JVP = base.jvp(outputs=denoised, inputs=x, conditioning=sigma, vector=eps, precondition=precondition)   
    score = eps - JVP
    del x
    return E, score
##############################################################################
def ALPS_old_stepsize(opts,A, net, y,tstCsm,tstMask, isALPS=True, storeIntermediate=False):
    

    device = y.device
    t_steps = giveTsteps(opts.sigma_max,opts.sigma_min,opts.rho,opts.num_steps,device)
    
    
   
    Atb = A.adjoint(y,tstCsm)
    inference_std = opts.inference_std
    rhs=((1/(inference_std**2))*Atb)
    xtilde = A.init(Atb, rhs, inference_std, t_steps[0], tstCsm, tstMask)
    n = A.(torch.randn_like(xtilde),tstMask,t_steps[0],inference_std)
    x = xtilde + n.detach() 
    

    if storeIntermediate:
        xshape = list(x.shape)
        xshape[0] = len(t_steps+1)
        denoised = torch.zeros(xshape).cpu( )
        xdc = torch.zeros(xshape).cpu( )
        xsample = torch.zeros(xshape).cpu( )
        tsample = torch.zeros(len(t_steps+1)).cpu()

    for i in range(len(t_steps)):
        for k in range(opts.K):
            t = t_steps[i]
            # denoising
            if isALPS and (k<opts.K-1): # not doing it for the last iteration at scale t_i
                # Adding noise to generate posterior samples in p_{t_i}  
                _, score = giveScore(torch.cat((x.real,x.imag),1), net, t.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
                score= score.detach()
                score = score[:,0,:,:] +1j* score[:,1,:,:]
                score = torch.unsqueeze(score,0)  
                d = (x-score).detach()
                # CG update
                xtilde = cg_quadratic(x,d,t,inference_std,tstCsm,tstMask,y,A)
                n = A.NoiseModulation(torch.randn_like(x), tstMask,t,inference_std)
                x = x + ((opts.step_size**2)/2 *(xtilde-x)) + opts.step_size*n.detach()
            else:
                if i < len(t_steps) - 1:
                    t_next = t_steps[i + 1]
                    _, score = giveScore(torch.cat((x.real,x.imag),1), net, t_next.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
                    score= score.detach()
                    score = score[:,0,:,:] +1j* score[:,1,:,:]
                    score = torch.unsqueeze(score,0) 
                    d = (x-score).detach()
                    # CG update
                    x = cg_quadratic(x,d,t_next,inference_std,tstCsm,tstMask,y,A)
                else:
                    _, score = giveScore(torch.cat((x.real,x.imag),1), net, t.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
                    score= score.detach()
                    score = score[:,0,:,:] +1j* score[:,1,:,:]
                    score = torch.unsqueeze(score,0)
                    d = (x-score).detach()
                    xtilde = cg_quadratic(x,d,t,inference_std,tstCsm,tstMask,y,A)  
                    x = xtilde
                

        # Adding noise to generate posterior samples in p_{t_{i+1}} 
        # No noise was added at the last iteration at scale t_i; now moving to t_{i+1}

        if(i< len(t_steps)-1):
            n = A.NoiseModulation(torch.randn_like(x),tstMask,t_steps[i+1],inference_std)
            #x = x + ((opts.step_size**2)/2 *(xtilde-x)) + opts.step_size*n.detach()
            x = x+ opts.step_size*n.detach()

 
        
        if storeIntermediate:
            denoised[i] = d.abs().detach().cpu()
            xdc[i]= xtilde.abs().detach().cpu()
            xsample[i] = x.abs().detach().cpu()
            tsample[i] = t.detach().cpu()
            
       

          
   
    if storeIntermediate: 
        return x, xsample,tsample,denoised,xdc 
    else:
        return x
###############################
def ALPS_old_stepsize_MALA(A, net, y, tstCsm,tstMask,opts, isALPS=True, storeIntermediate=False):
    
    device = y.device
    t_steps = giveTsteps(opts.sigma_max,opts.sigma_min,opts.rho,opts.num_steps,device)
    
    Atb = A.adjoint(y,tstCsm)
    inference_std = opts.inference_std
    rhs=((1/(inference_std**2))*Atb)
    xtilde = A.init(Atb, rhs, inference_std, t_steps[0], tstCsm, tstMask)
    n = A.NoiseModulation(torch.randn_like(xtilde),tstMask,t_steps[0],inference_std)
    x = xtilde + n.detach() 
    
    
        
    acc_attempt_i = torch.zeros(len(t_steps), device=device, dtype=torch.long)
    acc_accept_i  = torch.zeros(len(t_steps), device=device, dtype=torch.long)
    rate_i = torch.zeros(len(t_steps), device=device, dtype=torch.float32)
    if storeIntermediate:
        xshape = list(x.shape)
        xshape[0] = len(t_steps+1)
        denoised = torch.zeros(xshape).cpu( )
        xdc = torch.zeros(xshape).cpu( )
        xsample = torch.zeros(xshape).cpu( )
        tsample = torch.zeros(len(t_steps+1)).cpu()
        posterior_energy=torch.zeros(len(t_steps+1)).cpu()

    for i in range(len(t_steps)):
        k=0 
        count = 0
        t = t_steps[i]
        # denoising
        _, score = giveScore(torch.cat((x.real,x.imag),1), net, t.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
        score= score.detach()
        score = score[:,0,:,:] +1j* score[:,1,:,:]
        score = torch.unsqueeze(score,0)  
        d = (x-score).detach()
        # CG update
        xtilde = cg_quadratic(x,d,t,opts.inference_std,tstCsm,tstMask,y,A)

        
        while k<=opts.K-2 and count<=2*opts.K: # not doing it for the last iteration at scale t_i


            n = A.NoiseModulation(torch.randn_like(x), tstMask,t,opts.inference_std)
            x_new = x + ((opts.step_size**2)/2 *(xtilde-x)) + opts.step_size*n.detach()                                       

            # ---- MH acceptance ----
            log_alpha =  -cost_old(x_new,y,tstCsm,tstMask,t,net,A,opts)+cost_old(x,y,tstCsm,tstMask,t,net,A,opts)-(0.5*(1/opts.step_size**2)*delta_old(x,x_new,score,t,y,tstCsm,tstMask,A,net,opts))
            log_alpha = opts.beta*(log_alpha.detach())
            
            if torch.log(torch.rand((), device=device)) < log_alpha:
            
                #print(cost_old(x_new,y,t,net,A)- cost_old(x,y,t,net,A))
                x = x_new
                # denoising
                _, score = giveScore(torch.cat((x.real,x.imag),1), net, t.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
                score= score.detach()
                score = score[:,0,:,:] +1j* score[:,1,:,:]
                score = torch.unsqueeze(score,0)
                d = (x-score).detach()
                # CG update
                xtilde = cg_quadratic(x,d,t,opts.inference_std,tstCsm,tstMask,y,A)
                k = k+ 1
            count = count+1
        #print(i,count)
        acc_attempt_i[i] =count 
        #print(acc_attempt_i[i])
        rate_i[i]= (k / acc_attempt_i[i].float()).item()
                
        

 

        if i < len(t_steps) - 1:
            t_next = t_steps[i + 1]
            _, score = giveScore(torch.cat((x.real,x.imag),1), net, t_next.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
            score= score.detach()
            score = score[:,0,:,:] +1j* score[:,1,:,:]
            score = torch.unsqueeze(score,0) 
            d = (x-score).detach()
            # CG update
            x = cg_quadratic(x,d,t_next,opts.inference_std,tstCsm,tstMask,y,A)
            n = A.NoiseModulation(torch.randn_like(x),tstMask,t_steps[i+1],opts.inference_std)
            x = x+ opts.step_size*n.detach()
        else:
            _, score = giveScore(torch.cat((x.real,x.imag),1), net, t.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
            score= score.detach()
            score = score[:,0,:,:] +1j* score[:,1,:,:]
            score = torch.unsqueeze(score,0)
            d = (x-score).detach()
            xtilde = cg_quadratic(x,d,t,opts.inference_std,tstCsm,tstMask,y,A)  
            x = xtilde
 
        
        if storeIntermediate:
            denoised[i] = d.abs().detach().cpu()
            xdc[i]= xtilde.abs().detach().cpu()
            posterior_energy[i]= cost_old(x, y, tstCsm,tstMask,t, net, A,opts)
            xsample[i] = x.abs().detach().cpu()
            tsample[i] = t.detach().cpu()

    if storeIntermediate:
        return x, xsample,tsample,denoised,xdc,acc_attempt_i,rate_i,posterior_energy
    else:
        return x
    

#######################
def mm_without_guidance(A, net, y, tstCsm,tstMask,opts):
    device = y.device
    t_steps = giveTsteps(opts.sigma_max,opts.sigma_min,opts.rho,opts.num_steps,device)
    #initialization
    Atb = A.adjoint(y,tstCsm)
    inference_std = opts.inference_std
    rhs=((1/(inference_std**2))*Atb)
    x_init = A.init(Atb, rhs, inference_std, t_steps[0], tstCsm, tstMask)
    
    #A.sense_sol(Atb,Atb,100,tstCsm,tstMask)
    x_iter =x_init
    ratio = (opts.inference_std)**2
    lip_const=opts.L
    for i in range(len(t_steps)):
        for k in range(opts.K):
            t = t_steps[i]
            cost, score = giveScore(torch.cat((x_iter.real,x_iter.imag),1), net, t.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
            score= score.detach()
            score = score[:,0,:,:] +1j* score[:,1,:,:]
            score = torch.unsqueeze(score,0)  
            rhs = Atb/ratio+ ((lip_const*x_iter)-(score) )/t**2
            x_iter = A.mm_inv(x_iter,rhs,tstCsm,tstMask,lip_const/t**2,1/ratio)
            
            
            
            
            
    return x_iter
    
#########################################################################


def cost_old(sample,measurement,tstCsm,tstMask,scale,net,A,opts):
    dc_adj = A.forward(sample,tstCsm,tstMask)
    nll = (torch.sum((dc_adj-measurement).abs()**2,dim =(1,2,3)))/ (2*opts.inference_std**2)

    nlp =giveScore(torch.cat((sample.real,sample.imag),1), net, scale.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)[0].detach()/(scale**2)
    
    return nll+nlp

def grad_C_old(sample,net,measurement,tstCsm,tstMask,scale,A, opts,pre_compute_score=True,score=None):
    if pre_compute_score:
        score = score
    else:
        _, score = giveScore(torch.cat((sample.real,sample.imag),1), net, scale.reshape(-1, 1, 1, 1),precondition=True,class_labels=opts.class_labels)
        score= score.detach()
        score = score[:,0,:,:] +1j* score[:,1,:,:]
        score = torch.unsqueeze(score,0)
    
    
    Atb = A.adjoint(measurement,tstCsm)
    grad_posterior = A.ATA(sample, tstCsm, tstMask)-Atb
    return grad_posterior/(opts.inference_std**2) +  score/scale**2
    


def quad_Binv_old(v, A, t,opts,tstCsm,tstMask):
    # v^T (A^T A / eta^2 + I / t^2) v  =  ||A v||^2 / eta^2 + ||v||^2 / t^2
    Av = A.forward(v,tstCsm,tstMask)
    return ((torch.sum((Av).abs()**2,dim =(1,2,3)))/ (opts.inference_std**2)) + (torch.sum((v).abs()**2,dim =(1,2,3)))/ (t**2)

def delta_old(old,new,score_old,scale,measurement,tstCsm,tstMask,A,net,opts):
    grad_old = grad_C_old(old,net,measurement,tstCsm,tstMask,scale,A, opts,pre_compute_score=True,score=score_old)
    grad_new =  grad_C_old(new,net,measurement,tstCsm,tstMask, scale,A, opts, pre_compute_score=False)
    precondition_new =A.dc(new, grad_new, tstCsm, tstMask,scale,opts.inference_std)
    precondition_old =A.dc(old, grad_old, tstCsm, tstMask,scale,opts.inference_std)
    v1 = old -new  + 0.5*opts.step_size**2*precondition_new
    v2 = new -old + 0.5*opts.step_size**2*precondition_old
   
    return quad_Binv_old(v1, A, scale,opts,tstCsm,tstMask) - quad_Binv_old(v2, A, scale, opts,tstCsm,tstMask)







def cg_quadratic(x_iter,denoised_img,score_std,inference_std,tstCsm,tstMask,b,A):
    
    
    
    
    Atb = A.adjoint(b,tstCsm)
    if x_iter.shape[1]!=1:
        x_iter = x_iter[:,0,:,:] +1j* x_iter[:,1,:,:]
        x_iter = torch.unsqueeze(x_iter,0)
        print('here')
        
    rhs = ((1/(inference_std**2))*Atb)  + ((1/(score_std**2))*denoised_img)
    x_iter = A.dc(denoised_img, rhs, tstCsm, tstMask,score_std,inference_std)

            
    return x_iter
    

def giveTsteps(sigma_max,sigma_min,rho,num_steps,device):
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return t_steps