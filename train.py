import torch

from models import *
from torchvision.transforms import functional
from torch.cuda.amp import GradScaler, autocast
import collections

VGG_MEAN = [123.68, 116.78, 103.94]
identity = torch.nn.Identity()

scaler = GradScaler()
lambda_gp=10.0
import random

# Computing EMA
class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


# VGG function for layer outputs
def VGG19_slim(input, reuse, deep_list=None, norm_flag=True):
    input_img = deprocess(input)
    input_img_ab = input_img * 255.0 - torch.tensor(VGG_MEAN)
    model = VGG19()
    _, output = model(input_img_ab)

    results = {}
    for key in output:
        if deep_list is None or key in deep_list:
            orig_deep_feature = output[key]
            if norm_flag:
                orig_len = torch.sqrt(torch.min(torch.square(orig_deep_feature), dim=1, keepdim=True) + 1e-12)
                results[key] = orig_deep_feature / orig_len
            else:
                results[key] = orig_deep_feature
    return results



def compute_gradient_penalty(network,real_samples,fake_samples):
    alpha=torch.cuda.FloatTensor(np.random.random((real_samples.size(0),1,1,1)))
    interpolates=((alpha*real_samples+(1-alpha)*fake_samples)).requires_grad_(True)
    out_plus_l2,out=network(interpolates)
    fake=torch.cuda.FloatTensor(np.ones(out_plus_l2.shape)).requires_grad_(False)
    gradients=torch.autograd.grad(outputs=out,inputs=interpolates,grad_outputs=fake,
                                  create_graph=True,retain_graph=True,only_inputs=True)[0]
    gradients=gradients.view(gradients.size(0),-1)
    gradients_penalty=((gradients.norm(2,dim=1)-1)**2).mean()
    return gradients_penalty

def TecoGAN_Was(r_inputs, r_targets, fnet, generator_F, args, Global_step, counter1, counter2, optimizer_g
            , optimizer_d,discriminator_Pic,discriminator_optimizer,scale,
            GAN_FLAG=True):
    # r_targets=r_targets[:,1,:,:,:].unsqueeze(1)
    Global_step += 1
    data_n, data_t, data_c, lr_h, lr_w = r_inputs.size()
    _, _, _, hr_h, hr_w = r_targets.size()
    inputimages = args.RNN_N
    outputimages= args.RNN_N
    # Getting inputs for Fnet using pingpang loss for forward and reverse videos

    output_channel = r_targets.shape[2]
    gen_outputs = []
    gen_warppre = []
    learning_rate = args.learning_rate
    Frame_t_pre = r_inputs[:, 0:-1, :, :, :]
    Frame_t = r_inputs[:, 1:, :, :, :]


    # Reshaping the fnet input and passing it to the model
    with autocast():
        fnet_input = fnet(Frame_t.reshape(data_n * (data_t - 1), data_c, lr_h, lr_w),
                          Frame_t_pre.reshape(data_n * (data_t - 1), data_c, lr_h, lr_w))

        gen_flow=fnet_input
        gen_flow = torch.reshape(gen_flow[:, 0:2],
                                 (data_n, (inputimages - 1), 2, args.crop_size1 * scale, args.crop_size2 * scale))

        input_frames = torch.reshape(Frame_t,
                                     (Frame_t.shape[0] * (inputimages - 1), output_channel, args.crop_size1,
                                      args.crop_size2))  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        s_input_warp = backward_warp(torch.reshape(Frame_t_pre, (
            Frame_t_pre.shape[0] * (inputimages - 1), output_channel, args.crop_size1, args.crop_size2)),
                                     gen_flow[:, 0, :, :,
                                     :])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        input0 = torch.cat(
            (r_inputs[:, 0, :, :, :], torch.zeros(size=(r_inputs.shape[0], data_c * scale * scale, args.crop_size1, args.crop_size2),
                                                  dtype=torch.float32).cuda()), dim=1)

        # Passing inputs into model and reshaping output
        gen_pre_output = generator_F(input0.detach())
        gen_pre_output = gen_pre_output.view(gen_pre_output.shape[0], data_c, args.crop_size1 * scale, args.crop_size2 * scale)
        gen_outputs.append(gen_pre_output)
        # Getting outputs of generator for each frame
        for frame_i in range(inputimages - 1):
            cur_flow = gen_flow[:, frame_i, :, :, :]
            gen_pre_output_warp=backward_warp(gen_pre_output, cur_flow.half().detach())
            gen_warppre.append(gen_pre_output_warp)

            # gen_pre_output_warp = preprocessLr(deprocess(gen_pre_output_warp))
            gen_pre_output_warp = preprocessLr(gen_pre_output_warp)
            gen_pre_output_reshape = gen_pre_output_warp.view(gen_pre_output_warp.shape[0], data_c, args.crop_size1, scale, args.crop_size2, scale)
            # gen_pre_output_reshape = gen_pre_output_reshape.permute(0, 1, 3, 5, 2, 4)

            gen_pre_output_reshape = torch.reshape(gen_pre_output_reshape,
                                                   (gen_pre_output_reshape.shape[0],data_c * scale * scale, args.crop_size1, args.crop_size2))
            inputs = torch.cat((r_inputs[:, frame_i + 1, :, :, :], gen_pre_output_reshape), dim=1)
            # inputs=torch.cat((r_inputs[:, frame_i + 1, :, :, :], r_inputs[:, frame_i , :, :, :]), dim=1)
            gen_output = generator_F(inputs.detach())
            gen_outputs.append(gen_output)
            gen_pre_output = gen_output
            gen_pre_output = gen_pre_output.view(gen_pre_output.shape[0], data_c, args.crop_size1 * scale, args.crop_size2 * scale)

        # Converting list of gen outputs and reshaping
        gen_outputs = torch.stack(gen_outputs, dim=1)
        gen_outputs = gen_outputs.view(gen_outputs.shape[0], outputimages, data_c, args.crop_size1 * scale, args.crop_size2 * scale)

        s_gen_output = torch.reshape(gen_outputs,
                                     (gen_outputs.shape[0] * outputimages, data_c, args.crop_size1 *scale, args.crop_size2 * scale))
        s_targets = torch.reshape(r_targets, (r_targets.shape[0] * outputimages, data_c, args.crop_size1 * scale, args.crop_size2 * scale))

        update_list = []#
        update_list_name = []#

        # Preparing vgg layers
###!!!!!!!!!!!!!!!
        if args.vgg_scaling > 0.0:
            vgg_layer_labels = ['vgg_19/conv2_2', 'vgg_19/conv3_4', 'vgg_19/conv4_4']
            gen_vgg = VGG19_slim(s_gen_output, deep_list=vgg_layer_labels)
            target_vgg = VGG19_slim(s_targets, deep_list=vgg_layer_labels)

###!!!

        if (GAN_FLAG):
            t_gen_output2 = torch.reshape(gen_outputs[:, outputimages - 1, :, :, :],
                                          (gen_outputs.shape[0] * 1, data_c, args.crop_size1 * scale,
                                           args.crop_size2 * scale))
            t_targets2 = torch.reshape(r_targets[:, outputimages - 1, :, :, :], (
                    r_targets.shape[0] * 1, data_c, args.crop_size1 * scale, args.crop_size2 * scale))
            dis_fake_output, fake_layer2 = discriminator_Pic(t_gen_output2.cuda())
            dis_real_output, real_layer2 = discriminator_Pic(t_targets2.cuda())
            #### Computing the layer loss using the VGG network and discriminator outputs
            if (args.D_LAYERLOSS):
                Fix_Range = 0.02
                Fix_margin = 0.0

                sum_layer_loss = 0
                d_layer_loss = 0

                layer_loss_list = []
                layer_n = len(real_layer2)
                layer_norm = [12.0, 14.0, 24.0, 100.0]
                for layer_i in range(layer_n):
                
                    real_layer22 = real_layer2[layer_i].detach()
                    fake_layer22 = fake_layer2[layer_i]
                    layer_diff2 = real_layer22- fake_layer22
                    layer_loss = torch.mean(torch.sum(torch.abs(layer_diff2), dim=[3]))



                    layer_loss_list += [layer_loss]

                    scaled_layer_loss = Fix_Range * layer_loss / layer_norm[layer_i]

                    sum_layer_loss += scaled_layer_loss
                    ###!!!!!!!!!!!!!!!
                    if Fix_margin > 0.0:
                        d_layer_loss += torch.max(0.0, torch.tensor(Fix_margin - scaled_layer_loss))
                    ###!!!
        # Computing the generator and fnet losses
        diff1_mse = s_gen_output- s_targets
        

        content_loss = torch.mean(torch.sum(torch.square(diff1_mse), dim=[3]))#
        gen_loss = content_loss###
        diff1_mse_sim=torch.mean(torch.sum(torch.square(gen_pre_output_reshape-t_gen_output2), dim=[3]))

        diff2_mse = input_frames - s_input_warp
        warp_loss = torch.mean(torch.sum(torch.square(diff2_mse), dim=[3]))
        fnet_loss = warp_loss
        vgg_loss = None
        vgg_loss_list = []
        ###!!!!!!!!!!!!!!!
        if args.vgg_scaling > 0.0:
            vgg_wei_list = [1.0, 1.0, 1.0, 1.0]
            vgg_loss = 0
            vgg_layer_n = len(vgg_layer_labels)

            for layer_i in range(vgg_layer_n):
                curvgg_diff = torch.sum(gen_vgg[vgg_layer_labels[layer_i]] * target_vgg[vgg_layer_labels[layer_i]],
                                        dim=[3])
                scaled_layer_loss = vgg_wei_list[layer_i] * curvgg_diff
                vgg_loss_list += [curvgg_diff]
                vgg_loss += scaled_layer_loss

            gen_loss += args.vgg_scaling * vgg_loss#########
            fnet_loss += args.vgg_scaling * vgg_loss.detach()
            vgg_loss_list += [vgg_loss]

            update_list += vgg_loss_list#
            update_list_name += ["vgg_loss_%d" % (_ + 2) for _ in range(len(vgg_loss_list) - 1)]#
            update_list_name += ["vgg_all"]#

        ###!!!
        if (GAN_FLAG):

            d_adversarial_loss2 = torch.mean(torch.log(1-dis_fake_output + args.EPS))
            t_adversarial_loss2 = torch.mean( torch.log(1-dis_fake_output.detach() + args.EPS))
            dt_ratio = torch.min(torch.tensor(args.Dt_ratio_max),
                                 args.Dt_ratio_0 + args.Dt_ratio_add * torch.tensor(Global_step, dtype=torch.float32))
        gen_loss+=args.ratio *t_adversarial_loss2.cpu()

        if (args.D_LAYERLOSS):
            gen_loss += sum_layer_loss.cpu() * dt_ratio ##############################
        gen_loss = gen_loss.cuda()

        # Computing discriminator loss
        if (GAN_FLAG):
            dis_fake_output, fake_layer2 = discriminator_Pic(t_gen_output2.cuda().detach())
            dis_real_output, real_layer2 = discriminator_Pic(t_targets2.cuda().detach())
            t_discrim_fake_loss2=torch.log(1-dis_fake_output+args.EPS)
            t_discrim_real_loss2=torch.log(dis_real_output+args.EPS)

            t_discrim_loss2 = -(torch.mean(t_discrim_fake_loss2) + torch.mean(t_discrim_real_loss2) +args.EPS)

            t_balance = torch.mean(t_discrim_real_loss2) + d_adversarial_loss2
            #
            ###!!!!!!!!!!!!!!!
            if (args.D_LAYERLOSS and Fix_margin > 0.0):
                discrim_loss = t_discrim_loss2 + d_layer_loss * dt_ratio
            ###!!!
            # Computing gradients for Discriminator and updating weights
            else:
                discrim_loss = t_discrim_loss2  ##############################

            t_discrim_loss2=t_discrim_loss2.cuda()
            tb_exp_averager = EMA(0.99)
            init_average = torch.zeros_like(t_balance)
            tb_exp_averager.register("TB_average", init_average)
            tb = tb_exp_averager.forward("TB_average", t_balance)


            tb_exp_averager.register("Loss_average", init_average)
            update_list_avg = [tb_exp_averager.forward("Loss_average", _) for _ in update_list]#
        # Computing gradients for fnet and updating weights
        optimizer_g.zero_grad()
        scaler.scale(gen_loss).backward()
        scaler.step(optimizer_g)
        scaler.update()
        # optimizer_d.zero_grad()
        scaler.scale(fnet_loss).backward()
        scaler.step(optimizer_d)
        scaler.update()
        discriminator_optimizer.zero_grad()
        scaler.scale(t_discrim_loss2).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()


        update_list_avg += [tb, dt_ratio] #
        update_list_name += ["t_balance", "Dst_ratio"]#

        update_list_avg += [counter1, counter2]#
        update_list_name += ["withD_counter", "w_o_D_counter"]#
    max_outputs = min(4, r_targets.shape[0])
    # Returning output tuple
    Network = collections.namedtuple('Network', 'gen_output, learning_rate, update_list, '
                                                'update_list_name, update_list_avg, global_step, d_loss, gen_loss, '
                                                'fnet_loss ,tb')
    return Network(
        gen_output=gen_outputs,
        learning_rate=learning_rate,
        update_list=update_list,
        update_list_name=update_list_name,
        update_list_avg=update_list_avg,
        global_step=Global_step,
        d_loss=t_discrim_loss2,
        gen_loss=gen_loss,
        fnet_loss=fnet_loss,
        tb=tb,
    )














def FRVSR_Train_Was(r_inputs, r_targets, args, discriminator_F, generator_F, step, counter1, counter2, optimizer_g,
                optimizer_d,discriminator_Pic,discriminator_optimizer):
    return TecoGAN_Was(r_inputs, r_targets, discriminator_F, generator_F, args, step, counter1, counter2,
                   optimizer_g, optimizer_d,discriminator_Pic,discriminator_optimizer,scale=1)