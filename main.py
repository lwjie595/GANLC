import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
import torchvision
from torch.utils.data import DataLoader
import time
import cv2
from torch.cuda.amp import autocast
from skimage.measure import compare_psnr,compare_ssim
from sklearn import metrics

sys.path.insert(1, './code')
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(48)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

####Rider dataset use train_dataset2
from train import FRVSR_Train_Was
from dataloader import train_dataset, inference_dataset
from models import generator
from tqdm import tqdm
from ops import *
from models import *

# All arguments. These are the same arguments as in the original TecoGan repo. I might prune them at a later date.

parser = argparse.ArgumentParser()
parser.add_argument('--rand_seed', default=1, type=int, help='random seed')

# Directories
parser.add_argument('--input_dir_LR', default='', nargs="?",
                    help='The directory of the input resolution input data, for inference mode')
parser.add_argument('--input_dir_len', default=-1, type=int,
                    help='length of the input for inference mode, -1 means all')
parser.add_argument('--input_dir_HR', default='', nargs="?",
                    help='The directory of the input resolution input data, for inference mode')
parser.add_argument('--mode', default='train', nargs="?", help='train, or inference')
parser.add_argument('--output_dir', default="output", help='The output directory of the checkpoint')
parser.add_argument('--output_pre', default='', nargs="?", help='The name of the subfolder for the images')
parser.add_argument('--output_name', default='output', nargs="?", help='The pre name of the outputs')
parser.add_argument('--output_ext', default='jpg', nargs="?", help='The format of the output when evaluating')
parser.add_argument('--summary_dir', default="summary", nargs="?", help='The dirctory to output the summary')
parser.add_argument('--videotype', default=".mp4", type=str, help="Video type for inference output")
parser.add_argument('--inferencetype', default="dataset", type=str, help="The type of input to the inference loop. "
                                                                         "Either video or dataset folder.")

# Models
parser.add_argument('--g_checkpoint', default='./output/pattient9/test_discrim3_fbpnet_num2_crop512_batch_4_15_nodetach_2/generator_99.pt',
                    help='If provided, the generator will be restored from the provided checkpoint')
parser.add_argument('--d_checkpoint', default='./output/pattient9/test_discrim3_fbpnet_num2_crop512_batch_4_15_nodetach_2/discrim_99.pt', nargs="?",
                    help='If provided, the discriminator will be restored from the provided checkpoint')
parser.add_argument('--pic_checkpoint', default='./output/pattient9/test_discrim3_fbpnet_num2_crop512_batch_4_15_nodetach_2/discrim_pic_99.pt', nargs="?",
                    help='If provided, the discriminator will be restored from the provided checkpoint')
# parser.add_argument('--f_checkpoint', default=None, nargs="?",
#                  help='If provided, the fnet will be restored from the provided checkpoint')
parser.add_argument('--num_resblock', type=int, default=16, help='How many residual blocks are there in the generator')
parser.add_argument('--discrim_resblocks', type=int, default=4, help='Number of resblocks in each resnet layer in the '
                                                                     'discriminator')
parser.add_argument('--discrim_channels', type=int, default=128, help='How many channels to use in the last two '
                                                                      'resnet blocks in the discriminator')
# Models for training
parser.add_argument('--pre_trained_model', type=str2bool, default=False,
                    help='If True, the weight of generator will be loaded as an initial point'
                         'If False, continue the training')
parser.add_argument('--vgg_ckpt', default=None, help='path to checkpoint file for the vgg19')

# Machine resources
parser.add_argument('--cudaID', default='1',type=str,  help='CUDA devices')
parser.add_argument('--queue_thread', default=8, type=int,
                    help='The threads of the queue (More threads can speedup the training process.')

# Training details
# The data preparing operation

parser.add_argument('--RNN_N', default=2, nargs="?", help='The number of the rnn recurrent length')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size of the input batch')
parser.add_argument('--flip', default=True, type=str2bool, help='Whether random flip data augmentation is applied')
parser.add_argument('--random_crop', default=True, type=str2bool, help='Whether perform the random crop')
parser.add_argument('--movingFirstFrame', default=True, type=str2bool,
                    help='Whether use constant moving first frame randomly.')
parser.add_argument('--crop_size1', default=512, type=int, help='The crop size of the training image with height')
parser.add_argument('--crop_size2', default=512, type=int, help='The crop size of the training image with width')
parser.add_argument('--crop_size', default=512, type=int, help='The crop size of the training image with width')
# Training data settings
parser.add_argument('--input_video_dir', type=str, default="./data_download/mayo_data_arranged_patientwise0",
                    help='The directory of the video input data, for training')
parser.add_argument('--input_video_pre', default='./data_download/bio_train/train_1.txt', type=str,
                    help='The pre of the directory of the video input data')
parser.add_argument('--valid_video_pre', default='./data_download/bio_valid/valid_1.txt', type=str,
                    help='The pre of the directory of the video input data')
parser.add_argument('--str_dir', default=1, type=int, help='The starting index of the video directory')
parser.add_argument('--end_dir', default=200, type=int, help='The ending index of the video directory')
parser.add_argument('--end_dir_val', default=2050, type=int,
                    help='The ending index for validation of the video directory')
parser.add_argument('--max_frm', default=25, type=int, help='The ending index of the video directory')
# The loss parameters

parser.add_argument('--vgg_scaling', default=-1.0,
                    type=float, help='The scaling factor for the VGG perceptual loss, disable with negative value')
parser.add_argument('--warp_scaling', default=1.0, type=float, help='The scaling factor for the warp')
parser.add_argument('--pingpang', default=False, type=str2bool, help='use bi-directional recurrent or not')
parser.add_argument('--pp_scaling', default=1.0, type=float,
                    help='factor of pingpang term, only works when pingpang is True')
# Training parameters
parser.add_argument('--EPS', default=1e-5, type=float, help='The eps added to prevent nan')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='The learning rate for the network')
parser.add_argument('--decay_step', default=100, type=int, help='The steps needed to decay the learning rate')
parser.add_argument('--decay_rate', default=0.8, type=float, help='The decay rate of each decay step')
parser.add_argument('--stair', default=False, type=str2bool,
                    help='Whether perform staircase decay. True => decay in discrete interval.')
parser.add_argument('--beta', default=0.9, type=float, help='The beta1 parameter for the Adam optimizer')
parser.add_argument('--adameps', default=1e-8, type=float, help='The eps parameter for the Adam optimizer')
parser.add_argument('--max_epochs', default=100, type=int, help='The max epoch for the training')
parser.add_argument('--savefreq', default=10, type=int, help='The save frequence for training')
parser.add_argument('--savepath', default="./output/pattient9/test_discrim3_fbpnet_num2_crop512_batch_4_15_nodetach_2/", type=str, help='The save path for training')
#
# Dst parameters
parser.add_argument('--ratio', default=0.01, type=float, help='The ratio between content loss and adversarial loss')
parser.add_argument('--Dt_mergeDs', default=True, type=str2bool, help='Whether only use a merged Discriminator.')
parser.add_argument('--Dt_ratio_0', default=1.0, type=float,
                    help='The starting ratio for the temporal adversarial loss')
parser.add_argument('--Dt_ratio_add', default=0.0, type=float,
                    help='The increasing ratio for the temporal adversarial loss')
parser.add_argument('--Dt_ratio_max', default=1.0, type=float, help='The max ratio for the temporal adversarial loss')
parser.add_argument('--Dbalance', default=0.4, type=float, help='An adaptive balancing for Discriminators')
parser.add_argument('--crop_dt', default=0.75, type=float,
                    help='factor of dt crop')  # dt input size = crop_size*crop_dt
parser.add_argument('--D_LAYERLOSS', default=True, type=str2bool, help='Whether use layer loss from D')
Scale=1
args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.cudaID

##Checking to make sure necessary args are filled.
if args.output_dir is None:
    raise ValueError("The output directory is needed")
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.exists(args.summary_dir):
    os.mkdir(args.summary_dir)

# an inference mode that I will complete soon
if args.mode == "inference":
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    dataset = train_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    generator_F = generator(1, args=args).cuda()

    g_checkpoint = torch.load(args.g_checkpoint)
    generator_F.load_state_dict(g_checkpoint["model_state_dict"])
    PSNR = 0
    SSIM = 0
    PSNR2 = 0
    SSIM2 = 0
    RMSE,RMSE2=0,0
    pic_position = '/picture_99_fnet/'
    if not os.path.exists(args.savepath + pic_position):
        os.makedirs(args.savepath + pic_position)
    logfile = open(args.savepath + pic_position + 'PSNR.txt', 'w+')
    fnet = FNet(1).cuda()
    d_checkpoint=torch.load(args.d_checkpoint)
    fnet.load_state_dict(d_checkpoint["model_state_dict"])
    PSNR_10,SSIM_10=[],[]
    PP10, SS10 = 0, 0
    with torch.no_grad():
        with autocast():
            for batch_idx, (r_inputs, r_targets, r) in enumerate(loader):
                if len(r.shape)==1:
                    break
                if batch_idx%235==1:
                    PSNR_10.append(PP10/235)
                    SSIM_10.append(SS10/235)
                    PP10, SS10 = 0, 0
                r_inputs = r_inputs.cuda()
                r_targets = r_targets.cuda()
                output_channel = r_inputs.shape[2]
                inputimages = r_inputs.shape[1]
                gen_outputs = []
                gen_warppre = []
                learning_rate = args.learning_rate
                data_n, data_t, data_c, lr_h, lr_w = r_inputs.size()
                _, _, _, hr_h, hr_w = r_targets.size()
                Frame_t_pre = r_inputs[:, 0:-1, :, :, :]
                Frame_t = r_inputs[:, 1:, :, :, :]

                fnet_input = fnet(Frame_t.reshape(data_n * (data_t - 1), data_c, lr_h, lr_w),
                                  Frame_t_pre.reshape(data_n * (data_t - 1), data_c, lr_h, lr_w))
                gen_flow = fnet_input
                gen_flow = torch.reshape(gen_flow[:, 0:2],
                                         (data_n, (inputimages - 1), 2, args.crop_size * Scale, args.crop_size * Scale))

                input0 = torch.cat(
                    (r_inputs[:, 0, :, :, :],
                     torch.zeros(size=(1, data_c * Scale * Scale, args.crop_size, args.crop_size),
                                 dtype=torch.float32).cuda()), dim=1)
                # Passing inputs into model and reshaping output
                gen_pre_output = generator_F(input0.cuda())
                gen_pre_output = gen_pre_output.view(gen_pre_output.shape[0], data_c, args.crop_size * Scale,
                                                     args.crop_size * Scale)
                # gen_outputs.append(gen_pre_output)
                # Getting outputs of generator for each frame
                for frame_i in range(inputimages - 1):
                    cur_flow = gen_flow[:, frame_i, :, :, :]
                    gen_pre_output_warp = backward_warp(gen_pre_output, cur_flow)
                    gen_warppre.append(gen_pre_output_warp)

                    gen_pre_output_warp = preprocessLr(gen_pre_output_warp)
                    gen_pre_output_reshape = gen_pre_output_warp.view(gen_pre_output_warp.shape[0], data_c,
                                                                      args.crop_size, Scale, args.crop_size,
                                                                      Scale)

                    gen_pre_output_reshape = torch.reshape(gen_pre_output_reshape,
                                                           (gen_pre_output_warp.shape[0], data_c * Scale * Scale,
                                                            args.crop_size, args.crop_size))
                    inputs = torch.cat((r_inputs[:, frame_i + 1, :, :, :], gen_pre_output_reshape), dim=1)
                    gen_output = generator_F(inputs.cuda())
                    gen_outputs.append(gen_output)
                    gen_pre_output = gen_output
                # Converting list of gen outputs and reshaping
                gen_outputs = torch.stack(gen_outputs, dim=1)

                name = batch_idx
                if batch_idx<50:
                    torchvision.utils.save_image(
                        gen_outputs.view(1,1,args.crop_size1 *Scale,args.crop_size2 *Scale),
                        fp='{}'.format(args.savepath) + pic_position + '{}_{}_G.png'.format(name, batch_idx))
                    torchvision.utils.save_image(
                        r_targets[:,1,:].view(1,1,args.crop_size1 *Scale,args.crop_size2 *Scale),
                        fp='{}'.format(args.savepath) + pic_position + '{}_{}_real.png'.format(name, batch_idx))

                    torchvision.utils.save_image(
                        r_inputs[:,1,:].view(1,1,args.crop_size1 *Scale,args.crop_size2 *Scale),
                        fp='{}'.format(args.savepath) + pic_position + '{}_{}_input.png'.format(name, batch_idx))
                    torchvision.utils.save_image(
                        r[:,1,:].view(1,1,args.crop_size1 *Scale,args.crop_size2 *Scale),
                        fp='{}'.format(args.savepath) + pic_position + '{}_{}_input_fb.png'.format(name, batch_idx))

                r_targets = r_targets[:,1,:,:,:].cpu().detach().numpy().squeeze()#
                r_targets = cut_image(r_targets, vmin=0.0, vmax=1.0)
                gen_outputs = gen_outputs.cpu().detach().numpy().squeeze()
                data_range = np.max(r_targets) - np.min(r_targets)
                fbp_image = r_inputs[:,1,:,:,:].cpu().detach().numpy().squeeze()#[:,1,:,:,:]
                for irange in range(inputimages-1):#
                    gen_outputs = cut_image(gen_outputs, vmin=0.0, vmax=data_range)
                    # gen_outputs = (gen_outputs - np.min(gen_outputs)) / (np.max(gen_outputs) - np.min(gen_outputs))
                    fbp_image = cut_image(fbp_image, vmin=0.0, vmax=data_range)
                    psnr_gan_ar = compare_psnr(r_targets, gen_outputs, data_range=data_range)
                    ssim_gan_ar = compare_ssim(r_targets, gen_outputs, data_range=data_range)
                    psnr_fbp = compare_psnr(r_targets, fbp_image, data_range=data_range)
                    ssim_fbp = compare_ssim(r_targets, fbp_image, data_range=data_range)
                    rmse=np.sqrt(metrics.mean_squared_error(r_targets,gen_outputs))
                    rmse_fbp = np.sqrt(metrics.mean_squared_error(r_targets, fbp_image))
                    PSNR += psnr_gan_ar
                    SSIM += ssim_gan_ar
                    PSNR2 += psnr_fbp
                    SSIM2 += ssim_fbp
                    PP10+=psnr_gan_ar
                    SS10+=ssim_gan_ar
                    RMSE+=rmse
                    RMSE2+=rmse_fbp
                    mess = 'batch idx:{}  psnr:{}  ssim:{} \n'.format(batch_idx, psnr_gan_ar, ssim_gan_ar)
                    mess2 = 'FBP batch idx :{}  psnr:{}  ssim:{} \n'.format(batch_idx, psnr_fbp, ssim_fbp)
                    logfile.write(mess)
                    logfile.write(mess2)
                    print(mess)
                    print(mess2)
                    Batch_idx=batch_idx
    batch_idx=Batch_idx
    print('batch idx:', batch_idx)
    PSNR = PSNR / (batch_idx )
    SSIM = SSIM / (batch_idx )
    PSNR2 = PSNR2 / (batch_idx )
    SSIM2 = SSIM2 / (batch_idx )
    RMSE=RMSE/ (batch_idx )
    RMSE2=RMSE2/(batch_idx )
    mess = 'Final average:{}  psnr:{}  ssim:{} rmse:{} \n'.format(batch_idx + 1, PSNR, SSIM,RMSE)
    mess2 = 'FBP Final batch idx :{}  psnr:{}  ssim:{}  rmse:{} \n'.format(batch_idx + 1, PSNR2, SSIM2,RMSE2)
    print(PSNR_10)
    print(SSIM_10)
    print(mess)
    logfile.write(mess)
    logfile.write(mess2)
    print(mess2)
    logfile.close()


elif args.mode == "train":
    # Defining dataset and dataloader
    dataset = train_dataset(args)
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    logfile = open(args.savepath  + 'train1.txt', 'w+')
    # Defining the models as well as the optimizers and lr schedulers
    generator_F = generator(1, args=args).cuda()
    discriminator_F = FNet(1).cuda()
    discriminator_Pic=discriminator_pic(args=args).cuda()
    counter1 = 0.
    counter2 = 0.
    min_gen_loss = np.inf
    tdis_learning_rate = args.learning_rate
    if not args.Dt_mergeDs:
        tdis_learning_rate = tdis_learning_rate * 0.3
    tdiscrim_optimizer = torch.optim.Adam(discriminator_F.parameters(), args.learning_rate,
                                          betas=(args.beta, 0.999),
                                          eps=args.adameps)
    gen_optimizer = torch.optim.Adam(generator_F.parameters(), args.learning_rate, betas=(args.beta, 0.999),
                                     eps=args.adameps)
    discriminator_optimizer=torch.optim.Adam(discriminator_Pic.parameters(),tdis_learning_rate,
                                             betas=(args.beta,0.999),eps=args.adameps)
    # fnet_optimizer = torch.optim.Adam(fnet.parameters(), args.learning_rate, betas=(args.beta, 0.999),
    # eps=args.adameps)
    GAN_FLAG = True
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(tdiscrim_optimizer, T_max=args.decay_step, eta_min=args.learning_rate/100)
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=args.decay_step, eta_min=args.learning_rate/100)
    d_pic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discriminator_optimizer, T_max=args.decay_step, eta_min=args.learning_rate/100)
    if args.pre_trained_model:
        g_checkpoint = torch.load(args.g_checkpoint)
        generator_F.load_state_dict(g_checkpoint["model_state_dict"])
        gen_optimizer.load_state_dict(g_checkpoint["optimizer_state_dict"])
        for param_group in gen_optimizer.param_groups:
            gen_lr = param_group["lr"]
            print(gen_lr)
        current_epoch = g_checkpoint["epoch"]+1
        d_checkpoint = torch.load(args.d_checkpoint)
        discriminator_F.load_state_dict(d_checkpoint["model_state_dict"])
        tdiscrim_optimizer.load_state_dict(d_checkpoint["optimizer_state_dict"])
        pic_checkpoint = torch.load(args.pic_checkpoint)
        discriminator_Pic.load_state_dict(pic_checkpoint["model_state_dict"])
        discriminator_optimizer.load_state_dict(pic_checkpoint["optimizer_state_dict"])
    else:
        current_epoch = 0
    # Starting epoch loop
    since = time.time()
    for e in tqdm(range(current_epoch, args.max_epochs)):
        d_loss = 0.
        g_loss = 0.
        f_loss = 0.
        batch_idx=-1
        for data in dataloader:
            batch_idx=batch_idx+1
            if batch_idx >= 540:
                a = 1
            inputs,targets=data[0],data[1]
            inputs = inputs.cuda()
            targets = targets.cuda()
            # Passing targets and inputs to the train function
            output = FRVSR_Train_Was(inputs, targets, args, discriminator_F, generator_F, batch_idx, counter1,
                                 counter2, gen_optimizer, tdiscrim_optimizer,discriminator_Pic,discriminator_optimizer)
            if (output.gen_loss.data>1 or output.gen_loss.data<0):
                print('g_loss: {} , batch idx:{}'.format(output.gen_loss.data, batch_idx))
            # Computing epoch losses
            f_loss = f_loss + ((1 / (batch_idx + 1)) * (output.fnet_loss.data - f_loss))

            g_loss = g_loss + ((1 / (batch_idx + 1)) * (output.gen_loss.data - g_loss))

            d_loss = d_loss + ((1 / (batch_idx + 1)) * (output.d_loss.data - d_loss))
        # Saving outputs as gifs and images
        print('--------------------------------')
        print('Final g_loss: {} , batch idx:{}'.format(output.gen_loss.data, batch_idx))
        print('--------------------------------')
        index = np.random.randint(0, targets.shape[0])
        d_scheduler.step()
        g_scheduler.step()
        d_pic_scheduler.step()
        # Printing out metrics
        print("Epoch: {}".format(e + 1))

        print("\nGenerator loss is: {} \nDiscriminator loss is: {}".format(g_loss, d_loss))
        for param_group in gen_optimizer.param_groups:
            gen_lr = param_group["lr"]
        for param_group in tdiscrim_optimizer.param_groups:
            disc_lr = param_group["lr"]
        mess = 'batch idx:{} f_loss{} g_loss:{}  d_loss:{}, generator lr :{}, Discriminator lr:{}\n'.format(e,f_loss, g_loss, d_loss,gen_lr,disc_lr)
        logfile.write(mess)
        print(f"\nGenerator lr is: {gen_lr}, Discriminator lr is: {disc_lr}")
        print("\nSaving model...")
        # Saving the models
        torch.save({
            'epoch': e,
            'model_state_dict': generator_F.state_dict(),
            'optimizer_state_dict': gen_optimizer.state_dict(),
        }, args.savepath+"generator.pt")

        torch.save({
            'model_state_dict': discriminator_F.state_dict(),
            'optimizer_state_dict': tdiscrim_optimizer.state_dict(),
        }, args.savepath+"discrim.pt")
        torch.save({
            'model_state_dict': discriminator_Pic.state_dict(),
            'optimizer_state_dict': discriminator_optimizer.state_dict(),
        }, args.savepath+"discrim_pic.pt")
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        if e % (args.savefreq ) ==9:
            torch.save({
                'epoch': e,
                'model_state_dict': generator_F.state_dict(),
                'optimizer_state_dict': gen_optimizer.state_dict(),
            }, args.savepath + "generator_{}.pt".format( e))

            torch.save({
                'model_state_dict': discriminator_F.state_dict(),
                'optimizer_state_dict': tdiscrim_optimizer.state_dict(),
            }, args.savepath + "discrim_{}.pt".format(e))
            torch.save({
                'model_state_dict': discriminator_Pic.state_dict(),
                'optimizer_state_dict': discriminator_optimizer.state_dict(),
            }, args.savepath + "discrim_pic_{}.pt".format(e))

    logfile.close()
