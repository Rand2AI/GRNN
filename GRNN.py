import time, datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from Generator.model import Generator
from TFLogger.logger import TFLogger
from Backbone import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device0 = 0 # for GRNN training
    device1 = 1 # for local training, if you have only one GPU, please set device1 to 0
    batchsize = 1
    save_img = True # whether same generated image and its relevant true image
    Iteration = 1000 # how many optimization steps on GRNN
    num_exp = 10 # experiment number
    g_in = 128 # dimention of GRNN input
    plot_num = 30
    net_name = 'lenet' # global model
    net_name_set = ['lenet', 'res18']
    dataset = 'lfw'
    dataset_set = ['mnist', 'cifar100', 'lfw', 'VGGFace', 'ilsvrc']
    shape_img = (32, 32)
    root_path = os.path.abspath(os.curdir)
    data_path = os.path.join(root_path, 'Data/')
    save_path = os.path.join(root_path, f"Results/GRNN-{net_name}-{dataset}-S{shape_img[0]}-B{str(batchsize).zfill(3)}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/") # path to saving results
    print('>' * 10, save_path)
    save_img_path = os.path.join(save_path, 'saved_img/')
    dst, num_classes= gen_dataset(dataset, data_path, shape_img) # read local data
    tp = transforms.Compose([transforms.ToPILImage()])
    train_loader = iter(torch.utils.data.DataLoader(dst, batch_size=batchsize, shuffle=True))
    criterion = nn.CrossEntropyLoss().cuda(device1)
    print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: {save_path}')
    for idx_net in range(num_exp):
        train_tfLogger = TFLogger(f'{save_path}/tfrecoard-exp-{str(idx_net).zfill(2)}') # tensorboard record
        print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: running {idx_net+1}|{num_exp} experiment')
        if net_name == 'lenet':
            net = LeNet(num_classes=num_classes)
        elif net_name == 'res18':
            net = ResNet18(num_classes=num_classes)
        net = net.cuda(device1)
        Gnet = Generator(num_classes, channel=3, shape_img=shape_img[0],
                         batchsize=batchsize, g_in=g_in).cuda(device0)
        net.apply(weights_init)
        Gnet.weight_init(mean=0.0, std=0.02)
        G_optimizer = torch.optim.RMSprop(Gnet.parameters(), lr=0.0001, momentum=0.99)
        tv_loss = TVLoss()
        gt_data,gt_label = next(train_loader)
        gt_data, gt_label = gt_data.cuda(device1), gt_label.cuda(device1) # assign to device1 to generate true graident
        pred = net(gt_data)
        y = criterion(pred, gt_label)
        dy_dx = torch.autograd.grad(y, net.parameters()) # obtain true gradient
        flatten_true_g = flatten_gradients(dy_dx)
        G_ran_in = torch.randn(batchsize, g_in).cuda(device0) # initialize GRNN input
        iter_bar = tqdm(range(Iteration),
                        total=Iteration,
                        desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}',
                        ncols=180)
        history = []
        history_l = []
        for iters in iter_bar: # start  optimizing GRNN
            Gout, Glabel = Gnet(G_ran_in) # produce recovered data
            Gout, Glabel = Gout.cuda(device1), Glabel.cuda(device1) # assign to device1 as global model's input to generate fake gradient
            Gpred = net(Gout)
            Gloss = - torch.mean(torch.sum(Glabel * torch.log(torch.softmax(Gpred, 1)), dim=-1)) # cross-entropy loss
            G_dy_dx = torch.autograd.grad(Gloss, net.parameters(), create_graph=True) # obtain fake gradient
            flatten_fake_g = flatten_gradients(G_dy_dx).cuda(device1)
            grad_diff_l2 = loss_f('l2', flatten_fake_g, flatten_true_g, device1)
            grad_diff_wd = loss_f('wd', flatten_fake_g, flatten_true_g, device1)
            if net_name == 'lenet':
                tvloss = 1e-3 * tv_loss(Gout)
            elif net_name == 'res18':
                tvloss = 1e-6 * tv_loss(Gout)
            grad_diff = grad_diff_l2 + grad_diff_wd + tvloss # loss for GRNN
            G_optimizer.zero_grad()
            grad_diff.backward()
            G_optimizer.step()
            iter_bar.set_postfix(loss_l2 = np.round(grad_diff_l2.item(), 8),
                                 loss_wd=np.round(grad_diff_wd.item(), 8),
                                 loss_tv = np.round(tvloss.item(), 8),
                                 img_mses=round(torch.mean(abs(Gout-gt_data)).item(), 8),
                                 img_wd=round(wasserstein_distance(Gout.view(1,-1), gt_data.view(1,-1)).item(), 8))

            train_tfLogger.scalar_summary('g_l2', grad_diff_l2.item(), iters)
            train_tfLogger.scalar_summary('g_wd', grad_diff_wd.item(), iters)
            train_tfLogger.scalar_summary('g_tv', tvloss.item(), iters)
            train_tfLogger.scalar_summary('img_mses', torch.mean(abs(Gout-gt_data)).item(), iters)
            train_tfLogger.scalar_summary('img_wd', wasserstein_distance(Gout.view(1,-1), gt_data.view(1,-1)).item(), iters)
            train_tfLogger.scalar_summary('toal_loss', grad_diff.item(), iters)

            if iters % int(Iteration / plot_num) == 0:
                history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(batchsize)])
                history_l.append([Glabel.argmax(dim=1)[imidx].item() for imidx in range(batchsize)])
            torch.cuda.empty_cache()
            del Gloss, G_dy_dx, flatten_fake_g, grad_diff_l2, grad_diff_wd, grad_diff, tvloss

        # visualization
        for imidx in range(batchsize):
            plt.figure(figsize=(12, 8))
            plt.subplot(plot_num//10, 10, 1)
            plt.imshow(tp(gt_data[imidx].cpu()))
            for i in range(min(len(history), plot_num-1)):
                plt.subplot(plot_num//10, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.title('l=%d' % (history_l[i][imidx]))
                # plt.title('i=%d,l=%d' % (history_iters[i], history_l[i][imidx]))
                plt.axis('off')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if save_img:
                true_path = os.path.join(save_img_path, f'true_data/exp{str(idx_net).zfill(3)}/')
                fake_path = os.path.join(save_img_path, f'fake_data/exp{str(idx_net).zfill(3)}/')
                if not os.path.exists(true_path) or not os.path.exists(fake_path):
                    os.makedirs(true_path)
                    os.makedirs(fake_path)
                tp(gt_data[imidx].cpu()).save(os.path.join(true_path, f'{imidx}_{gt_label[imidx].item()}.png'))
                history[-1][imidx].save(os.path.join(fake_path, f'{imidx}_{Glabel.argmax(dim=1)[imidx].item()}.png'))
            plt.savefig(save_path + '/exp:%03d-imidx:%02d-tlabel:%d-Glabel:%d.png' % (idx_net,imidx , gt_label[imidx].item(),Glabel.argmax(dim=1)[imidx].item()))
            plt.close()

        del Glabel, Gout, flatten_true_g, G_ran_in, net, Gnet
        torch.cuda.empty_cache()
        history.clear()
        history_l.clear()
        iter_bar.close()
        train_tfLogger.close()
        print('----------------------')

if __name__ == '__main__':
    main()