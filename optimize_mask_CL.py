from __future__ import division

import PIL.Image as Image
import numpy as np
import os
import pdb
import time
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from collections import OrderedDict
from copy import deepcopy
from torch.utils.data import DataLoader, RandomSampler

import config
import models
import poisoned_dataset

args = config.get_arguments().parse_args()

if os.path.exists(args.output_dir):
    print('dir exists', args.output_dir)
    pdb.set_trace()
else:
    os.mkdir(args.output_dir)
    print(args.output_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def img_read(img_path):
    return Image.open(img_path).convert('RGB')


def split_dataset(dataset, val_frac=0.1, perm=None):
    """
    :param dataset: The whole dataset which will be split. DataFolder修改其中的samples
    :param val_frac: the fraction of validation set.
    :param perm: A predefined permutation for sampling. If perm is None, generate one.
    :return: A validation set
    """
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_val = int(val_frac * len(dataset))

    # generate the training set
    val_set = deepcopy(dataset)
    samples_temp = list()
    # pdb.set_trace()
    for i in perm[:nb_val]:
        samples_temp.append(val_set.samples[i])
        # pdb.set_trace()
    val_set.samples = samples_temp
    return val_set


def main():
    # pdb.set_trace()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    # load network
    state_dict = torch.load(args.test_model, map_location=device)
    net = getattr(models, args.arch)(num_classes=10, norm_layer=models.NoisyBatchNorm2d)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    dataset = torchvision.datasets.DatasetFolder
    trainset = dataset(
        root=os.path.join(args.data_root, 'train'),  # './data/cifar10/train'
        loader=img_read,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = dataset(
        root=os.path.join(args.data_root, 'test'),
        loader=img_read,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    poisoned_test_dataset = poisoned_dataset.get_dataset(trainset, testset, args)

    clean_val = split_dataset(trainset, val_frac=args.val_frac,
                              perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    clean_test = testset
    poison_test = poisoned_test_dataset
    # repeated sampling for clean val set
    random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples=len(clean_val) * 5)
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0)

    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

    # Step 2: load model checkpoints and trigger info
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

    # Step 3: get the target label and perform inversion
    target_label = get_targetlabel(model=net, criterion=criterion, data_loader=clean_val_loader,
                                   noise_opt=noise_optimizer)
    print('target label', target_label)
    trigger, mask = inversion(model=net, criterion=criterion, target_label=target_label, data_loader=clean_val_loader)
    # Step 4: optimize the mask values
    print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    nb_repeat = int(np.ceil(150 / 5))  # 2000/500=4
    ## additional add
    cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
        (0) * args.print_every, 0.2, 0, 0, 0, po_test_loss, po_test_acc,
        cl_test_loss, cl_test_acc))
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(mask_optimizer, factor=0.5, mode='max', patience=2,
                                                              cooldown=0, min_lr=0.01)
    for i in range(nb_repeat):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train_modify(model=net, criterion=criterion, data_loader=clean_val_loader,
                                                  mask_opt=mask_optimizer, noise_opt=noise_optimizer, trigger=trigger,
                                                  mask=mask)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        end = time.time()
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * 5, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))

        save_mask_scores(net.state_dict(), os.path.join(args.output_dir, str(i) + 'mask_values.txt'))
        scheduler_lr.step(train_acc)


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()  # delete '_noisy'
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):

    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def keep_orimodel(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.keep_orimodel()


def keep_maskmodel(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.keep_maskmodel()


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def get_targetlabel(model, criterion, noise_opt, data_loader):
    model.eval()

    nb_samples = 0

    for i, (images, _) in enumerate(data_loader):
        images = images.to(device)

        nb_samples += images.size(0)
        # step 0 : predict the pseudo label
        exclude_noise(model)
        prediction_pseudo = model(images)
        pseudo_labels = prediction_pseudo.data.max(1)[1]

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.anp_steps):  # args.anp_steps=1
                noise_opt.zero_grad()

                include_noise(model)  # perturb BN layers
                output_noise = model(images)

                loss_noise = - criterion(output_noise, pseudo_labels)  # optimize the perturbation to maximize the loss

                loss_noise.backward()
                sign_grad(model)  # set the gradients on BN layers
                noise_opt.step()  # only update the perturbation on BN layers

        # step 2: get the pred_labels after perturbation
        if args.anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            exclude_noise(model)
        pred_labels = output_noise.data.max(1)[1]

        if i == 0:
            pseudo_labels_all = deepcopy(pseudo_labels.cpu().detach())
            pred_labels_all = deepcopy(pred_labels.cpu().detach())
        else:
            pseudo_labels_all = torch.cat((pseudo_labels_all, deepcopy(pseudo_labels.cpu().detach())))
            pred_labels_all = torch.cat((pred_labels_all, deepcopy(pred_labels.cpu().detach())))

    # step 3: judge the target label
    label_increases = list()
    for i in range(args.classes_num):
        label_increase = (pred_labels_all == i).sum() - (pseudo_labels_all == i).sum()
        label_increases.append(label_increase.item())
    target_label = np.argmax(label_increases)
    # pdb.set_trace()
    return target_label


def inversion(model, criterion, target_label, data_loader):
    print("Processing label: {}".format(target_label))

    width, height = args.input_height, args.input_height
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = 20

    min_norm = np.inf
    min_norm_count = 0

    optimizer = torch.optim.Adam([{"params": trigger}, {"params": mask}], lr=0.005)
    model.to(device)
    model.eval()
    keep_orimodel(model)
    for epoch in range(Epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(data_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + args.lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    attack_with_trigger(model, data_loader, target_label, trigger, mask)
    keep_maskmodel(model)
    return trigger, mask  # CPU tensor


def attack_with_trigger(model, data_loader, target_label, trigger, mask):
    correct = 0
    total = 0
    trigger = trigger.to(device)
    mask = mask.to(device)
    model.eval()

    with torch.no_grad():
        for images, _ in tqdm.tqdm(data_loader):
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)

            _, y_pred = y_pred.max(1)
            correct += y_pred.eq(y_target).sum().item()
            total += images.size(0)

        print(correct / total)

    return trigger.cpu(), mask.cpu()


def mask_train_modify(model, criterion, mask_opt, noise_opt, data_loader, trigger, mask):
    model.train()

    total_correct = 0
    total_loss = 0.0
    nb_samples = 0

    trigger = trigger.to(device)
    mask = mask.to(device)
    cos = torch.nn.CosineSimilarity(dim=-1)
    for i, (images, _) in enumerate(data_loader):
        images = images.to(device)
        poisoned_images = deepcopy(images)
        poisoned_images = (1 - torch.unsqueeze(mask, dim=0)) * poisoned_images + torch.unsqueeze(mask, dim=0) * trigger

        nb_samples += images.size(0)
        # step 0 : predict the pseudo label
        exclude_noise(model)
        prediction_pseudo = model(images)
        pseudo_labels = prediction_pseudo.data.max(1)[1]

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.anp_steps):  # args.anp_steps=1
                noise_opt.zero_grad()

                include_noise(model)  # perturb BN layers
                output_noise = model(images)

                loss_noise = - criterion(output_noise, pseudo_labels)  # optimize the perturbation to maximize the loss

                loss_noise.backward()
                sign_grad(model)  # set the gradients on BN layers
                noise_opt.step()  # only update the perturbation on BN layers

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        include_noise(model)
        feature_c_noise = model.get_final_fm(images)
        exclude_noise(model)
        feature_c_mask = model.get_final_fm(images)
        feature_p_mask = model.get_final_fm(poisoned_images)

        output_p_mask = model(poisoned_images)
        keep_orimodel(model)
        feature_c_ori = model.get_final_fm(images)
        output_c_ori = model(images)
        feature_p_ori = model.get_final_fm(poisoned_images)
        keep_maskmodel(model)

        # posi1=cos(feature_c_mask,feature_p_mask).reshape(-1,1)
        posi2 = cos(feature_c_mask, feature_c_ori).reshape(-1, 1)
        posi3 = cos(feature_p_mask, feature_c_ori).reshape(-1, 1)
        posi4 = cos(feature_c_noise, feature_c_ori).reshape(-1, 1)

        nega1 = cos(feature_c_noise, feature_p_ori).reshape(-1, 1)
        nega2 = cos(feature_c_mask, feature_p_ori).reshape(-1, 1)
        nega3 = cos(feature_p_mask, feature_p_ori).reshape(-1, 1)
        logits = torch.cat((posi2, posi3, posi4, nega1, nega2, nega3), dim=1)

        logits /= args.temperature

        loss = 0
        for i in range(3):
            labels = (torch.ones(images.size(0)) * i).cuda().long()  # [64]

            loss += criterion(logits, labels) / 3
        total_loss += loss.item()

        pred_c_ori = output_c_ori.data.max(1)[1]
        pred_p_mask = output_p_mask.data.max(1)[1]
        total_correct += pred_c_ori.eq(pred_p_mask).sum()

        loss.backward()
        mask_opt.step()  # only update the mask values
        clip_mask(model)

    loss = total_loss / len(data_loader)

    return loss, total_correct / nb_samples


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


if __name__ == '__main__':
    main()
