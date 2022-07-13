import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
import models.resnet as resnet

from tqdm import tqdm
from loguru import logger
from models.ADSH_Loss import ADSH_Loss
from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.SEMICON as SEMICON

def train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):

    num_classes, att_size, feat_size = args.num_classes, 1, 2048
    model = SEMICON.semicon(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                            device=args.device, pretrained=True)


    # print(list(model.children()))
    model.to(args.device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=True)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1)
    criterion = ADSH_Loss(code_length, args.gamma)

    num_retrieval = len(retrieval_dataloader.dataset) #len = train data
    U = torch.zeros(args.num_samples, code_length).to(args.device)
    B = torch.randn(num_retrieval, code_length).to(args.device)#每个初始化满足标准正态分布
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)#len = train data
    #print(len(retrieval_targets))
    cnn_losses, hash_losses, quan_losses = AverageMeter(), AverageMeter(), AverageMeter()
    start = time.time()
    best_mAP = 0
    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)#len = num samples
        S = (train_targets @ retrieval_targets.t() > 0).float() #num samples * train num
        # print(S[:1])
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        # print(r)
        S = S * (1 + r) - r
        # print(S[:1])
        # Training CNN model
        for epoch in range(args.max_epoch):
            cnn_losses.reset()
            hash_losses.reset()
            quan_losses.reset()
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:
                data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                optimizer.zero_grad()

                F, local_f = model(data)
                U[index, :] = F.data
                cnn_loss, hash_loss, quan_loss = criterion(F, B, S[index, :], sample_index[index])
                cnn_losses.update(cnn_loss.item())
                hash_losses.update(hash_loss.item())
                quan_losses.update(quan_loss.item())

                cnn_loss.backward()
                optimizer.step()
                # print(optimizer.param_groups[0]['lr'])
            logger.info('[epoch:{}/{}][cnn_loss:{:.6f}][hash_loss:{:.6f}][quan_loss:{:.6f}]'.format(epoch+1, args.max_epoch,
                        cnn_losses.avg, hash_losses.avg, quan_losses.avg))
        scheduler.step()
        # Update B
        expand_U = torch.zeros(B.shape).to(args.device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, args.gamma)

        logger.info('[iter:{}/{}][iter_time:{:.2f}]'.format(it+1, args.max_iter, time.time()-iter_start))

        if it%1==0:
            query_code = generate_code(model, query_dataloader, code_length, args.device)
            # print(len(query_dataloader))
            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                B,
                query_dataloader.dataset.get_onehot_targets().to(args.device),
                retrieval_targets,
                args.device,
                args.topk,
            )
            logger.info(
                '[iter:{}/{}][code_length:{}][mAP:{:.5f}]'.format(it + 1, args.max_iter, code_length,
                                                                                   mAP))
            if mAP > best_mAP:
                best_mAP = mAP
                ret_path = os.path.join('checkpoints', args.info, str(code_length))
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.pth'))
                torch.save(B.cpu(), os.path.join(ret_path, 'database_code.pth'))
                torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join(ret_path, 'query_targets.pth'))
                torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.pth'))
                torch.save(model.state_dict(), os.path.join(ret_path, 'model.pth'))

            logger.info('[iter:{}/{}][code_length:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(it+1, args.max_iter, code_length, mAP, best_mAP))
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))

    return best_mAP


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length]).to(device)
        for batch, (data, targets, index) in enumerate(dataloader):
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            hash_code, _ = model(data)
            code[index, :] = hash_code.sign()
    model.train()
    return code
