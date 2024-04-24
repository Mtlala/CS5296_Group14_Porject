import os
import argparse
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.model import get_model
from utils.dataPartitioner import DataPartitioner

import timeit
import utils.allreduce as allreduce

device = "cpu"
torch.set_num_threads(4)
batch_size = 256
randomseed = 1234
np.random.seed(randomseed)

def partition_dataset(rank, size, normalize):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    dataset = datasets.CIFAR10(root="./datas", train=True,
                                transform=transform_train)

    bsz = int(batch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(rank)
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         num_workers=2,
                                         sampler=None,
                                         shuffle=True,
                                         pin_memory=True)
    return train_set, bsz

def test_model(model, test_loader, criterion, totoal_time, outputfile):
    
    fp = open(outputfile, "a")

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    
    print('\n================================================================')
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Train Time: {:.2f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), totoal_time))
    
    fp.write('\n================================================================\n')
    fp.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Train Time: {:.2f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), totoal_time))
    fp.close()

def run_with_allreduce(rank, size, model, maxIter):
    normalize = transforms.Normalize(
        mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    """set randomseed"""
    torch.manual_seed(randomseed)

    """set up data"""
    train_set, bsz = partition_dataset(rank, size, normalize)

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    test_set = datasets.CIFAR10(root="./datas", train=False,
                                download=True, transform=transform_test)
    bsz = int(batch_size / float(size))
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=bsz,
                                              shuffle=False,
                                              pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    """write output to file"""
    
    outputdir = os.path.join("outputs", f"allReduce_{model}_Nodes_{size}_Epoches_{maxIter}")
    outputfile = os.path.join(outputdir, f"allReduce_{model}_Nodes_{size}_Epoches_{maxIter}.part{rank}")

    fp = open(outputfile, "a")

    """set up model"""
    model = get_model(model)
    
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
     
    """start training"""
    total_time = 0
    
    for epoch in range(maxIter):
        
        # # training start from here
        running_loss = 0.0
        
        # remember to exit the train loop at end of the epoch
        for batch_idx, (data, target) in enumerate(train_set):

            start = timeit.default_timer()

            # zero the parameter gradients
            optimizer.zero_grad()       
            # forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            loss.backward()
            allreduce.average_gradients(model)
            optimizer.step()
            
            t = timeit.default_timer() - start
            total_time += t
            
            if batch_idx % 20 == 19:  # print every 20 mini-batches
                print('Rank[%d] Epoch[%d, %5d] loss: %.3f Time: %.3f s' % (rank, epoch + 1, batch_idx + 1, running_loss / 20, t))
                fp.write('Rank[%d] Epoch[%d, %5d] loss: %.3f Time: %.3f s\n' % (rank, epoch + 1, batch_idx + 1, running_loss / 20, t))
                running_loss = 0.0

    # # training stop
    fp.close()

    test_model(model, test_loader, criterion, total_time, outputfile)

def run_with_ddp(rank, size, model, maxIter):
    normalize = transforms.Normalize(
        mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    """set randomseed"""
    torch.manual_seed(randomseed)

    """set up data"""
    train_set, _ = partition_dataset(rank, size, normalize)

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    test_set = datasets.CIFAR10(root="./datas", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    """write output to file"""
    
    outputdir = os.path.join("outputs", f"DDP_{model}_Nodes_{size}_Epoches_{maxIter}")
    outputfile = os.path.join(outputdir, f"DDP_{model}_Nodes_{size}_Epoches_{maxIter}.part{rank}")
    
    fp = open(outputfile, "a")

    """set up model"""
    model = get_model(model)
    
    model.to(device)
    model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)

    """start training"""
    
    total_time = 0
    
    for epoch in range(maxIter):
        running_loss = 0.0
        
        # remember to exit the train loop at end of the epoch
        for batch_idx, (data, target) in enumerate(train_set):
            
            start = timeit.default_timer()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            t = timeit.default_timer() - start
            total_time += t
            
            if batch_idx % 20 == 19:  # print every 20 mini-batches
                print('Rank[%d] Epoch[%d, %5d] loss: %.3f Time: %.3f s' % (rank, epoch + 1, batch_idx + 1, running_loss / 20, t))
                fp.write('Rank[%d] Epoch[%d, %5d] loss: %.3f Time: %.3f s\n' % (rank, epoch + 1, batch_idx + 1, running_loss / 20, t))
                running_loss = 0.0

    # # training stop
    fp.close()

    test_model(model, test_loader, criterion, total_time, outputfile)

def init_process(rank, size, mode, model, master_ip, epoch, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = '5678'
    
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=size)
    
    if mode == 'DDP':
        run_with_ddp(rank, size, model, epoch)
    elif mode == 'allReduce':
        run_with_allreduce(rank, size, model, epoch)
  
if __name__ == "__main__":
    '''
        Get the arguments 
    '''
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training')
    parser.add_argument('--model', type=str, default='VGG11', help='CNN architecture')
    parser.add_argument("--master_ip", type=str, default='127.0.0.1')
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--mode", type=str, default='DDP')
    
    args = parser.parse_args()
    
    mode = args.mode
    size = args.nodes
    
    mp.set_start_method("spawn")  
    processes = []
    
    outputdir = os.path.join("outputs", f"{mode}_{args.model}_Nodes_{size}_Epoches_{args.epoch}")
    
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, mode, args.model, args.master_ip, args.epoch))
        p.start()
        
        processes.append(p)

    for p in processes:
        p.join()