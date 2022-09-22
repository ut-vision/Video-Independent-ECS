import os, torch, argparse, sys, random, glob
from pathlib import Path
import numpy as np
import pandas as pd
import re
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(".")
from models.MSTCN.model import MS_TCN2
from models.voxdataloader import GazeFeatureDataloader
from models.MSTCN.evaluator import edit_score, f_score
from torch.utils.tensorboard import SummaryWriter

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(epoch):
    mse = nn.MSELoss(reduction='none')
    print("EPOCH", epoch, "\n")
    allLoss = []
    model.train()
    total_loss = 0
    n_correct = 0
    n_samples = 0
    n_fp = 0
    n_fn = 0
    n_tp = 0
    n_tn = 0
    for _, (features, labels) in enumerate(train_loader):  
        if args.cuda: features = features.cuda()
        if args.cuda: labels = labels.cuda()
        features = features.to(torch.float32)
        optimizer.zero_grad()
        output = model(features)
        print(output)
        labels = labels.long().reshape(-1,)
        loss = 0
        for p in output: # 5, 1, 2, num_frames
            loss += criterion(p.transpose(2, 1).contiguous().view(-1, 2), labels.view(-1))
            if args.mse: loss += 0.15*torch.mean(torch.clamp(mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
        
        loss.backward()    
        optimizer.step()
        total_loss += loss.item()
        output = output[-1]
        _, predicted = torch.max(output.data, 1)

        n_fp += ((predicted == 1) & (labels == 0)).sum().item()
        n_fn += ((predicted == 0) & (labels == 1)).sum().item()
        n_tp += ((predicted == 1) & (labels == 1)).sum().item()
        n_tn += ((predicted == 0) & (labels == 0)).sum().item()
        n_correct += (n_tp + n_tn)
        n_samples += (n_tp + n_tn + n_fp + n_fn)

    print('Train Epoch Finished, average loss : {:6f}\n'.format(total_loss / len(train_loader)))
    allLoss.append(total_loss / len(train_loader))
    writer.add_scalar('Loss/Train', total_loss / len(train_loader), epoch)
    acc = 100.0 * n_correct / n_samples
    print('Train set Acc.: {:.2f}%\n'.format(acc))
    writer.add_scalar('Accuracy/Train', acc, epoch)

def validate():
    mse = nn.MSELoss(reduction='none')
    model.eval()
    total_loss = 0
    n_correct = 0
    n_samples = 0
    n_fp = 0
    n_fn = 0
    n_tp = 0
    n_tn = 0
    score_query = [0.1, 0.25, 0.5]
    f1tp = [0, 0, 0]
    f1fp = [0, 0, 0]
    f1fn = [0, 0, 0]
    edit = 0 

    with torch.no_grad():
        for i, (features, labels) in enumerate(val_loader):  
            if args.cuda: features = features.cuda()
            if args.cuda: labels = labels.cuda()
            features = features.to(torch.float32)
            labels = labels.long().reshape(-1,)
            
            output = model(features)
            test_loss = 0
            for p in output:
                test_loss += criterion(p.transpose(2, 1).contiguous().view(-1, 2), labels.view(-1))
                if args.mse: test_loss += 0.15*torch.mean(torch.clamp(mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))

            total_loss += test_loss.item()
            output = output[-1]
            _, predicted = torch.max(output.data, 1)
            
            n_fp += ((predicted == 1) & (labels == 0)).sum().item()
            n_fn += ((predicted == 0) & (labels == 1)).sum().item()
            n_tp += ((predicted == 1) & (labels == 1)).sum().item()
            n_tn += ((predicted == 0) & (labels == 0)).sum().item()
            n_correct += (n_tp + n_tn)
            n_samples += (n_tp + n_tn + n_fp + n_fn)

            predicted_cpu = predicted.view(-1,).data.tolist()
            labels_cpu = labels.tolist()
            for i in range(len(labels_cpu)):
                if labels_cpu[i] == -1: labels_cpu[i] = predicted_cpu[i]

            edit += edit_score(predicted_cpu, labels_cpu)
            
            for i in range(len(score_query)):
                tp, fp, fn = f_score(predicted_cpu, labels_cpu, score_query[i])
                f1tp[i] += tp
                f1fp[i] += fp
                f1fn[i] += fn

    avgLoss = total_loss / len(val_loader)
    acc = 100.0 * n_correct / n_samples
    edit = edit / len(val_loader)
    for s in range(len(score_query)):
        precision = f1tp[s] / float(f1tp[s] + f1fp[s])
        recall = f1tp[s] / float(f1tp[s] + f1fn[s])

        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (score_query[s], f1))
    
    print('Val set: Average loss: {:.6f}, Acc.: {:.2f}%, edit: {:.2f}\n'.format(avgLoss, acc, edit))
    return avgLoss, acc

def test():
    mse = nn.MSELoss(reduction='none')
    model.eval()
    total_loss = 0
    n_correct = 0
    n_samples = 0
    score_query = [0.1, 0.25, 0.5]
    f1tp = [0, 0, 0]
    f1fp = [0, 0, 0]
    f1fn = [0, 0, 0]
    f1s = []
    edit = 0 

    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):  
            if args.cuda: features = features.cuda()
            if args.cuda: labels = labels.cuda()
            features = features.to(torch.float32)
            labels = labels.long().reshape(-1,)
            
            output = model(features)
            test_loss = 0
            for p in output:
                test_loss += criterion(p.transpose(2, 1).contiguous().view(-1, 2), labels.view(-1))
                if args.mse: test_loss += 0.3*torch.mean(torch.clamp(mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))

            total_loss += test_loss.item()

            output = output[-1]
            _, predicted = torch.max(output.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels.view(1, -1)).sum().item()

            predicted_cpu = predicted.view(-1,).data.tolist()
            labels_cpu = labels.tolist()

            edit += edit_score(predicted_cpu, labels_cpu)
            
            for i in range(len(score_query)):
                tp, fp, fn = f_score(predicted_cpu, labels_cpu, score_query[i])
                f1tp[i] += tp
                f1fp[i] += fp
                f1fn[i] += fn

    avgLoss = total_loss / len(test_loader)
    acc = 100.0 * n_correct / n_samples
    edit = edit / len(test_loader)
    for s in range(len(score_query)):
        precision = f1tp[s] / float(f1tp[s] + f1fp[s])
        recall = f1tp[s] / float(f1tp[s] + f1fn[s])
        f1 = 2.0 * (precision*recall) / (precision+recall)
        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (score_query[s], f1))
        f1s.append(f1)
    
    print('\nTest set: Average loss: {:.6f}, Acc.: {:.2f}%, edit: {:.2f}\n'.format(avgLoss, acc, edit))
    return avgLoss, acc, edit, f1s


def rePL(model):
    model.eval()
    print("Generating new PLs for the new global epoch")
    csvPath = "voxdataset_train.csv"
    dpath = "<path to the tracklet folder>" # NEED to change these two lines if enviroment changes.
    df = pd.read_csv(csvPath)
    for _, row in df.iterrows():
        vid = row['id']
        vname = row['video']
        trackletid = row['tracklet']
        x = os.path.join(dpath, vid, "tracklets", vname, trackletid)
        x = np.load(x)["gaze_feature"].T
        ypath = os.path.join(dpath, vid, "tracklets", vname, "processed", "iter_" + trackletid)
        x = torch.from_numpy(x).to(torch.float32).cuda().view(1, 2048, -1)

        with torch.no_grad():
            output = model(x)
        output = output[-1] 
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu().data.numpy()
        predicted = predicted.reshape(-1)

        np.savez_compressed(ypath, label = predicted)
    print("Done! Proceed to training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCN for VoxCeleb2')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='batch size (default: 1) (mostly has to be 1)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (default: 0.0)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit (default: 50)')
    parser.add_argument('--ksize', type=int, default=3,
                        help='kernel size (default: 3)')
                      
    parser.add_argument('--num_layers_PG', type=int, default=11,
                        help='# of PG layers (default: 11)')
    parser.add_argument('--num_layers_R', type=int, default=10,
                        help='# of R layers (default: 10)')
    parser.add_argument('--num_R', type=int, default=4,
                        help='# of R stages (default: 4)')
    parser.add_argument('--num_f_maps', type=int, default=64,
                        help='# of feature maps (default: 64)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate (default: 5e-4)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=10, metavar='T',
                        help='Number of dataloader works. Default: 10')   
    parser.add_argument('--finetune', action='store_true',
                        help='Option for fintuning on MPII dataset') 
    parser.add_argument('--eval', action='store_true',
                        help='Option for taking basePath model for evaluation only') 

    parser.add_argument('--basePath', type=str, default="models/TorchModels/temp.pth",
                        help='the pretrained model path. Effective when finetune option is set') 

    parser.add_argument('--boardname', type=str, default="temp_run",
                        help='name for tensorboard session') 
    parser.add_argument('--mse', action='store_true',
                        help='Option for adding truncated mse loss') 
    parser.add_argument('--stack', action='store_true',
                        help='Iterative training of models. When combined with --finetune, you are allowed to select your model from previous iteration by renaming it as temp.pth') 

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    n_classes = 2
    batch_size = args.batch_size
    epochs = args.epochs

    print(args)

    input_channels = 2048

    trainingSet = GazeFeatureDataloader("Training")
    valSet = GazeFeatureDataloader("Validation")
    testSet = GazeFeatureDataloader("Test")

    train_loader = DataLoader(trainingSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = MS_TCN2(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, input_channels, n_classes)
    if args.finetune:
        model.load_state_dict(torch.load(args.basePath))

    if args.cuda: model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index= -1)
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    if args.finetune: suffix = "finetuned"
    else: suffix = ""

    now = re.sub("[^0-9]", "", str(datetime.now()))
    Path(os.path.join("TCN", "TorchModels", args.boardname)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", args.boardname))

    allLoss = []
    allAcc = []
    
    if args.stack:
        if args.finetune:
            print("Testing current model")
            tloss, acc, edit, f1 = test()
            rePL(model)
        
            trainingSet = GazeFeatureDataloader("Training", stacked = True)
            valSet = GazeFeatureDataloader("Validation", stacked = True)
            train_loader = DataLoader(trainingSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
            model = MS_TCN2(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, input_channels, n_classes)
            if args.cuda: model.cuda()
            criterion = nn.CrossEntropyLoss(ignore_index= -1)
            optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


        for global_epoch in range(0, 5):
            for ep in range(epochs * global_epoch + 1, epochs * (global_epoch + 1) + 1):

                train(ep)
                tloss, acc = validate()
                    
                torch.save(model.state_dict(), os.path.join("TCN", "TorchModels", args.boardname, suffix + "epoch_" + str(ep) + ".pth"))
                
                allLoss.append(tloss)
                allAcc.append(acc)
                writer.add_scalar('Loss/Val', tloss, ep)
                writer.add_scalar('Accuracy/Val', acc, ep)

                tloss, acc, edit, f1 = test()
                writer.add_scalar('Loss/Test', tloss, ep)
                writer.add_scalar('Accuracy/Test', acc, ep)
                writer.add_scalar('edit/Test', edit, ep)
                writer.add_scalar('F1@10/Test', f1[0], ep)
                writer.add_scalar('F1@25/Test', f1[1], ep)
                writer.add_scalar('F1@50/Test', f1[2], ep)
            
            # generate pseduo labels again
            rePL(model)
            # reinitialize model, dataloader and continue training from scratch
            trainingSet = GazeFeatureDataloader("Training", stacked = True)
            valSet = GazeFeatureDataloader("Validation", stacked = True)
            train_loader = DataLoader(trainingSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
            model = MS_TCN2(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, input_channels, n_classes)
            if args.cuda: model.cuda()
            criterion = nn.CrossEntropyLoss(ignore_index= -1)
            optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
            # then new global epoch ....
    else:
        for ep in range(1, epochs+1):
            if args.finetune: MPIItrain(ep)
            else: train(ep)

            if args.finetune: tloss, acc = MPIIevaluate()
            else: tloss, acc = validate()
                
            torch.save(model.state_dict(), os.path.join("TCN", "TorchModels", args.boardname, suffix + "epoch_" + str(ep) + ".pth"))

            allLoss.append(tloss)
            allAcc.append(acc)
            writer.add_scalar('Loss/Val', tloss, ep)
            writer.add_scalar('Accuracy/Val', acc, ep)

            tloss, acc, edit, f1 = test()
            writer.add_scalar('Loss/Test', tloss, ep)
            writer.add_scalar('Accuracy/Test', acc, ep)
            writer.add_scalar('edit/Test', edit, ep)
            writer.add_scalar('F1@10/Test', f1[0], ep)
            writer.add_scalar('F1@25/Test', f1[1], ep)
            writer.add_scalar('F1@50/Test', f1[2], ep)
    writer.close()
