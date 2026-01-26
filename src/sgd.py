from os import makedirs

import torch
from torch.nn.utils import parameters_to_vector
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader

import argparse

from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset, make_batch_stepper, compute_rayleigh_quotient, estimate_batch_sharpness
from data import load_dataset, take_first, DATASETS


def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, batch_size: int, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0, wd: float =0, resume_model=None, eval_freq=250, bs_freq=10, max_bs_batches=500, cautious=False, decoupled=False):
    print(f'wd:{wd}')

    directory = get_gd_directory(dataset, lr, arch_id, seed, "sgd", loss, wd, beta)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)
    next_train_batch = make_batch_stepper(train_dataset, batch_size=batch_size, shuffle=True)
    sharp_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_architecture(arch_id, dataset).to(device)

    if resume_model is not None:
        print(f"Loading pretrained model weights from: {resume_model}")
        checkpoint = torch.load(resume_model, map_location=device)
        network.load_state_dict(checkpoint)

    # torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    if cautious or decoupled:
        wdd = 0.0
    else:
        wdd = wd
    optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta, wdd, cautious)

    # train_loss, test_loss, train_acc, test_acc = \
    #     torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    def n_points(max_steps, freq):
    # number of times step%freq==0 for step in [0, max_steps-1]
        return (max_steps - 1) // freq + 1

    train_loss = torch.zeros(n_points(max_steps, eval_freq)) 
    test_loss = torch.zeros(n_points(max_steps, eval_freq))
    train_acc = torch.zeros(n_points(max_steps, eval_freq))
    test_acc = torch.zeros(n_points(max_steps, eval_freq))

    iterates = torch.zeros(n_points(max_steps, iterate_freq), len(projectors)) if iterate_freq > 0 else torch.zeros(0, len(projectors))
    eigs     = torch.zeros(n_points(max_steps, eig_freq), neigs)               if eig_freq > 0 else torch.zeros(0, neigs)
    kappa    = torch.zeros(n_points(max_steps, eig_freq))                      if eig_freq > 0 else torch.zeros(0)
    bs       = torch.zeros(n_points(max_steps, bs_freq))                       if bs_freq > 0 else torch.zeros(0)


    for step in range(0, max_steps):
        if step % eval_freq ==0: 
            train_loss[step // eval_freq], train_acc[step // eval_freq] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                            physical_batch_size)
            test_loss[step // eval_freq], test_acc[step // eval_freq] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)
            print(f"{step}\t{train_loss[step //eval_freq]:.3f}\t{train_acc[step // eval_freq]:.3f}\t{test_loss[step // eval_freq]:.3f}\t{test_acc[step // eval_freq]:.3f}")

        if step % bs_freq == 0:
            X, Y = train_dataset.tensors
            bs[step // bs_freq] = estimate_batch_sharpness(network, X, Y, loss_fn, batch_size)
            print( "Batch sharpness", bs[step // bs_freq])

        if eig_freq != -1 and step % eig_freq == 0:
           
            evals, evecs = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)                                                 
            eigs[step // eig_freq, :] = evals
            kappa[step // eig_freq] = cosine_similarity(evecs[:, 0], parameters_to_vector(network.parameters()).cpu().detach(), dim=0).item()

            print("eigenvalues: ", eigs[step // eig_freq, :])
            print("kappa: ", kappa[step // eig_freq])

            


        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        
        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                   ("bs", bs[:step // bs_freq]),
                                   ("train_loss", train_loss[:step // eval_freq]), ("test_loss", test_loss[:step // eval_freq]),
                                   ("train_acc", train_acc[:step // eval_freq]), ("test_acc", test_acc[:step // eval_freq])])

        

        if (loss_goal != None and train_loss[step // eval_freq] < loss_goal) or (acc_goal != None and train_acc[step // eval_freq] > acc_goal):
            break
        
        network.train()
        optimizer.zero_grad()
        X, y = next_train_batch()  # ONE minibatch per step
        B = X.size(0)
        loss = loss_fn(network(X), y) / B   # loss_fn is SUM reduction
        loss.backward()
        old = {}
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        old[p] = p.detach().clone()

        optimizer.step()

        #Cautious wd!
        if wd != 0 and (cautious or decoupled):
            with torch.no_grad():
                for group in optimizer.param_groups:
                    lr = group["lr"]
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        # cautious mask based on update direction (grad) since momentum=0
                        w_old = old[p]
                        if cautious:
                            mask = (w_old * p.grad) > 0
                            p.add_(w_old * mask.to(p.dtype), alpha=-lr * wd)
                        elif decoupled:
                            p.add(w_old, alpha=-lr*wd)
                            print('THIS WORKS')
                        

    num_eigs = (step // eig_freq) + 1
    save_files_final(directory,
                     [("eigs", eigs[:num_eigs]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("bs", bs),
                      ("train_loss", train_loss), ("test_loss", test_loss),
                      ("train_acc", train_acc), ("test_acc", test_acc),
                      ("kappa", kappa[:num_eigs])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("wd", type=float, help="the weight decay of the GD")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)")
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--eval_freq", type=str, default=250)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="SGD minibatch size (used if mode=minibatch)")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model,
         wd=args.wd, beta=args.beta, nproj=args.nproj, loss_goal=args.loss_goal,
         acc_goal=args.acc_goal, abridged_size=args.abridged_size, seed=args.seed)
