# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Python version: 3.6


# import os
# import copy
# import time
# import pickle
# import numpy as np

# import torch
# # from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter


# from options import args_parser
# from update import LocalUpdate
# from utils import test_inference
# from models import CNNMnist, CNNFashion_Mnist, CNNCifar
# from utils import average_weights, exp_details
# from datasets_secure import get_train_dataset, get_test_dataset

# # from opacus.dp_model_inspector import DPModelInspector
# from opacus.utils import module_modification
# from mpi4py import MPI
# import random
# import seal
# from seal import ChooserEvaluator, \
# 	Ciphertext, \
# 	Decryptor, \
# 	Encryptor, \
# 	EncryptionParameters, \
# 	Evaluator, \
# 	IntegerEncoder, \
# 	FractionalEncoder, \
# 	KeyGenerator, \
# 	MemoryPoolHandle, \
# 	Plaintext, \
# 	SEALContext, \
# 	EvaluationKeys, \
# 	GaloisKeys, \
# 	PolyCRTBuilder, \
# 	ChooserEncoder, \
# 	ChooserEvaluator, \
# 	ChooserPoly

# from itertools import islice, chain, repeat
# def chunk_pad(it, size, padval=None):
#     it = chain(iter(it), repeat(padval))
#     return iter(lambda: tuple(islice(it, size)), (padval,) * size)

# def send_enc():
#     list_params = []
#     for param_tensor in model.state_dict():
#         list_params += model.state_dict()[param_tensor].flatten().tolist()
#     length = len(list_params)
#     list_params_int = [0]*length
#     for h in range(length):
#         # Convert float values to integer so that they can be encrypted
#         # length_integer is the maximum number of digits in the integer representation.
#         # The returned value min_prec is the min number of dec places before first non-zero digit in float value.
#         list_params_int[h] = round(list_params[h],3)*1000
#         #length_integer = 3
#         #list_params_int[h], min_prec = conv_int(list_params[h], length_integer)
#     slot_count = int(crtbuilder.slot_count())
#     pod_iter = iter(list_params_int)
#     pod_sliced = list(chunk_pad(pod_iter, slot_count, 0))  # partitions the vector pod_vec into chunks of size equal to the number of batching slots
#     for h in range(len(pod_sliced)):
#         pod_sliced[h] = [int(pod_sliced[h][j]) for j in range(len(pod_sliced[h]))]
#         for j in range(len(pod_sliced[h])):
#             if pod_sliced[h][j] < 0:
#                 pod_sliced[h][j] = parms.plain_modulus().value() + pod_sliced[h][j]
#     comm.send(len(pod_sliced), dest = 0, tag = 1)
#     for chunk in range(len(pod_sliced)):
#         encrypted_vec = Ciphertext()
#         plain_vector = Plaintext()
#         crtbuilder.compose(list(pod_sliced[chunk]), plain_vector)
#         encryptor.encrypt(plain_vector, encrypted_vec)
#         comm.send(encrypted_vec, dest = 0, tag=chunk+2)


# def recv_enc(idx_local):
#     num_chunks = comm.recv(source=idx_local, tag=1)  # receive the number of chunks to be sent by the worker
#     list_params_enc = []
#     for chunk in range(num_chunks):
#         list_params_enc += [comm.recv(source = idx_local, tag=chunk+2)]
#     return list_params_enc

# def average_weights_enc():
#     list_sums = []
#     for j in range(len(local_weights[0])):
#         enc_sum = Ciphertext()
#         list_columns = [local_weights[h][j] for h in range(len(local_weights))]  # length of list_columns is equal to the number of active workers
#         evaluator.add_many(list_columns,enc_sum)
#         list_sums.append(enc_sum)
#     return list_sums   # element-wise sum of encrypted weight vectors received from active workers

# def dec_recompose(enc_wts): # function to decrypt the aggregated weight vector received from parameter server
#     dec_wts = []
#     global_aggregate = {}
#     chunks = len(enc_wts)
#     for h in range(chunks):
#         plain_agg = Plaintext()
#         decryptor.decrypt(enc_wts[h], plain_agg)
#         crtbuilder.decompose(plain_agg)
#         dec_wts += [plain_agg.coeff_at(h) for h in range(plain_agg.coeff_count())]
#     for h in range(len(dec_wts)):
#         if dec_wts[h] > int(parms.plain_modulus().value() - 1) / 2:
#             dec_wts[h] = dec_wts[h] - parms.plain_modulus().value()
#     for h in range(len(dec_wts)):
#         dec_wts[h] = float(dec_wts[h])/(1000*m)
#     pos_start=0
#     for param_tensor in flattened_lengths:
#         pos_end = pos_start + flattened_lengths[param_tensor]
#         global_aggregate.update({param_tensor:torch.tensor(dec_wts[pos_start:pos_end]).reshape(shapes[param_tensor])})
#         pos_start=pos_end
#     return global_aggregate


# if __name__ == '__main__':
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     nworkers = comm.Get_size()-1
#     workers = list(range(1,nworkers+1))
#     start_time = time.time()
#     print("Ids of workers: ", str(workers))
#     print("Number of workers: ", str(nworkers))
#     ## Generate encryption parameters at worker 1 and share them with all workers.
#     # Share context object with the parameter server, so that it can aggregate encrypted values.
#     if rank==1:
#         # Initialize the parameters used for FHE
#         parms = EncryptionParameters()
#         parms.set_poly_modulus("1x^4096 + 1")
#         parms.set_coeff_modulus(seal.coeff_modulus_128(4096))  # 128-bit security
#         parms.set_plain_modulus(40961)
#         context = SEALContext(parms)
#         comm.send(context, dest=0, tag=0)  # send the context to the parameter server
#         print("Sent context to server")
#         for i in workers[1:]:
#             print("Sent context to worker ", str(i))
#             comm.send(context, dest=i, tag=0)  # send the context to the other workers

#         keygen = KeyGenerator(context)
#         public_key = keygen.public_key()
#         secret_key = keygen.secret_key()
#         for i in workers[1:]:
#             print("Sending keys to worker ", str(i))
#             comm.send(public_key, dest=i, tag=1)  # send the public key to worker i
#             comm.send(secret_key, dest=i, tag=2)  # send the secret key to worker i

#         encryptor = Encryptor(context, public_key)
#         decryptor = Decryptor(context, secret_key)
#         # Batching is done through an instance of the PolyCRTBuilder class so need
#         # to start by constructing one.
#         crtbuilder = PolyCRTBuilder(context)

#     elif rank==0:  # Parameter server
#         context = comm.recv(source=1, tag=0)  # parameter server receives only the context from worker 1
#         evaluator = Evaluator(context)

#     else: # Workers 2:num_users
#         # The workers receive the context, public key, and secret key from worker 1
#         context = comm.recv(source=1, tag=0)
#         parms = context.parms()
#         public_key = comm.recv(source=1, tag=1)
#         secret_key = comm.recv(source=1,tag=2)
#         encryptor = Encryptor(context, public_key)
#         decryptor = Decryptor(context, secret_key)
#         # Batching is done through an instance of the PolyCRTBuilder class so need
#         # to start by constructing one.
#         crtbuilder = PolyCRTBuilder(context)

#     # define paths
#     path_project = os.path.abspath('..')
#     logger = SummaryWriter('../logs')

#     args = args_parser()
#     exp_details(args)

#     if args.gpu:
#         torch.cuda.set_device(args.gpu)
#     device = 'cuda' if args.gpu else 'cpu'
#     m = max(int(args.frac * args.num_users), 1)  # Number of active workers in each round
#     # load datasets in each user. Note that the datasets are not loaded in the parameter server.
#     if rank >= 1:
#         train_dataset, user_groups = get_train_dataset(args,rank)  # user_groups consists of the ids of the data samples that belong to current worker

#     if rank == 0:
#         # BUILD MODEL
#         test_dataset = get_test_dataset(args)
#         if args.model == 'cnn':
#             # Convolutional neural network
#             if args.dataset == 'mnist':
#                 global_model = CNNMnist(args=args)
#             elif args.dataset == 'fmnist':
#                 global_model = CNNFashion_Mnist(args=args)
#             elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
#                 global_model = CNNCifar(args=args)
#         else:
#             exit('Error: unrecognized model')

#         ### DPSGD OPACUS ###
#         if args.withDP:
#             try:
#                 global_model = module_modification.convert_batchnorm_modules(global_model)
#                 # inspector = DPModelInspector()
#                 # inspector.validate(global_model)
#                 print("Model's already Valid!\n")
#             # except:
#             except Exception as e:
#                 # global_model = module_modification.convert_batchnorm_modules(global_model)
#                 # inspector = DPModelInspector()
#                 # print(f"Is the model valid? {inspector.validate(global_model)}")
#                 # print("Model is convereted to be Valid!\n")
#                 print("Warning: Could not convert BatchNorm layers:", e)


#         u_steps = np.zeros(args.num_users)
#         ###########

#         # Set the model to train and send it to device.
#         global_model.to(device)
#         global_model.train()
#         print(global_model)

#         # copy weights
#         global_weights = global_model.state_dict()

#         # Training
#         val_acc_list, net_list = [], []
#         cv_loss, cv_acc = [], []
#         print_every = 10
#         val_loss_pre, counter = 0, 0

#         for idx in range(1, args.num_users + 1):  # this loop sends the global model to all the workers
#             comm.send(global_model, dest=idx, tag=idx)

#         for epoch in range(args.epochs):
#             local_weights, local_losses = [], []
#             idxs_users = np.random.choice(range(1,args.num_users+1), m, replace=False)
#             print("active users: ", str(idxs_users))
#             for i in idxs_users:  # send the epoch values only to the active workers in the current round
#                 print("Sending epoch to active_user: ", str(i))
#                 comm.send(epoch, dest=i,tag=i)

#             if (epoch > 0) and (epoch < args.epochs -1):
#                 for idx in idxs_users:  # this loop sends the encrypted global aggregate to all active workers of the current round
#                     comm.send(global_weights, dest=idx, tag=idx)

#             #The receives are decoupled from the sends, ELSE the code will send global model, then wait for updates, and only then
#             #move to send the global model to the second worker. But we want the processing to be in parallel. Therefore, all the sends are
#             #implemented in a loop, and after that all the receives are implemented in a loop
#             for idx in idxs_users: # this loop receives the updates from the active workers
#                 u_steps[idx-1] = comm.recv(source=idx, tag=idx)
#                 print(idx)
#                 if epoch == args.epochs-1:
#                     local_weights.append(comm.recv(source=idx, tag=idx))  # receive unencrypted model update parameters from worker i
#                 elif epoch < args.epochs-1:
#                     local_weights.append(recv_enc(idx))  # receive encrypted model update parameters from worker i

#             #In the last epoch, the weights are received unencrypted,
#             #therefore, the weights are aggregated and the global model is updated
#             print("Length of local weights: ", str(len(local_weights)))
#             if epoch == args.epochs-1:
#                 # update global weights
#                 global_weights = average_weights(local_weights)  # Add the unencrypted weights received
#                 global_model.load_state_dict(global_weights)
#                 workers = list(range(1, nworkers + 1))
#                 print("Workers: ", str(workers))
#                 print("Active users in last round: ", str(idxs_users))
#                 for wkr in idxs_users:
#                     workers.remove(wkr)
#                 print("Residue workers: ", str(workers)) # Printing the ids of workers which are still listening for next round's communication.
#                 for i in workers:
#                     print("Sending exit signal to residue worker: ", str(i))
#                     comm.send(-1, dest=i, tag=i)
#                 break  # break out of the epoch loop.
#             elif epoch < args.epochs-1:
#                 # Add the encrypted weights
#                 global_weights = average_weights_enc()  # Add the encrypted weights received

#     elif rank >= 1:
#         local_model = LocalUpdate(args=args, dataset=train_dataset,
#                                   u_id=rank,
#                                   idxs=user_groups, logger=logger)
#         u_step=0
#         model = comm.recv(source=0, tag=rank)  # global model is received from the parameter server
#         print("Worker ", str(rank), " received global model")
#         flattened_lengths = {param_tensor: model.state_dict()[param_tensor].numel() for param_tensor in
#                              model.state_dict()}
#         shapes = {param_tensor: list(model.state_dict()[param_tensor].size()) for param_tensor in
#                   model.state_dict()}  # dictionary of shapes of tensors

#         while True:
#             epoch = comm.recv(source=0, tag=rank) # receive the epoch/communication round that the parameter server is in
#             if epoch == -1:
#                 break
#             #The server sends the latest weight aggregate to this worker,
#             #based on which the local model is updated before starting to train.
#             if (epoch < args.epochs-1) and (epoch > 0):
#                 enc_global_aggregate = comm.recv(source=0, tag=rank)
#                 ## decrypt and recompose enc_global_aggregate
#                 global_aggregate = dec_recompose(enc_global_aggregate)
#                 model.load_state_dict(global_aggregate)
#             u_step += 1
#             # Now perform one iteration
#             w, loss, u_step = local_model.update_weights(model=model, global_round=epoch,u_step=u_step)
#             comm.send(u_step, dest=0, tag=rank)  # send the step number
#             if epoch < args.epochs-1:
#                 send_enc()  # send encrypted model update parameters to the global agent
#             elif epoch == args.epochs-1:
#                 comm.send(w, dest=0, tag = rank)  # send unencrypted model update parameters to the global agent
#                 break

#     if rank==0:
#         # Test inference after completion of training
#         test_acc, test_loss = test_inference(args, global_model, test_dataset)

#         print(f' \n Results after {args.epochs} global rounds of training:')
#         print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

#         print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))




























































































































































# FLDP_secure_aggregation.py
"""
Robust MPI federated learning driver that auto-configures models/dataset specifics.
- Calls models with args (CNNMnist(args), etc.)
- Fills args.num_channels and args.num_classes automatically if missing
- Safe GPU parsing
- Workers always reply (prevents MPI deadlocks)
- Sends only CPU tensors over MPI
- Sends per-worker accuracy to server; server prints all round accuracies at the end
"""

import argparse
import sys
import os
import time
import traceback
from copy import deepcopy
from mpi4py import MPI
import torch
import numpy as np

# Try to import models.py (your file). It must define CNNMnist, CNNFashion_Mnist, CNNCifar
try:
    import models
    CNNMnist = models.CNNMnist
    CNNFashion_Mnist = models.CNNFashion_Mnist
    CNNCifar = models.CNNCifar
except Exception as e:
    print("Could not import models.py or expected classes not present:", e)
    raise

# Import LocalUpdate (must be in same folder)
try:
    from update import LocalUpdate
except Exception:
    print("Failed to import update.LocalUpdate. Ensure update.py exists in same folder.")
    raise

# Utilities: dataset loader + averaging
def get_dataset_and_groups(args):
    """
    Attempt to load dataset using user's utils (if present), otherwise fallback to torchvision.
    Auto-creates user_groups mapping.
    """
    # Auto-fill sensible defaults if missing
    model_name = getattr(args, "model", "mnist").lower()
    if model_name in ["mnist", "cnnmnist", "fashion-mnist", "fashion", "fashion_mnist", "fashion_mnist"]:
        default_channels = 1
        default_classes = getattr(args, "num_classes", 10)
        img_size = 28
    elif model_name in ["cifar", "cifar10", "cnncifar"]:
        default_channels = 3
        default_classes = getattr(args, "num_classes", 10)
        img_size = 32
    else:
        # Generic defaults
        default_channels = int(getattr(args, "num_channels", 1))
        default_classes = int(getattr(args, "num_classes", 10))
        img_size = 28

    # Set into args if absent
    if not hasattr(args, "num_channels") or args.num_channels is None:
        args.num_channels = default_channels
    if not hasattr(args, "num_classes") or args.num_classes is None:
        args.num_classes = default_classes

    # Try user utils
    try:
        from utils import get_dataset as user_get_dataset
        dataset, user_groups = user_get_dataset(args)
        return dataset, user_groups
    except Exception:
        # Fallback to torchvision datasets
        try:
            from torchvision import datasets, transforms
            transform = transforms.Compose([transforms.ToTensor()])
            if model_name.startswith("cifar"):
                dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
            elif "fashion" in model_name:
                dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
            else:
                dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

            # Partition dataset into num_users (simple equal partition)
            num_users = int(getattr(args, "num_users", 10))
            total = len(dataset)
            per = total // num_users
            user_groups = {i: list(range(i * per, (i + 1) * per)) for i in range(num_users)}
            # last user gets remainder
            if num_users * per < total:
                user_groups[num_users - 1].extend(list(range(num_users * per, total)))
            return dataset, user_groups
        except Exception as e:
            print("Failed to load fallback torchvision dataset:", e)
            raise

def average_weights(weight_list):
    """
    Average a list of state_dicts. Skip empty dicts.
    """
    if not weight_list:
        return {}
    # find first non-empty as base
    base = None
    for w in weight_list:
        if isinstance(w, dict) and len(w) > 0:
            base = w
            break
    if base is None:
        return {}

    avg = {}
    count = {}
    for k in base.keys():
        avg[k] = torch.zeros_like(base[k], dtype=torch.float64)
        count[k] = 0

    for w in weight_list:
        if not isinstance(w, dict) or len(w) == 0:
            continue
        for k, v in w.items():
            try:
                avg[k] += v.cpu().to(torch.float64)
                count[k] += 1
            except Exception:
                pass

    for k in list(avg.keys()):
        if count[k] > 0:
            avg[k] = (avg[k] / float(count[k])).type(base[k].dtype).to(base[k].device)
        else:
            avg[k] = base[k]
    return avg

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def _safe_gpu_parse(raw):
    try:
        if isinstance(raw, str):
            if raw.isdigit():
                return int(raw)
            if raw.startswith("cuda:"):
                return int(raw.split(":")[1])
            return -1
        else:
            return int(raw)
    except Exception:
        return -1

def server_loop(args):
    gpu_id = _safe_gpu_parse(getattr(args, "gpu", -1))
    device = torch.device("cuda" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
    print(f"[Server] Starting. Device={device}. MPI size={size}")

    train_dataset, user_groups = get_dataset_and_groups(args)
    num_users = len(user_groups)
    print(f"[Server] num_users={num_users}")

    # Instantiate global model with args
    # Choose model based on args.model
    model_choice = getattr(args, "model", "cnn").lower()
    if "cifar" in model_choice:
        global_model = CNNCifar(args)
    elif "fashion" in model_choice:
        global_model = CNNFashion_Mnist(args)
    else:
        global_model = CNNMnist(args)

    global_model.to(device)
    global_weights = deepcopy(global_model.state_dict())

    m = max(1, int(getattr(args, "frac", 0.1) * num_users))
    epochs = int(getattr(args, "epochs", 10))

    # Build primitive args to send (small)
    primitive_args = {
        "model": getattr(args, "model", "cnn"),
        "gpu": int(getattr(args, "gpu", -1)),
        "local_ep": int(getattr(args, "local_ep", 1)),
        "local_bs": int(getattr(args, "local_bs", 64)),
        "dp": bool(getattr(args, "dp", False)),
        "noise_multiplier": float(getattr(args, "noise_multiplier", 1.0)),
        "max_grad_norm": float(getattr(args, "max_grad_norm", 1.0)),
        "lr": float(getattr(args, "lr", 0.01)),
        "num_users": int(num_users),
        "num_channels": int(getattr(args, "num_channels", 1)),
        "num_classes": int(getattr(args, "num_classes", 10))
    }

    # Keep per-round aggregated accuracies
    round_accuracies = []

    for epoch in range(1, epochs + 1):
        print(f"\n[Server] Round {epoch}. Sampling {m} users.")
        selected_users = list(range(m))

        # send command to workers
        for dest in range(1, size):
            try:
                comm.send({
                    "cmd": "train_round",
                    "global_weights": global_weights,
                    "selected_users": selected_users,
                    "epoch": epoch,
                    "primitives": primitive_args
                }, dest=dest, tag=100)
            except Exception as e:
                print(f"[Server] Failed to send to worker {dest}: {e}")

        # collect replies (one per worker)
        collected = []
        responses = 0
        accs_received = []  # collect accuracies this round
        for _ in range(size - 1):
            try:
                status = MPI.Status()
                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                src = status.Get_source()
                responses += 1
            except Exception as e:
                print(f"[Server] MPI recv error: {e}")
                continue

            if not isinstance(data, dict):
                print(f"[Server] Unexpected message type from {src}: {type(data)}")
                continue

            weights = data.get("weights", {})
            report = data.get("report", "")
            user_id = data.get("user_id", None)
            loss = data.get("loss", None)
            acc = data.get("acc", None)

            if isinstance(weights, dict) and len(weights) > 0:
                collected.append(weights)
                print(f"[Server] Received weights from worker {src} (user {user_id}), loss={loss}, acc={acc}")
            else:
                print(f"[Server] Worker {src} sent empty weights (report={report}, acc={acc})")

            # store acc if present (none are skipped)
            if acc is not None:
                try:
                    accs_received.append(float(acc))
                except Exception:
                    pass

        # compute mean accuracy for this round (if any)
        round_acc_value = None
        if len(accs_received) > 0:
            round_acc_value = float(sum(accs_received) / len(accs_received))
            round_accuracies.append((epoch, round_acc_value))
        else:
            # append None to keep mapping
            round_accuracies.append((epoch, None))

        print(f"[Server] Received {responses} responses; aggregating {len(collected)} weight sets.")
        if len(collected) > 0:
            new_global = average_weights(collected)
            if new_global and len(new_global) > 0:
                global_weights = deepcopy(new_global)
                try:
                    global_model.load_state_dict(global_weights)
                except Exception as e:
                    print(f"[Server] Warning: load_state_dict failed: {e}")
                print(f"[Server] Global model updated at round {epoch}.")
            else:
                print("[Server] Aggregation returned empty result; skipping update.")
        else:
            print("[Server] No valid weights to aggregate this round.")

    # Print all round accuracies together
    print("\n===== FINAL ROUND ACCURACIES =====")
    for r, a in round_accuracies:
        if a is None:
            print(f"Round {r}: N/A")
        else:
            try:
                print(f"Round {r}: {a:.2f}%")
            except Exception:
                print(f"Round {r}: {a}")
    print("==================================\n")

    # shutdown workers
    for dest in range(1, size):
        try:
            comm.send({"cmd": "shutdown"}, dest=dest, tag=999)
        except Exception:
            pass

    print("[Server] Done.")

def build_args_from_primitives(prims):
    ns = argparse.Namespace()
    for k, v in prims.items():
        setattr(ns, k, v)
    # safe gpu parse
    ns.gpu = _safe_gpu_parse(getattr(ns, "gpu", -1))
    return ns

def worker_loop():
    rank_worker = comm.Get_rank()
    print(f"[Worker {rank_worker}] Starting.")

    assigned_user = None
    local_updater = None
    local_model = None
    u_step = 0

    while True:
        try:
            status = MPI.Status()
            msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        except Exception as e:
            print(f"[Worker {rank_worker}] recv failed: {e}")
            traceback.print_exc()
            try:
                comm.send({"u_step": u_step, "weights": {}, "loss": None, "user_id": assigned_user, "report": f"recv_failed:{e}"}, dest=0)
            except Exception:
                pass
            continue

        if not isinstance(msg, dict):
            print(f"[Worker {rank_worker}] Unexpected message type, skipping.")
            continue

        cmd = msg.get("cmd", None)
        if cmd == "shutdown":
            print(f"[Worker {rank_worker}] Shutdown received.")
            break

        if cmd == "train_round":
            epoch = msg.get("epoch", 0)
            selected_users = msg.get("selected_users", [])
            global_weights = msg.get("global_weights", None)
            primitives = msg.get("primitives", {})

            args = build_args_from_primitives(primitives)
            train_dataset, user_groups = get_dataset_and_groups(args)
            num_users = len(user_groups)

            # assign a user to this worker (deterministic)
            if assigned_user is None:
                assigned_user = (rank_worker - 1) % max(1, num_users)
                local_updater = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups.get(assigned_user, []),
                                            logger=None, u_id=assigned_user)

            if assigned_user not in selected_users:
                try:
                    comm.send({"u_step": u_step, "weights": {}, "loss": None, "acc": None, "user_id": assigned_user, "report": "not_selected"}, dest=0)
                except Exception:
                    pass
                continue

            # instantiate local model with args
            try:
                model_choice = getattr(args, "model", "cnn").lower()
                if "cifar" in model_choice:
                    local_model = CNNCifar(args)
                elif "fashion" in model_choice:
                    local_model = CNNFashion_Mnist(args)
                else:
                    local_model = CNNMnist(args)
                # load global weights if provided
                if global_weights is not None:
                    try:
                        local_model.load_state_dict(global_weights)
                    except Exception:
                        try:
                            local_model.load_state_dict(global_weights, strict=False)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[Worker {rank_worker}] Failed to prepare model: {e}")
                traceback.print_exc()
                try:
                    comm.send({"u_step": u_step, "weights": {}, "loss": None, "acc": None, "user_id": assigned_user, "report": f"model_prep_fail:{e}"}, dest=0)
                except Exception:
                    pass
                continue

            # Run local update
            try:
                weights, loss, u_step = local_updater.update_weights(model=local_model, global_round=epoch, u_step=u_step)

                # Compute local accuracy on worker's train data (so server can log round accuracy)
                local_acc = None
                try:
                    # evaluate on local_updater.trainloader
                    local_model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for imgs, labs in local_updater.trainloader:
                            device = torch.device("cuda" if torch.cuda.is_available() and _safe_gpu_parse(getattr(args, "gpu", -1)) >= 0 else "cpu")
                            imgs = imgs.to(device)
                            labs = labs.to(device)
                            local_model.to(device)
                            outs = local_model(imgs)
                            _, preds = torch.max(outs, 1)
                            correct += (preds == labs).sum().item()
                            total += labs.size(0)
                    if total > 0:
                        local_acc = 100.0 * (correct / total)
                except Exception:
                    local_acc = None

                # Convert weights to CPU tensors
                weights_cpu = {}
                for k, v in weights.items():
                    try:
                        weights_cpu[k] = v.cpu()
                    except Exception:
                        try:
                            weights_cpu[k] = torch.tensor(v)
                        except Exception:
                            pass

                comm.send({
                    "u_step": u_step,
                    "weights": weights_cpu,
                    "loss": loss,
                    "acc": local_acc,
                    "user_id": assigned_user,
                    "report": "ok"
                }, dest=0)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Worker {rank_worker}] Exception during update_weights: {e}")
                print(tb)
                try:
                    comm.send({"u_step": u_step, "weights": {}, "loss": None, "acc": None, "user_id": assigned_user, "report": f"exception:{str(e)[:200]}", "traceback": tb}, dest=0)
                except Exception:
                    pass
                continue
        else:
            try:
                comm.send({"u_step": u_step, "weights": {}, "loss": None, "acc": None, "user_id": assigned_user, "report": "unknown_cmd"}, dest=0)
            except Exception:
                pass

    print(f"[Worker {rank_worker}] Exiting.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--gpu", default=0)  # accept string or int
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--local_ep", type=int, default=1)
    parser.add_argument("--local_bs", type=int, default=64)
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--noise_multiplier", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args() if rank == 0 else None

    if rank == 0:
        server_loop(args)
    else:
        worker_loop()

if __name__ == "__main__":
    main()

