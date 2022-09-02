import torch

import data
import model
import loss
import option
# from trainer.trainer_swint_hsa_nsf_offical import Trainer_SWINT_HSA_NSF
from trainer.trainer_swint_hsa_nsf import Trainer_SWINT_HSA_NSF
from logger import logger

args = option.args
torch.manual_seed(args.seed)
chkp = logger.Logger(args)

if args.task == 'VideoDeblur':
    print("Selected task: {}".format(args.task))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = Trainer_SWINT_HSA_NSF(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()
else:
    raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

chkp.done()
